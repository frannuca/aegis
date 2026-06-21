using Aegis.Instruments;
using AnalyticalPricers;
using MCPricer.Equity;
using RandomSimulator;
using NodaTime;

namespace MCPricer.Tests;

/// <summary>
/// Validates EQDigitalOptionMCPricer against the closed-form digital
/// (cash-or-nothing / asset-or-nothing) prices under Black-Scholes.
///
/// Structurally identical to FXDigitalOptionMCPricerTests with r_d → r,
/// r_f → q (see <see cref="DigitalBlackScholes"/>, rate = r, carry = q).
///
/// ── Model-free digital parity ────────────────────────────────────────────────
///   CashOrNothing(Call) + CashOrNothing(Put)   = e^{-r·T}·Q
///   AssetOrNothing(Call) + AssetOrNothing(Put) = S·e^{-q·T}
/// </summary>
public sealed class EQDigitalOptionMCPricerTests
{
    // Equity: spot = 100 (ATM), σ = 20%, r = 5%, q = 2%
    private const double S0     = 100.0;
    private const double K      = 100.0;
    private const double T      = 1.0;
    private const double Sigma  = 0.20;
    private const double R      = 0.05;
    private const double Q      = 0.02;
    private const double Payout = 1.0;     // cash-or-nothing payout, in settlement currency

    private const int Paths       = 100_000;
    private const int Steps       = 1;       // single step suffices for European payoffs
    private const int DefaultSeed = 42;

    private static readonly LocalDate ValuationDate = new(2026, 1, 1);
    private static readonly LocalDate CurveMaturity = new(2036, 1, 1);

    // ── Cash-or-nothing ────────────────────────────────────────────────────────

    [Fact]
    public void AtmCashOrNothingCall_MatchesAnalytical()
    {
        var (mc, analytical) = RunAndCompare(S0, K, T, Sigma, R, Q, DigitalSettlementType.CashOrNothing, isCall: true);
        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    [Fact]
    public void AtmCashOrNothingPut_MatchesAnalytical()
    {
        var (mc, analytical) = RunAndCompare(S0, K, T, Sigma, R, Q, DigitalSettlementType.CashOrNothing, isCall: false);
        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    [Fact]
    public void OtmCashOrNothingCall_MatchesAnalytical()
    {
        var (mc, analytical) = RunAndCompare(S0, strike: 120.0, T, Sigma, R, Q, DigitalSettlementType.CashOrNothing, isCall: true);
        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    [Fact]
    public void ItmCashOrNothingCall_MatchesAnalytical()
    {
        var (mc, analytical) = RunAndCompare(S0, strike: 80.0, T, Sigma, R, Q, DigitalSettlementType.CashOrNothing, isCall: true);
        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    // ── Asset-or-nothing ───────────────────────────────────────────────────────

    [Fact]
    public void AtmAssetOrNothingCall_MatchesAnalytical()
    {
        var (mc, analytical) = RunAndCompare(S0, K, T, Sigma, R, Q, DigitalSettlementType.AssetOrNothing, isCall: true);
        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    [Fact]
    public void AtmAssetOrNothingPut_MatchesAnalytical()
    {
        var (mc, analytical) = RunAndCompare(S0, K, T, Sigma, R, Q, DigitalSettlementType.AssetOrNothing, isCall: false);
        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    // ── Model-free digital put-call parity ─────────────────────────────────────

    [Fact]
    public void CashOrNothing_CallPlusPut_EqualsDiscountedPayout()
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, R, Q);
        var mcCall = BuildPricer(S0, K, T, Sigma, R, Q, DigitalSettlementType.CashOrNothing, true,  cube).Price(market);
        var mcPut  = BuildPricer(S0, K, T, Sigma, R, Q, DigitalSettlementType.CashOrNothing, false, cube).Price(market);

        var mcSum         = mcCall.Price + mcPut.Price;
        var analyticalSum = Math.Exp(-R * T) * Payout;
        var seSum         = mcCall.StandardError + mcPut.StandardError;

        Assert.True(Math.Abs(mcSum - analyticalSum) < 5.0 * seSum,
            $"MC sum={mcSum:F6}  analytical={analyticalSum:F6}  5σ tol={5.0 * seSum:F6}");
    }

    [Fact]
    public void AssetOrNothing_CallPlusPut_EqualsDiscountedSpot()
    {
        // Call + Put = S·e^{-q·T}
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, R, Q);
        var mcCall = BuildPricer(S0, K, T, Sigma, R, Q, DigitalSettlementType.AssetOrNothing, true,  cube).Price(market);
        var mcPut  = BuildPricer(S0, K, T, Sigma, R, Q, DigitalSettlementType.AssetOrNothing, false, cube).Price(market);

        var mcSum         = mcCall.Price + mcPut.Price;
        var analyticalSum = S0 * Math.Exp(-Q * T);
        var seSum         = mcCall.StandardError + mcPut.StandardError;

        Assert.True(Math.Abs(mcSum - analyticalSum) < 5.0 * seSum,
            $"MC sum={mcSum:F6}  analytical={analyticalSum:F6}  5σ tol={5.0 * seSum:F6}");
    }

    // ── Zero volatility edge cases ─────────────────────────────────────────────

    [Fact]
    public void ZeroVol_DeepItmCashOrNothingCall_PaysDiscountedPayout()
    {
        // σ = 0: forward F = S·e^{(r-q)T} ≈ 103.05 > K = 90 (deep ITM).
        const double itmStrike = 90.0;
        var expected = Math.Exp(-R * T) * Payout;

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, R, Q);
        var mc     = BuildPricer(S0, itmStrike, T, sigma: 0.0, R, Q, DigitalSettlementType.CashOrNothing, isCall: true, cube).Price(market);

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_DeepOtmCashOrNothingCall_PaysZero()
    {
        // σ = 0: forward F ≈ 103.05 < K = 120 (deep OTM) ⇒ never pays.
        const double otmStrike = 120.0;
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, R, Q);
        var mc     = BuildPricer(S0, otmStrike, T, sigma: 0.0, R, Q, DigitalSettlementType.CashOrNothing, isCall: true, cube).Price(market);

        Assert.Equal(0.0, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_DeepItmAssetOrNothingCall_PaysDiscountedSpot()
    {
        // σ = 0, deep ITM ⇒ payoff = F with certainty, so price = e^{-rT}·F (= S·e^{-qT}
        // mathematically, but computed here exactly as the pricer does — forward then
        // discount — to avoid an ULP mismatch from reassociating the two Math.Exp calls).
        const double itmStrike = 90.0;
        var forward  = S0 * Math.Exp((R - Q) * T);
        var expected = Math.Exp(-R * T) * forward;

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, R, Q);
        var mc     = BuildPricer(S0, itmStrike, T, sigma: 0.0, R, Q, DigitalSettlementType.AssetOrNothing, isCall: true, cube).Price(market);

        // precision: 8, not 10 — every path returns the *same* deterministic payoff
        // (~98), and summing ~10⁵ copies of a value of that magnitude accumulates
        // ~10⁻⁹ of floating-point rounding noise (vs ~10⁻¹³ for the ~0.03-magnitude
        // vanilla payoffs elsewhere, which is why those zero-vol tests use precision: 10).
        Assert.Equal(expected, mc.Price, precision: 8);
    }

    // ── Input validation ───────────────────────────────────────────────────────

    [Fact]
    public void WrongOptionKind_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        var vanillaOpt = new Option
        {
            Underlying    = "AAPL",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Vanilla       = new VanillaOption()
        };
        Assert.Throws<ArgumentException>(() =>
            new EQDigitalOptionMCPricer(vanillaOpt, ValuationDate, Sigma, cube));
    }

    [Fact]
    public void UnspecifiedSettlementType_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        var option = new Option
        {
            Underlying    = "AAPL",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Digital       = new DigitalOption { Payout = Payout }
        };
        Assert.Throws<ArgumentException>(() =>
            new EQDigitalOptionMCPricer(option, ValuationDate, Sigma, cube));
    }

    [Fact]
    public void ZeroPayout_CashOrNothing_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        var option = new Option
        {
            Underlying    = "AAPL",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Digital       = new DigitalOption { Payout = 0.0, SettlementType = DigitalSettlementType.CashOrNothing }
        };
        Assert.Throws<ArgumentException>(() =>
            new EQDigitalOptionMCPricer(option, ValuationDate, Sigma, cube));
    }

    [Fact]
    public void ZeroStrike_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        Assert.Throws<ArgumentException>(() =>
            BuildPricer(S0, strike: 0.0, T, Sigma, R, Q, DigitalSettlementType.CashOrNothing, true, cube));
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private (PricingResult mc, double analytical) RunAndCompare(
        double spot, double strike, double expiry, double sigma,
        double r, double q, DigitalSettlementType settlementType, bool isCall)
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(spot, r, q);
        var mc     = BuildPricer(spot, strike, expiry, sigma, r, q, settlementType, isCall, cube).Price(market);

        var analytical = settlementType == DigitalSettlementType.AssetOrNothing
            ? DigitalBlackScholes.AssetOrNothing(spot, strike, expiry, sigma, r, q, isCall)
            : DigitalBlackScholes.CashOrNothing(spot, strike, expiry, sigma, r, q, Payout, isCall);

        return (mc, analytical);
    }

    private static EQDigitalOptionMCPricer BuildPricer(
        double spot, double strike, double expiry, double sigma,
        double r, double q, DigitalSettlementType settlementType, bool isCall, SimulationCube cube)
    {
        var option = new Option
        {
            Underlying    = "AAPL",
            Strike        = strike,
            ExpiryYears   = expiry,
            OptionType    = isCall ? OptionType.Call : OptionType.Put,
            ExerciseStyle = ExerciseStyle.European,
            Digital       = new DigitalOption { Payout = Payout, SettlementType = settlementType }
        };
        return new EQDigitalOptionMCPricer(option, ValuationDate, sigma, cube);
    }

    private static EqMarketData MakeMarket(double spot, double r, double q) =>
        new()
        {
            Ticker        = "AAPL",
            Spot          = spot,
            RiskFreeRate  = ZeroCurve.Flat(r, CurveMaturity),
            DividendYield = ZeroCurve.Flat(q, CurveMaturity),
            VolSurface    = new VolatilitySurface()
        };

    private static void AssertWithinMcBounds(PricingResult mc, double analytical, double sigma)
    {
        var tolerance = sigma * mc.StandardError;
        Assert.True(
            Math.Abs(mc.Price - analytical) < tolerance,
            $"MC={mc.Price:F6}  Analytical={analytical:F6}  " +
            $"Diff={Math.Abs(mc.Price - analytical):F6}  " +
            $"{sigma}σ tol={tolerance:F6}  SE={mc.StandardError:F6}");
    }
}
