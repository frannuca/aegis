using Aegis.Instruments;
using AnalyticalPricers;
using MCPricer.FX;
using RandomSimulator;
using NodaTime;

namespace MCPricer.Tests;

/// <summary>
/// Validates FXDigitalOptionMCPricer against the closed-form digital
/// (cash-or-nothing / asset-or-nothing) prices under Garman-Kohlhagen.
///
/// ── Analytical benchmark ─────────────────────────────────────────────────────
/// See <see cref="DigitalBlackScholes"/> for the formulas (rate = r_d, carry = r_f):
///
///   Cash-or-nothing  Call = e^{-r_d·T}·Q·Φ(d₂)     Put = e^{-r_d·T}·Q·Φ(−d₂)
///   Asset-or-nothing Call = S·e^{-r_f·T}·Φ(d₁)     Put = S·e^{-r_f·T}·Φ(−d₁)
///
/// ── Model-free digital parity ────────────────────────────────────────────────
/// Exactly one of {S(T) &gt; K} / {S(T) &lt; K} occurs (the boundary has probability
/// zero), so:
///   CashOrNothing(Call) + CashOrNothing(Put)   = e^{-r_d·T}·Q                 (= DF · payout)
///   AssetOrNothing(Call) + AssetOrNothing(Put) = S·e^{-r_f·T} = e^{-r_d·T}·F  (= DF · forward)
///
/// ── Confidence interval methodology ─────────────────────────────────────────
/// Same 5σ MC-vs-analytical bound as FXVanillaOptionMCPricerTests.
/// </summary>
public sealed class FXDigitalOptionMCPricerTests
{
    // EURUSD: spot = 1.10, K = 1.10 (ATM), T = 1y, σ = 20%, r_d (USD) = 5%, r_f (EUR) = 2%
    private const double S0     = 1.10;
    private const double K      = 1.10;
    private const double T      = 1.0;
    private const double Sigma  = 0.20;
    private const double Rd     = 0.05;
    private const double Rf     = 0.02;
    private const double Payout = 1.0;     // Q: $1 cash-or-nothing payout

    private const int Paths      = 100_000;
    private const int Steps      = 1;       // single step suffices for European payoffs
    private const int DefaultSeed = 42;

    private static readonly LocalDate ValuationDate = new(2026, 1, 1);
    private static readonly LocalDate CurveMaturity = new(2036, 1, 1);

    // ── Cash-or-nothing ────────────────────────────────────────────────────────

    [Fact]
    public void AtmCashOrNothingCall_MatchesAnalytical()
    {
        var (mc, analytical) = RunAndCompare(S0, K, T, Sigma, Rd, Rf, DigitalSettlementType.CashOrNothing, isCall: true);
        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    [Fact]
    public void AtmCashOrNothingPut_MatchesAnalytical()
    {
        var (mc, analytical) = RunAndCompare(S0, K, T, Sigma, Rd, Rf, DigitalSettlementType.CashOrNothing, isCall: false);
        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    [Fact]
    public void OtmCashOrNothingCall_MatchesAnalytical()
    {
        // K = 1.20 is ~9% OTM: low ITM probability, small price.
        var (mc, analytical) = RunAndCompare(S0, strike: 1.20, T, Sigma, Rd, Rf, DigitalSettlementType.CashOrNothing, isCall: true);
        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    [Fact]
    public void ItmCashOrNothingCall_MatchesAnalytical()
    {
        // K = 1.00 is ~9% ITM: high ITM probability, price close to DF·Q.
        var (mc, analytical) = RunAndCompare(S0, strike: 1.00, T, Sigma, Rd, Rf, DigitalSettlementType.CashOrNothing, isCall: true);
        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    // ── Asset-or-nothing ───────────────────────────────────────────────────────

    [Fact]
    public void AtmAssetOrNothingCall_MatchesAnalytical()
    {
        var (mc, analytical) = RunAndCompare(S0, K, T, Sigma, Rd, Rf, DigitalSettlementType.AssetOrNothing, isCall: true);
        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    [Fact]
    public void AtmAssetOrNothingPut_MatchesAnalytical()
    {
        var (mc, analytical) = RunAndCompare(S0, K, T, Sigma, Rd, Rf, DigitalSettlementType.AssetOrNothing, isCall: false);
        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    // ── Model-free digital put-call parity ─────────────────────────────────────

    [Fact]
    public void CashOrNothing_CallPlusPut_EqualsDiscountedPayout()
    {
        // Exactly one of {S(T) > K}/{S(T) < K} occurs ⇒ Call + Put = DF · Q,
        // independent of any model. Use the same cube to minimise cancellation noise.
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var mcCall = BuildPricer(S0, K, T, Sigma, Rd, Rf, DigitalSettlementType.CashOrNothing, true,  cube).Price(market);
        var mcPut  = BuildPricer(S0, K, T, Sigma, Rd, Rf, DigitalSettlementType.CashOrNothing, false, cube).Price(market);

        var mcSum  = mcCall.Price + mcPut.Price;
        var analyticalSum = Math.Exp(-Rd * T) * Payout;
        var seSum  = mcCall.StandardError + mcPut.StandardError;

        Assert.True(Math.Abs(mcSum - analyticalSum) < 5.0 * seSum,
            $"MC sum={mcSum:F6}  analytical={analyticalSum:F6}  5σ tol={5.0 * seSum:F6}");
    }

    [Fact]
    public void AssetOrNothing_CallPlusPut_EqualsDiscountedForward()
    {
        // Call + Put = S·e^{-r_f·T} = e^{-r_d·T}·F  (forward, discounted at r_d).
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var mcCall = BuildPricer(S0, K, T, Sigma, Rd, Rf, DigitalSettlementType.AssetOrNothing, true,  cube).Price(market);
        var mcPut  = BuildPricer(S0, K, T, Sigma, Rd, Rf, DigitalSettlementType.AssetOrNothing, false, cube).Price(market);

        var mcSum  = mcCall.Price + mcPut.Price;
        var analyticalSum = S0 * Math.Exp(-Rf * T);
        var seSum  = mcCall.StandardError + mcPut.StandardError;

        Assert.True(Math.Abs(mcSum - analyticalSum) < 5.0 * seSum,
            $"MC sum={mcSum:F6}  analytical={analyticalSum:F6}  5σ tol={5.0 * seSum:F6}");
    }

    // ── Zero volatility edge cases ─────────────────────────────────────────────

    [Fact]
    public void ZeroVol_DeepItmCashOrNothingCall_PaysDiscountedPayout()
    {
        // σ = 0: deterministic forward F = S·e^{(r_d-r_f)T} ≈ 1.133 > K = 1.00 (deep ITM).
        const double itmStrike = 1.00;
        var expected = Math.Exp(-Rd * T) * Payout;

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var mc     = BuildPricer(S0, itmStrike, T, sigma: 0.0, Rd, Rf, DigitalSettlementType.CashOrNothing, isCall: true, cube).Price(market);

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_DeepOtmCashOrNothingCall_PaysZero()
    {
        // σ = 0: forward F ≈ 1.133 < K = 1.30 (deep OTM) ⇒ never pays.
        const double otmStrike = 1.30;
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var mc     = BuildPricer(S0, otmStrike, T, sigma: 0.0, Rd, Rf, DigitalSettlementType.CashOrNothing, isCall: true, cube).Price(market);

        Assert.Equal(0.0, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_DeepItmAssetOrNothingCall_PaysDiscountedForward()
    {
        // σ = 0, deep ITM ⇒ payoff = F with certainty, so price = DF · F (= S·e^{-r_f·T}
        // mathematically, but computed here exactly as the pricer does — forward then
        // discount — to avoid an ULP mismatch from reassociating the two Math.Exp calls).
        const double itmStrike = 1.00;
        var forward  = S0 * Math.Exp((Rd - Rf) * T);
        var expected = Math.Exp(-Rd * T) * forward;

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var mc     = BuildPricer(S0, itmStrike, T, sigma: 0.0, Rd, Rf, DigitalSettlementType.AssetOrNothing, isCall: true, cube).Price(market);

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    // ── Input validation ───────────────────────────────────────────────────────

    [Fact]
    public void WrongOptionKind_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        var vanillaOpt = new Option
        {
            Underlying    = "EURUSD",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Vanilla       = new VanillaOption()
        };
        Assert.Throws<ArgumentException>(() =>
            new FXDigitalOptionMCPricer(vanillaOpt, ValuationDate, Sigma, cube));
    }

    [Fact]
    public void UnspecifiedSettlementType_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        var option = new Option
        {
            Underlying    = "EURUSD",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Digital       = new DigitalOption { Payout = Payout }   // SettlementType left UNSPECIFIED
        };
        Assert.Throws<ArgumentException>(() =>
            new FXDigitalOptionMCPricer(option, ValuationDate, Sigma, cube));
    }

    [Fact]
    public void ZeroPayout_CashOrNothing_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        var option = new Option
        {
            Underlying    = "EURUSD",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Digital       = new DigitalOption { Payout = 0.0, SettlementType = DigitalSettlementType.CashOrNothing }
        };
        Assert.Throws<ArgumentException>(() =>
            new FXDigitalOptionMCPricer(option, ValuationDate, Sigma, cube));
    }

    [Fact]
    public void ZeroStrike_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        Assert.Throws<ArgumentException>(() =>
            BuildPricer(S0, strike: 0.0, T, Sigma, Rd, Rf, DigitalSettlementType.CashOrNothing, true, cube));
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private (PricingResult mc, double analytical) RunAndCompare(
        double spot, double strike, double expiry, double sigma,
        double rd, double rf, DigitalSettlementType settlementType, bool isCall)
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(spot, rd, rf);
        var mc     = BuildPricer(spot, strike, expiry, sigma, rd, rf, settlementType, isCall, cube).Price(market);

        var analytical = settlementType == DigitalSettlementType.AssetOrNothing
            ? DigitalBlackScholes.AssetOrNothing(spot, strike, expiry, sigma, rd, rf, isCall)
            : DigitalBlackScholes.CashOrNothing(spot, strike, expiry, sigma, rd, rf, Payout, isCall);

        return (mc, analytical);
    }

    private static FXDigitalOptionMCPricer BuildPricer(
        double spot, double strike, double expiry, double sigma,
        double rd, double rf, DigitalSettlementType settlementType, bool isCall, SimulationCube cube)
    {
        var option = new Option
        {
            Underlying    = "EURUSD",
            Strike        = strike,
            ExpiryYears   = expiry,
            OptionType    = isCall ? OptionType.Call : OptionType.Put,
            ExerciseStyle = ExerciseStyle.European,
            Digital       = new DigitalOption { Payout = Payout, SettlementType = settlementType }
        };
        return new FXDigitalOptionMCPricer(option, ValuationDate, sigma, cube);
    }

    private static FxMarketData MakeMarket(double spot, double rd, double rf) =>
        new()
        {
            CurrencyPair = "EURUSD",
            Spot         = spot,
            DomesticRate = ZeroCurve.Flat(rd, CurveMaturity),
            ForeignRate  = ZeroCurve.Flat(rf, CurveMaturity),
            VolSurface   = new VolatilitySurface()
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
