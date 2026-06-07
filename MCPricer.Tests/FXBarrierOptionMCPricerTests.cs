using Aegis.Instruments;
using AnalyticalPricers;
using MCPricer.FX;
using RandomSimulator;

namespace MCPricer.Tests;

/// <summary>
/// Validates FXBarrierOptionMCPricer via three complementary strategies:
///
///   1. KI + KO = Vanilla parity  (path-by-path identity, exact with same cube)
///   2. Zero-vol deterministic tests  (S(T) is known analytically; barrier hit or not)
///   3. Far-barrier limit  (barrier unreachable → matches Garman-Kohlhagen vanilla)
///
/// Analytical (Merton-Reiner-Rubinstein) formulas are not used directly because:
///   a) They require H ≤ min(S,K) or H ≥ max(S,K) case splits to implement safely.
///   b) The three strategies above give stronger coverage by validating the
///      barrier-detection logic, payoff switching, and limit behaviour independently.
///
/// ── KI + KO parity ───────────────────────────────────────────────────────────
/// For any path ω:
///   KnockIn_payoff(ω) + KnockOut_payoff(ω) = Vanilla_payoff(ω)   (rebate = 0)
///
/// Running all three on the same SimulationCube makes the identity exact at the
/// level of floating-point summation across paths.
///
/// ── Continuous vs discrete ordering ─────────────────────────────────────────
/// Continuous monitoring catches crossings between time steps; discrete misses them.
/// For UpAndOut (knock-out): more knockouts → lower price.
///   continuous_price ≤ discrete_price  (in expectation; verified with slack)
/// </summary>
public sealed class FXBarrierOptionMCPricerTests
{
    private const double S0     = 1.10;
    private const double K      = 1.10;
    private const double T      = 1.0;
    private const double Sigma  = 0.20;
    private const double Rd     = 0.05;
    private const double Rf     = 0.02;
    private const double H_Up   = 1.30;   // above spot
    private const double H_Down = 0.90;   // below spot

    // Pricers now take rates as Pillars zero-rate curves (term structures), interpolated
    // at the option's expiry from this valuation date — see ZeroCurve.InterpolateRate.
    // Tests use single-pillar "flat" curves so the interpolated rate equals the scalar
    // reference rate (Rd, Rf) regardless of T, keeping the analytical comparisons exact.
    private static readonly DateOnly ValuationDate = new(2026, 1, 1);
    private static readonly DateOnly CurveMaturity = new(2036, 1, 1);

    private const int Paths       = 100_000;
    private const int Steps       = 50;     // daily-ish monitoring for barriers
    private const int DefaultSeed = 42;

    // ── KI + KO = Vanilla parity ─────────────────────────────────────────────

    [Theory]
    [InlineData(true,  BarrierType.UpAndIn,   BarrierType.UpAndOut,   H_Up,   "up call")]
    [InlineData(false, BarrierType.UpAndIn,   BarrierType.UpAndOut,   H_Up,   "up put")]
    [InlineData(true,  BarrierType.DownAndIn, BarrierType.DownAndOut, H_Down, "down call")]
    [InlineData(false, BarrierType.DownAndIn, BarrierType.DownAndOut, H_Down, "down put")]
    public void KnockInPlusKnockOut_EqualsVanilla(
        bool isCall, BarrierType kiType, BarrierType koType, double h, string _)
    {
        var cube = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);

        var vanilla = BuildVanilla(S0, K, T, Sigma, Rd, Rf, isCall, cube).Price();
        var ki      = BuildBarrier(S0, K, T, Sigma, Rd, Rf, h, kiType,
                                   BarrierObservation.Discrete, isCall, rebate: 0.0, cube).Price();
        var ko      = BuildBarrier(S0, K, T, Sigma, Rd, Rf, h, koType,
                                   BarrierObservation.Discrete, isCall, rebate: 0.0, cube).Price();

        // Same cube → same path payoffs → KI + KO = Vanilla exactly up to float rounding.
        Assert.True(
            Math.Abs(ki.Price + ko.Price - vanilla.Price) < 1e-10,
            $"KI={ki.Price:F8} + KO={ko.Price:F8} = {ki.Price + ko.Price:F8}, vanilla={vanilla.Price:F8}");
    }

    // ── Zero-vol: UpAndOut, barrier never reached → vanilla intrinsic ─────────

    [Fact]
    public void ZeroVol_UpAndOut_BarrierNeverReached_PricesAtDiscountedIntrinsic()
    {
        // σ=0, rd > rf: S grows to S0·exp((rd-rf)·T) ≈ 1.133 < H_Up=1.30 → not knocked out
        var forward  = S0 * Math.Exp((Rd - Rf) * T);
        var expected = Math.Exp(-Rd * T) * Math.Max(forward - K, 0.0);

        var cube = SimulationCube.GenerateIndependent(1_000, 1, 1, DefaultSeed);
        var mc   = BuildBarrier(S0, K, T, 0.0, Rd, Rf, H_Up, BarrierType.UpAndOut,
                                BarrierObservation.Discrete, isCall: true, rebate: 0.0, cube).Price();

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    // ── Zero-vol: UpAndIn, barrier never reached → 0 (not activated) ─────────

    [Fact]
    public void ZeroVol_UpAndIn_BarrierNeverReached_PricesAtZero()
    {
        // Same deterministic path as above; UpAndIn requires barrier hit → not activated
        var cube = SimulationCube.GenerateIndependent(1_000, 1, 1, DefaultSeed);
        var mc   = BuildBarrier(S0, K, T, 0.0, Rd, Rf, H_Up, BarrierType.UpAndIn,
                                BarrierObservation.Discrete, isCall: true, rebate: 0.0, cube).Price();

        Assert.Equal(0.0, mc.Price, precision: 10);
    }

    // ── Zero-vol: DownAndOut, barrier always breached → rebate ───────────────

    [Fact]
    public void ZeroVol_DownAndOut_BarrierAlwaysBreached_PaysRebate()
    {
        // σ=0, rf > rd: S falls to S0·exp(-rf·T)=1.10·exp(-0.10)≈0.995 ≤ H=1.0 → knocked out
        const double h      = 1.0;
        const double rd0    = 0.0;
        const double rf0    = 0.10;
        const double rebate = 0.02;

        var cube = SimulationCube.GenerateIndependent(1_000, 1, 1, DefaultSeed);
        var mc   = BuildBarrier(S0, K, T, 0.0, rd0, rf0, h, BarrierType.DownAndOut,
                                BarrierObservation.Discrete, isCall: true, rebate, cube).Price();

        // rd=0 → discount factor = 1; every path is knocked out → pays rebate
        Assert.Equal(rebate, mc.Price, precision: 10);
    }

    // ── Zero-vol: DownAndIn, barrier always breached → vanilla intrinsic ──────

    [Fact]
    public void ZeroVol_DownAndIn_BarrierAlwaysBreached_PricesAtDiscountedIntrinsic()
    {
        // Same deterministic path; DownAndIn activated → pays put intrinsic
        const double h   = 1.0;
        const double rd0 = 0.0;
        const double rf0 = 0.10;

        var sT       = S0 * Math.Exp(-rf0 * T);            // ≈ 0.995
        var expected = Math.Max(K - sT, 0.0);               // put; rd=0 → DF=1

        var cube = SimulationCube.GenerateIndependent(1_000, 1, 1, DefaultSeed);
        var mc   = BuildBarrier(S0, K, T, 0.0, rd0, rf0, h, BarrierType.DownAndIn,
                                BarrierObservation.Discrete, isCall: false, rebate: 0.0, cube).Price();

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    // ── Barrier far from spot → matches vanilla (GK) ─────────────────────────

    [Fact]
    public void UpAndOut_BarrierVeryHigh_EqualsVanilla()
    {
        // H=100 is unreachable → UpAndOut never triggers → price = vanilla call
        var cube = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var mc   = BuildBarrier(S0, K, T, Sigma, Rd, Rf, barrierLevel: 100.0,
                                BarrierType.UpAndOut, BarrierObservation.Discrete,
                                isCall: true, rebate: 0.0, cube).Price();

        var gk = GarmanKohlhagen.Price(S0, K, T, Sigma, Rd, Rf, isCall: true);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void DownAndOut_BarrierVeryLow_EqualsVanilla()
    {
        // H=0.001 is unreachable → DownAndOut never triggers → price = vanilla call
        var cube = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var mc   = BuildBarrier(S0, K, T, Sigma, Rd, Rf, barrierLevel: 0.001,
                                BarrierType.DownAndOut, BarrierObservation.Discrete,
                                isCall: true, rebate: 0.0, cube).Price();

        var gk = GarmanKohlhagen.Price(S0, K, T, Sigma, Rd, Rf, isCall: true);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    // ── Rebate: monotone in rebate amount ─────────────────────────────────────

    [Fact]
    public void UpAndOut_RebateIncreasesPrice()
    {
        // A larger rebate on knockout increases the UpAndOut value
        var cube      = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var noRebate  = BuildBarrier(S0, K, T, Sigma, Rd, Rf, H_Up, BarrierType.UpAndOut,
                                     BarrierObservation.Discrete, isCall: true, rebate: 0.0,  cube).Price();
        var withRebate = BuildBarrier(S0, K, T, Sigma, Rd, Rf, H_Up, BarrierType.UpAndOut,
                                      BarrierObservation.Discrete, isCall: true, rebate: 0.05, cube).Price();

        Assert.True(withRebate.Price > noRebate.Price,
            $"No-rebate={noRebate.Price:F6}  With-rebate={withRebate.Price:F6}");
    }

    [Fact]
    public void UpAndIn_RebateIncreasesPrice()
    {
        // Rebate is paid when barrier is NOT hit (i.e., on non-activated paths)
        var cube      = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var noRebate  = BuildBarrier(S0, K, T, Sigma, Rd, Rf, H_Up, BarrierType.UpAndIn,
                                     BarrierObservation.Discrete, isCall: true, rebate: 0.0,  cube).Price();
        var withRebate = BuildBarrier(S0, K, T, Sigma, Rd, Rf, H_Up, BarrierType.UpAndIn,
                                      BarrierObservation.Discrete, isCall: true, rebate: 0.05, cube).Price();

        Assert.True(withRebate.Price > noRebate.Price,
            $"No-rebate={noRebate.Price:F6}  With-rebate={withRebate.Price:F6}");
    }

    // ── Continuous monitoring ≤ discrete (UpAndOut) ───────────────────────────

    [Fact]
    public void Continuous_UpAndOut_NotGreaterThanDiscrete()
    {
        // Continuous monitoring catches more crossings than discrete → more knockouts
        // → lower UpAndOut price. Verified with 5σ slack to absorb MC noise from the
        // separate bridge RNG draws used in continuous mode.
        var cube       = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var discrete   = BuildBarrier(S0, K, T, Sigma, Rd, Rf, H_Up, BarrierType.UpAndOut,
                                      BarrierObservation.Discrete, isCall: true, rebate: 0.0, cube).Price();
        var continuous = BuildBarrier(S0, K, T, Sigma, Rd, Rf, H_Up, BarrierType.UpAndOut,
                                      BarrierObservation.Continuous, isCall: true, rebate: 0.0, cube).Price();

        var tolerance = 5.0 * (discrete.StandardError + continuous.StandardError);
        Assert.True(continuous.Price <= discrete.Price + tolerance,
            $"Continuous={continuous.Price:F6}  Discrete={discrete.Price:F6}  5σ slack={tolerance:F6}");
    }

    // ── Greeks: barrier pricer returns finite delta/vega ─────────────────────

    [Fact]
    public void GreeksDelta_IsFinite_ForSpotAwayFromBarrier()
    {
        // With spot well below H_Up=1.30, the delta estimator is stable
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        using var pricer = BuildBarrier(S0, K, T, Sigma, Rd, Rf, H_Up, BarrierType.UpAndOut,
                                        BarrierObservation.Discrete, isCall: true, rebate: 0.0, cube);

        var greeks = pricer.ComputeGreeks(spotEps: 0.01, volEps: 0.001, rateEps: 0.0001);

        Assert.True(double.IsFinite(greeks.Delta), $"Delta={greeks.Delta}");
        Assert.True(double.IsFinite(greeks.Gamma), $"Gamma={greeks.Gamma}");
        Assert.True(double.IsFinite(greeks.Vega),  $"Vega={greeks.Vega}");
        Assert.True(double.IsFinite(greeks.Rho),   $"Rho={greeks.Rho}");
        Assert.True(double.IsNaN(greeks.Theta),    "Theta must be NaN (time-bump not supported)");
    }

    [Fact]
    public void GreeksDelta_UpAndOut_LessThanVanillaDelta()
    {
        // UpAndOut delta ≤ vanilla delta: the knock-out risk (lost payoff if barrier is hit)
        // partially offsets the usual positive delta. Close to the barrier the delta can even
        // be negative. This test validates the economic ordering rather than the sign.
        //
        // MC noise in barrier bump-and-reprice is high (SE of delta ≈ 0.07 for these params).
        // We therefore use a generous 5σ tolerance on the comparison.
        var cube = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);

        using var barrierPricer = BuildBarrier(S0, K, T, Sigma, Rd, Rf, H_Up, BarrierType.UpAndOut,
                                               BarrierObservation.Discrete, isCall: true, rebate: 0.0, cube);
        using var vanillaPricer = BuildVanilla(S0, K, T, Sigma, Rd, Rf, isCall: true,
                                               SimulationCube.GenerateIndependent(Paths, 1, 1, DefaultSeed));

        var barrierDelta = barrierPricer.ComputeGreeks(spotEps: 0.01).Delta;
        var vanillaDelta = vanillaPricer.ComputeGreeks(spotEps: 0.01).Delta;

        // Allow 5σ MC slack ≈ 5 × 0.07 (SE of each bump-and-reprice delta estimator)
        Assert.True(barrierDelta < vanillaDelta + 0.35,
            $"UpAndOut delta={barrierDelta:F4} must be < vanilla delta={vanillaDelta:F4} + 0.35 slack");
    }

    // ── Input validation ──────────────────────────────────────────────────────

    [Fact]
    public void ZeroBarrierLevel_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        Assert.Throws<ArgumentException>(() =>
            BuildBarrier(S0, K, T, Sigma, Rd, Rf, barrierLevel: 0.0, BarrierType.UpAndOut,
                         BarrierObservation.Discrete, isCall: true, rebate: 0.0, cube));
    }

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
        var market = MakeMarket(S0, Rd, Rf);
        Assert.Throws<ArgumentException>(() =>
            new FXBarrierOptionMCPricer(vanillaOpt, ValuationDate, market, Sigma, cube));
    }

    // ── Greeks: vanilla pricer vs analytical delta ────────────────────────────
    //
    // Because FXVanillaOptionMCPricer also uses the same Greek infrastructure,
    // a sanity check here verifies the bump mechanics without needing a separate
    // file for vanilla Greek tests.

    [Fact]
    public void VanillaGreeks_Delta_MatchesAnalytical()
    {
        // GK analytical delta (call): e^{-rf·T}·N(d₁)
        var sqrtT = Math.Sqrt(T);
        var d1    = (Math.Log(S0 / K) + (Rd - Rf + 0.5 * Sigma * Sigma) * T) / (Sigma * sqrtT);
        var analyticalDelta = Math.Exp(-Rf * T) * NormalDistribution.Cdf(d1);

        var cube = SimulationCube.GenerateIndependent(Paths, 1, 1, DefaultSeed);
        using var pricer = BuildVanilla(S0, K, T, Sigma, Rd, Rf, isCall: true, cube);
        var greeks = pricer.ComputeGreeks(spotEps: 0.01);

        // Bump-and-reprice delta is estimated with finite MC noise; use 5σ bound
        // on the error contribution of the three pricing runs (mid, up, down).
        var seBound = 3.0 * 5.0 * cube.Paths;  // rough: 3 × 5σ × SE per run
        // Actually just use a practical absolute tolerance for ATM delta ≈ 0.57
        Assert.True(Math.Abs(greeks.Delta - analyticalDelta) < 0.01,
            $"MC delta={greeks.Delta:F4}  Analytical={analyticalDelta:F4}");
    }

    [Fact]
    public void VanillaGreeks_Vega_MatchesAnalytical()
    {
        // GK analytical vega (call = put): S·e^{-rf·T}·N'(d₁)·√T
        var sqrtT  = Math.Sqrt(T);
        var d1     = (Math.Log(S0 / K) + (Rd - Rf + 0.5 * Sigma * Sigma) * T) / (Sigma * sqrtT);
        var nprime = NormalDistribution.Pdf(d1);
        var analyticalVega = S0 * Math.Exp(-Rf * T) * nprime * sqrtT;

        var cube = SimulationCube.GenerateIndependent(Paths, 1, 1, DefaultSeed);
        using var pricer = BuildVanilla(S0, K, T, Sigma, Rd, Rf, isCall: true, cube);
        var greeks = pricer.ComputeGreeks(volEps: 0.001);

        Assert.True(Math.Abs(greeks.Vega - analyticalVega) < 0.01,
            $"MC vega={greeks.Vega:F4}  Analytical={analyticalVega:F4}");
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private static FXBarrierOptionMCPricer BuildBarrier(
        double spot, double strike, double expiry, double sigma,
        double rd, double rf, double barrierLevel,
        BarrierType barrierType, BarrierObservation observation,
        bool isCall, double rebate, SimulationCube cube)
    {
        var option = new Option
        {
            Underlying    = "EURUSD",
            Strike        = strike,
            ExpiryYears   = expiry,
            OptionType    = isCall ? OptionType.Call : OptionType.Put,
            ExerciseStyle = ExerciseStyle.European,
            Barrier       = new BarrierOption
            {
                BarrierLevel = barrierLevel,
                BarrierType  = barrierType,
                Observation  = observation,
                Rebate       = rebate
            }
        };
        return new FXBarrierOptionMCPricer(option, ValuationDate, MakeMarket(spot, rd, rf), sigma, cube);
    }

    private static FXVanillaOptionMCPricer BuildVanilla(
        double spot, double strike, double expiry, double sigma,
        double rd, double rf, bool isCall, SimulationCube cube)
    {
        var option = new Option
        {
            Underlying    = "EURUSD",
            Strike        = strike,
            ExpiryYears   = expiry,
            OptionType    = isCall ? OptionType.Call : OptionType.Put,
            ExerciseStyle = ExerciseStyle.European,
            Vanilla       = new VanillaOption()
        };
        return new FXVanillaOptionMCPricer(option, ValuationDate, MakeMarket(spot, rd, rf), sigma, cube);
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
            $"Diff={Math.Abs(mc.Price - analytical):F6}  {sigma}σ tol={tolerance:F6}");
    }

}
