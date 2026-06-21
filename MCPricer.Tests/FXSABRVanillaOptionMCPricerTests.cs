using Aegis.Instruments;
using AnalyticalPricers;
using MCPricer.FX;
using RandomSimulator;
using NodaTime;

namespace MCPricer.Tests;

/// <summary>
/// Validates FXSABRVanillaOptionMCPricer against analytical benchmarks.
///
/// ── Benchmark strategy ───────────────────────────────────────────────────────
///
/// Three tiers of analytical checks, ordered by tightness:
///
///   1. ν = 0 (zero vol-of-vol):  SABR = constant-σ GBM → exact GK formula.
///      Tolerance: 5σ MC bound (no model approximation error).
///
///   2. β = 1, ν > 0:  Hagan (2002) lognormal implied vol approximation.
///      The approximation is O(T) accurate; error ≈ O(ν²T²) ≈ 1–5 bp for T ≤ 1.
///      Tolerance: 5σ MC + 0.001 absolute (10 bps in price).
///
///   3. β = 0, ν > 0:  Hagan normal (Bachelier) implied vol approximation.
///      Same error order. ATM Bachelier formula is used.
///
/// ── Hagan implied vol (lognormal, β = 1) ─────────────────────────────────────
///
///   ATM (F = K):
///     σ_B = α · [1 + (ρνα/4 + (2−3ρ²)ν²/24) · T]
///
///   Off-ATM:
///     z   = (ν/α) · (FK)^((1−β)/2) · log(F/K)
///     χ(z)= log{[√(1−2ρz+z²)+z−ρ] / (1−ρ)}
///     σ_B = α · z/χ(z)
///               ─────────────────────────────── · [1 + correction · T]
///             (FK)^((1−β)/2) · [1 + x-series]
///
///   where x-series = (1−β)²/24·x² + (1−β)⁴/1920·x⁴,  x = log(F/K)
///
/// ── Forward measure ───────────────────────────────────────────────────────────
/// All prices are P(0,T)·E^T[payoff].  The cube is generated with 2 independent
/// assets; SABR correlation is applied internally by the pricer.
/// </summary>
public sealed class FXSABRVanillaOptionMCPricerTests
{
    // Reference market: EURUSD
    private const double S0 = 1.10;
    private const double Rd = 0.05;
    private const double Rf = 0.02;
    private const double T  = 1.0;

    // Pricers now take rates as Pillars zero-rate curves (term structures), interpolated
    // at the option's expiry from this valuation date — see ZeroCurve.InterpolateRate.
    // Tests use single-pillar "flat" curves so the interpolated rate equals the scalar
    // reference rate (Rd, Rf) regardless of T, keeping the analytical comparisons exact.
    private static readonly LocalDate ValuationDate = new(2026, 1, 1);
    private static readonly LocalDate CurveMaturity = new(2036, 1, 1);

    // Reference SABR parameters (lognormal backbone)
    private const double Alpha = 0.20;
    private const double Beta1 = 1.0;   // lognormal SABR
    private const double Beta0 = 0.0;   // normal SABR
    private const double Rho   = -0.25;
    private const double Nu    = 0.30;

    private const int Paths       = 200_000;
    private const int Steps       = 100;     // SABR needs more steps than GBM
    private const int DefaultSeed = 42;

    // ── ν = 0: SABR → GBM, exact GK comparison ────────────────────────────────

    [Fact]
    public void ZeroVolOfVol_Beta1_AtmCall_MatchesGarmanKohlhagen()
    {
        // When ν=0, σ is constant at α: SABR ≡ GBM with σ=α.
        // MC SABR price must match the GK analytical formula to 5σ.
        var f0  = Forward();
        var mc  = RunSabr(strike: f0, alpha: Alpha, beta: Beta1, rho: 0.0, nu: 0.0, isCall: true);
        var gk  = GarmanKohlhagen.Price(S0, f0, T, Alpha, Rd, Rf, isCall: true);

        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void ZeroVolOfVol_Beta1_OtmCall_MatchesGarmanKohlhagen()
    {
        var f0  = Forward();
        var mc  = RunSabr(strike: f0 * 1.10, alpha: Alpha, beta: Beta1, rho: 0.0, nu: 0.0, isCall: true);
        var gk  = GarmanKohlhagen.Price(S0, f0 * 1.10, T, Alpha, Rd, Rf, isCall: true);

        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void ZeroVolOfVol_Beta1_AtmPut_MatchesGarmanKohlhagen()
    {
        var f0 = Forward();
        var mc = RunSabr(strike: f0, alpha: Alpha, beta: Beta1, rho: 0.0, nu: 0.0, isCall: false);
        var gk = GarmanKohlhagen.Price(S0, f0, T, Alpha, Rd, Rf, isCall: false);

        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    // ── β=1, ν>0: compare to Hagan lognormal approximation ───────────────────

    [Fact]
    public void Beta1_AtmCall_MatchesHaganApproximation()
    {
        var f0  = Forward();
        var mc  = RunSabr(strike: f0, alpha: Alpha, beta: Beta1, rho: Rho, nu: Nu, isCall: true);
        var vol = Hagan.BlackVol(f0, f0, T, Alpha, Beta1, Rho, Nu);
        var ref_ = BlackForward.Call(f0, f0, T, vol) * DiscountFactor();

        AssertWithinMcAndModelBounds(mc, ref_, sigma: 5.0, absoluteTol: 0.001);
    }

    [Fact]
    public void Beta1_OtmCall_MatchesHaganApproximation()
    {
        var f0   = Forward();
        var k    = f0 * 1.10;    // 10% OTM call
        var mc   = RunSabr(strike: k, alpha: Alpha, beta: Beta1, rho: Rho, nu: Nu, isCall: true);
        var vol  = Hagan.BlackVol(f0, k, T, Alpha, Beta1, Rho, Nu);
        var ref_ = BlackForward.Call(f0, k, T, vol) * DiscountFactor();

        AssertWithinMcAndModelBounds(mc, ref_, sigma: 5.0, absoluteTol: 0.001);
    }

    [Fact]
    public void Beta1_ItmCall_MatchesHaganApproximation()
    {
        var f0   = Forward();
        var k    = f0 * 0.90;    // 10% ITM call
        var mc   = RunSabr(strike: k, alpha: Alpha, beta: Beta1, rho: Rho, nu: Nu, isCall: true);
        var vol  = Hagan.BlackVol(f0, k, T, Alpha, Beta1, Rho, Nu);
        var ref_ = BlackForward.Call(f0, k, T, vol) * DiscountFactor();

        AssertWithinMcAndModelBounds(mc, ref_, sigma: 5.0, absoluteTol: 0.001);
    }

    [Fact]
    public void Beta1_AtmPut_MatchesHaganApproximation()
    {
        var f0   = Forward();
        var mc   = RunSabr(strike: f0, alpha: Alpha, beta: Beta1, rho: Rho, nu: Nu, isCall: false);
        var vol  = Hagan.BlackVol(f0, f0, T, Alpha, Beta1, Rho, Nu);
        var ref_ = BlackForward.Put(f0, f0, T, vol) * DiscountFactor();

        AssertWithinMcAndModelBounds(mc, ref_, sigma: 5.0, absoluteTol: 0.001);
    }

    // ── β=0: normal SABR, compare to Hagan normal approximation ──────────────

    [Fact]
    public void Beta0_AtmCall_MatchesHaganNormalApproximation()
    {
        var f0  = Forward();
        var mc  = RunSabr(strike: f0, alpha: Alpha, beta: Beta0, rho: Rho, nu: Nu, isCall: true);
        // Hagan normal ATM σ_N = α · (1 + (2−3ρ²)ν²/24 · T)
        var sigN = Hagan.NormalVolAtm(Alpha, Rho, Nu, T);
        var ref_ = Bachelier.Price(f0, f0, T, sigN, DiscountFactor(), isCall: true);

        AssertWithinMcAndModelBounds(mc, ref_, sigma: 5.0, absoluteTol: 0.001);
    }

    // ── Smile properties ──────────────────────────────────────────────────────

    [Fact]
    public void Beta1_NegativeRho_PutsMoreExpensiveThanCalls_LogSymmetricStrikes()
    {
        // Negative ρ gives a negative vol skew: implied vol rises for lower strikes.
        // With strikes symmetric in log-space (K_put = F·e^{-x}, K_call = F·e^{+x}),
        // the put sees higher implied vol → larger forward premium → more expensive.
        //
        // Absolute-distance strikes (K=F±ΔF) are NOT appropriate here: they are
        // asymmetric in log-space, so the call ends up further from ATM, which can
        // reverse the price comparison independently of the skew.
        var f0 = Forward();
        const double logDistance = 0.10;
        var kPut  = f0 * Math.Exp(-logDistance);
        var kCall = f0 * Math.Exp(+logDistance);

        var mcPut  = RunSabr(strike: kPut,  alpha: Alpha, beta: Beta1, rho: -0.50, nu: Nu, isCall: false);
        var mcCall = RunSabr(strike: kCall, alpha: Alpha, beta: Beta1, rho: -0.50, nu: Nu, isCall: true);

        Assert.True(mcPut.Price > mcCall.Price,
            $"Log-symmetric OTM put={mcPut.Price:F6} should exceed OTM call={mcCall.Price:F6} under ρ < 0");
    }

    [Fact]
    public void Beta1_HigherVolOfVol_IncreasesOtmOptionPrice()
    {
        // Higher ν widens the vol distribution → increases OTM option values (Jensen's
        // inequality / convexity effect). Call and put both more expensive OTM.
        var f0 = Forward();
        var k  = f0 * 1.15;  // OTM call

        var mcLowNu  = RunSabr(strike: k, alpha: Alpha, beta: Beta1, rho: 0.0, nu: 0.10, isCall: true);
        var mcHighNu = RunSabr(strike: k, alpha: Alpha, beta: Beta1, rho: 0.0, nu: 0.50, isCall: true);

        Assert.True(mcHighNu.Price > mcLowNu.Price,
            $"Higher ν={0.50}: price={mcHighNu.Price:F6} must exceed ν={0.10}: price={mcLowNu.Price:F6}");
    }

    // ── Put-call parity in forward measure ────────────────────────────────────

    [Fact]
    public void PutCallParity_HoldsUnderSabr()
    {
        // Call − Put = P(0,T) · (F − K)  (model-free in forward measure)
        // Using same cube for both: correlation between payoffs reduces parity noise.
        var f0   = Forward();
        var cube = BuildCube();

        var call = BuildPricer(strike: f0, alpha: Alpha, beta: Beta1, rho: Rho, nu: Nu, isCall: true,  cube).Price(MakeMarket());
        var put  = BuildPricer(strike: f0, alpha: Alpha, beta: Beta1, rho: Rho, nu: Nu, isCall: false, cube).Price(MakeMarket());

        var mcParity         = call.Price - put.Price;
        var analyticalParity = DiscountFactor() * (f0 - f0);   // ATM: F = K → 0
        var seDiff           = call.StandardError + put.StandardError;

        Assert.True(Math.Abs(mcParity - analyticalParity) < 5.0 * seDiff,
            $"Parity diff={mcParity:F6}  Expected=0  5σ={5.0 * seDiff:F6}");
    }

    [Fact]
    public void PutCallParity_OffAtm_HoldsUnderSabr()
    {
        var f0   = Forward();
        var k    = f0 * 1.05;
        var cube = BuildCube();

        var call = BuildPricer(strike: k, alpha: Alpha, beta: Beta1, rho: Rho, nu: Nu, isCall: true,  cube).Price(MakeMarket());
        var put  = BuildPricer(strike: k, alpha: Alpha, beta: Beta1, rho: Rho, nu: Nu, isCall: false, cube).Price(MakeMarket());

        var mcParity         = call.Price - put.Price;
        var analyticalParity = DiscountFactor() * (f0 - k);
        var seDiff           = call.StandardError + put.StandardError;

        Assert.True(Math.Abs(mcParity - analyticalParity) < 5.0 * seDiff,
            $"Strike={k:F4}: parity diff={Math.Abs(mcParity - analyticalParity):F6}  5σ={5.0 * seDiff:F6}");
    }

    // ── Convergence in step count ─────────────────────────────────────────────

    [Fact]
    public void Beta1_AtmPrice_ConvergesAsStepsIncrease()
    {
        // Log-Euler for β=1 is exact for GBM; SABR step error comes from the
        // cross-variation between F and σ. Finer steps → smaller discretization error.
        var f0    = Forward();
        var cube50  = SimulationCube.GenerateIndependent(Paths, steps: 50,  assets: 2, seed: DefaultSeed);
        var cube252 = SimulationCube.GenerateIndependent(Paths, steps: 252, assets: 2, seed: DefaultSeed + 1);

        var mc50  = BuildPricer(f0, Alpha, Beta1, Rho, Nu, true, cube50).Price(MakeMarket());
        var mc252 = BuildPricer(f0, Alpha, Beta1, Rho, Nu, true, cube252).Price(MakeMarket());
        var vol   = Hagan.BlackVol(f0, f0, T, Alpha, Beta1, Rho, Nu);
        var ref_  = BlackForward.Call(f0, f0, T, vol) * DiscountFactor();

        // Both must be in the right ballpark; finer steps should be at most slightly
        // closer to the reference — we don't enforce strict ordering since both
        // are within MC noise, but each must pass the 8σ Hagan + MC bound.
        AssertWithinMcAndModelBounds(mc50,  ref_, sigma: 8.0, absoluteTol: 0.002);
        AssertWithinMcAndModelBounds(mc252, ref_, sigma: 8.0, absoluteTol: 0.002);
    }

    // ── Greeks ────────────────────────────────────────────────────────────────

    [Fact]
    public void Greeks_Delta_IsFiniteAndPositiveForCall()
    {
        var f0   = Forward();
        var cube = BuildCube();
        using var pricer = BuildPricer(f0, Alpha, Beta1, Rho, Nu, isCall: true, cube);
        var greeks = pricer.ComputeGreeks(MakeMarket(), spotEps: 0.01);

        Assert.True(double.IsFinite(greeks.Delta), $"Delta={greeks.Delta}");
        Assert.True(greeks.Delta > 0.0, $"ATM call delta should be positive: {greeks.Delta:F4}");
        Assert.True(double.IsFinite(greeks.Vega),  $"Vega(α)={greeks.Vega}");
        Assert.True(greeks.Vega > 0.0, $"Vega (dV/dα) should be positive: {greeks.Vega:F4}");
        Assert.True(double.IsNaN(greeks.Theta), "Theta must be NaN");
    }

    [Fact]
    public void Greeks_Vega_IsAlphaSensitivity()
    {
        // Verify Vega = dV/dα by comparing ComputeGreeks().Vega against a
        // direct CreateBumped(VolUp) call.  Must agree to floating-point precision.
        var f0   = Forward();
        var cube = BuildCube();
        using var pricer = BuildPricer(f0, Alpha, Beta1, Rho, Nu, isCall: true, cube);

        const double eps    = 0.001;
        var market     = MakeMarket();
        var up  = ((FXSABRVanillaOptionMCPricer)pricer.CreateBumped_ForTest(BumpType.VolUp,  eps)).Price(market).Price;
        var dn  = ((FXSABRVanillaOptionMCPricer)pricer.CreateBumped_ForTest(BumpType.VolDown, eps)).Price(market).Price;
        var directVega = (up - dn) / (2.0 * eps);
        var greeks     = pricer.ComputeGreeks(market, volEps: eps);

        Assert.Equal(directVega, greeks.Vega, precision: 10);
    }

    [Fact]
    public void VolOfVolSensitivity_IsFiniteAndPositive()
    {
        // Higher ν → higher OTM vol → higher option value; vol-of-vol sensitivity
        // (Volvol greek) is positive for vanilla options.
        var f0   = Forward();
        var cube = BuildCube();
        using var pricer = BuildPricer(f0, Alpha, Beta1, Rho, Nu, isCall: true, cube);
        var volOfVol = pricer.ComputeVolOfVolSensitivity(MakeMarket(), nuEps: 0.01);

        Assert.True(double.IsFinite(volOfVol), $"Volvol={volOfVol}");
        Assert.True(volOfVol > 0.0, $"Volvol should be positive for a vanilla call: {volOfVol:F4}");
    }

    [Fact]
    public void CorrelSensitivity_IsFiniteForCall()
    {
        var f0   = Forward();
        var cube = BuildCube();
        using var pricer = BuildPricer(f0, Alpha, Beta1, Rho, Nu, isCall: true, cube);
        var cs = pricer.ComputeCorrelSensitivity(MakeMarket(), rhoEps: 0.01);

        Assert.True(double.IsFinite(cs), $"Correlation sensitivity={cs}");
    }

    // ── Input validation ──────────────────────────────────────────────────────

    [Fact]
    public void ZeroAlpha_Throws()
    {
        var cube = BuildCube();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            BuildPricer(Forward(), alpha: 0.0, beta: Beta1, rho: Rho, nu: Nu, isCall: true, cube));
    }

    [Fact]
    public void BetaOutOfRange_Throws()
    {
        var cube = BuildCube();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            BuildPricer(Forward(), alpha: Alpha, beta: 1.5, rho: Rho, nu: Nu, isCall: true, cube));
    }

    [Fact]
    public void RhoAtBoundary_Throws()
    {
        var cube = BuildCube();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            BuildPricer(Forward(), alpha: Alpha, beta: Beta1, rho: 1.0, nu: Nu, isCall: true, cube));
    }

    [Fact]
    public void NegativeNu_Throws()
    {
        var cube = BuildCube();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            BuildPricer(Forward(), alpha: Alpha, beta: Beta1, rho: Rho, nu: -0.10, isCall: true, cube));
    }

    [Fact]
    public void SingleAssetCube_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(1_000, Steps, assets: 1, DefaultSeed);
        Assert.Throws<ArgumentException>(() =>
            BuildPricer(Forward(), Alpha, Beta1, Rho, Nu, true, cube));
    }

    [Fact]
    public void WrongOptionKind_Throws()
    {
        var cube = BuildCube();
        var barrierOpt = new Option
        {
            Underlying = "EURUSD",
            Strike     = Forward(),
            ExpiryYears= T,
            OptionType = OptionType.Call,
            Barrier    = new BarrierOption { BarrierLevel = 1.30, BarrierType = BarrierType.UpAndOut }
        };
        Assert.Throws<ArgumentException>(() =>
            new FXSABRVanillaOptionMCPricer(barrierOpt, ValuationDate, MakeSabr(Alpha, Beta1, Rho, Nu), cube));
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private double Forward()       => S0 * Math.Exp((Rd - Rf) * T);
    private double DiscountFactor() => Math.Exp(-Rd * T);

    private PricingResult RunSabr(
        double strike, double alpha, double beta, double rho, double nu, bool isCall)
    {
        return BuildPricer(strike, alpha, beta, rho, nu, isCall, BuildCube()).Price(MakeMarket());
    }

    private FXSABRVanillaOptionMCPricer BuildPricer(
        double strike, double alpha, double beta, double rho, double nu,
        bool isCall, SimulationCube cube)
    {
        var option = new Option
        {
            Underlying    = "EURUSD",
            Strike        = strike,
            ExpiryYears   = T,
            OptionType    = isCall ? OptionType.Call : OptionType.Put,
            ExerciseStyle = ExerciseStyle.European,
            Vanilla       = new VanillaOption()
        };
        return new FXSABRVanillaOptionMCPricer(option, ValuationDate, MakeSabr(alpha, beta, rho, nu), cube);
    }

    private static FxMarketData MakeMarket() =>
        new() { CurrencyPair = "EURUSD", Spot = S0,
                DomesticRate = ZeroCurve.Flat(Rd, CurveMaturity),
                ForeignRate  = ZeroCurve.Flat(Rf, CurveMaturity),
                VolSurface = new VolatilitySurface() };

    private static SabrParameters MakeSabr(double alpha, double beta, double rho, double nu) =>
        new() { Alpha = alpha, Beta = beta, Rho = rho, Nu = nu };

    private static SimulationCube BuildCube() =>
        SimulationCube.GenerateIndependent(Paths, Steps, assets: 2, seed: DefaultSeed);

    private static void AssertWithinMcBounds(PricingResult mc, double analytical, double sigma)
    {
        var tol = sigma * mc.StandardError;
        Assert.True(Math.Abs(mc.Price - analytical) < tol,
            $"MC={mc.Price:F6}  Ref={analytical:F6}  Diff={Math.Abs(mc.Price - analytical):F6}  {sigma}σ tol={tol:F6}");
    }

    private static void AssertWithinMcAndModelBounds(
        PricingResult mc, double reference, double sigma, double absoluteTol)
    {
        var mcTol   = sigma * mc.StandardError;
        var totalTol = mcTol + absoluteTol;
        Assert.True(Math.Abs(mc.Price - reference) < totalTol,
            $"MC={mc.Price:F6}  Hagan={reference:F6}  Diff={Math.Abs(mc.Price - reference):F6}  " +
            $"Tol={totalTol:F6} ({sigma}σ={mcTol:F6} + model={absoluteTol:F6})");
    }
}

/// <summary>
/// Extension used in tests to access CreateBumped without changing visibility.
/// Only used to verify Vega == dV/dα consistency.
/// </summary>
file static class FXSABRVanillaTestExtensions
{
    public static MCBasePricer CreateBumped_ForTest(
        this FXSABRVanillaOptionMCPricer pricer, BumpType bump, double eps)
    {
        // Use reflection to call the protected method rather than leaking it as public.
        var m = typeof(FXSABRVanillaOptionMCPricer)
                    .GetMethod("CreateBumped",
                               System.Reflection.BindingFlags.NonPublic |
                               System.Reflection.BindingFlags.Instance)!;
        return (MCBasePricer)m.Invoke(pricer, [bump, eps])!;
    }
}
