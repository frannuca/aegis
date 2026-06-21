using Aegis.Instruments;
using AnalyticalPricers;
using MCPricer.Equity;
using RandomSimulator;
using NodaTime;

namespace MCPricer.Tests;

/// <summary>
/// Validates EQVanillaOptionMCPricer against the Black-Scholes formula.
///
/// Black-Scholes (GBM with continuous dividend yield q):
///
///   Call = S·e^{-q·T}·N(d₁) − K·e^{-r·T}·N(d₂)
///   Put  = K·e^{-r·T}·N(−d₂) − S·e^{-q·T}·N(−d₁)
///
///   d₁ = [log(S/K) + (r − q + ½σ²)·T] / (σ√T)
///   d₂ = d₁ − σ√T
///
/// Put-call parity (model-free):
///   Call − Put = S·e^{-q·T} − K·e^{-r·T}
///
/// The Black-Scholes formula is structurally identical to Garman-Kohlhagen
/// with r_d = r, r_f = q.  Tests mirror FXVanillaOptionMCPricerTests.
///
/// ── Confidence interval methodology ─────────────────────────────────────────
/// Same 5σ bound as vanilla FX tests:  |MC − BS| < 5 × SE.
/// </summary>
public sealed class EQVanillaOptionMCPricerTests
{
    // Reference parameters: equity S0=100 ATM, 20% vol, 5% r, 2% q
    private const double S0    = 100.0;
    private const double K     = 100.0;
    private const double T     = 1.0;
    private const double Sigma = 0.20;
    private const double R     = 0.05;
    private const double Q     = 0.02;

    // Pricers now take rates as Pillars zero-rate curves (term structures), interpolated
    // at the option's expiry from this valuation date — see ZeroCurve.InterpolateRate.
    // Tests use single-pillar "flat" curves so the interpolated rate equals the scalar
    // reference rate (R, Q) regardless of T, keeping the Black-Scholes comparison exact.
    private static readonly LocalDate ValuationDate = new(2026, 1, 1);
    private static readonly LocalDate CurveMaturity = new(2036, 1, 1);

    private const int Paths       = 100_000;
    private const int Steps       = 1;        // single step suffices for European payoffs
    private const int DefaultSeed = 42;

    // ── Black-Scholes price tests ─────────────────────────────────────────────

    [Fact]
    public void AtmCall_MatchesBlackScholes()
    {
        var (mc, bs) = RunAndCompare(S0, K, T, Sigma, R, Q, isCall: true);
        AssertWithinMcBounds(mc, bs, sigma: 5.0);
    }

    [Fact]
    public void AtmPut_MatchesBlackScholes()
    {
        var (mc, bs) = RunAndCompare(S0, K, T, Sigma, R, Q, isCall: false);
        AssertWithinMcBounds(mc, bs, sigma: 5.0);
    }

    [Fact]
    public void OtmCall_MatchesBlackScholes()
    {
        var (mc, bs) = RunAndCompare(S0, strike: 120.0, T, Sigma, R, Q, isCall: true);
        AssertWithinMcBounds(mc, bs, sigma: 5.0);
    }

    [Fact]
    public void ItmCall_MatchesBlackScholes()
    {
        var (mc, bs) = RunAndCompare(S0, strike: 80.0, T, Sigma, R, Q, isCall: true);
        AssertWithinMcBounds(mc, bs, sigma: 5.0);
    }

    [Fact]
    public void OtmPut_MatchesBlackScholes()
    {
        var (mc, bs) = RunAndCompare(S0, strike: 80.0, T, Sigma, R, Q, isCall: false);
        AssertWithinMcBounds(mc, bs, sigma: 5.0);
    }

    [Fact]
    public void ItmPut_MatchesBlackScholes()
    {
        var (mc, bs) = RunAndCompare(S0, strike: 120.0, T, Sigma, R, Q, isCall: false);
        AssertWithinMcBounds(mc, bs, sigma: 5.0);
    }

    [Fact]
    public void ZeroDividend_MatchesBlackScholes()
    {
        // q=0 reduces to standard Black-Scholes (no carry adjustment on the spot term)
        var (mc, bs) = RunAndCompare(S0, K, T, Sigma, R, q: 0.0, isCall: true);
        AssertWithinMcBounds(mc, bs, sigma: 5.0);
    }

    [Fact]
    public void ZeroRate_MatchesBlackScholes()
    {
        var (mc, bs) = RunAndCompare(S0, K, T, Sigma, r: 0.0, Q, isCall: true);
        AssertWithinMcBounds(mc, bs, sigma: 5.0);
    }

    [Fact]
    public void HighDividend_MatchesBlackScholes()
    {
        // High dividend yield (q > r): forward below spot, ATM call worth less than put
        var (mc, bs) = RunAndCompare(S0, K, T, Sigma, r: 0.02, q: 0.08, isCall: true);
        AssertWithinMcBounds(mc, bs, sigma: 5.0);
    }

    [Fact]
    public void ShortExpiry_MatchesBlackScholes()
    {
        var (mc, bs) = RunAndCompare(S0, K, expiry: 1.0 / 52, Sigma, R, Q, isCall: true);
        AssertWithinMcBounds(mc, bs, sigma: 5.0);
    }

    [Fact]
    public void LongExpiry_MatchesBlackScholes()
    {
        var (mc, bs) = RunAndCompare(S0, K, expiry: 5.0, Sigma, R, Q, isCall: true);
        AssertWithinMcBounds(mc, bs, sigma: 5.0);
    }

    // ── Put-call parity ───────────────────────────────────────────────────────

    [Theory]
    [InlineData(80.0,  "ITM call / OTM put")]
    [InlineData(100.0, "ATM")]
    [InlineData(120.0, "OTM call / ITM put")]
    public void PutCallParity_HoldsWithinMcNoise(double strike, string _)
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, R, Q);
        var call   = BuildPricer(S0, strike, T, Sigma, R, Q, isCall: true,  cube).Price(market);
        var put    = BuildPricer(S0, strike, T, Sigma, R, Q, isCall: false, cube).Price(market);

        var mcParity         = call.Price - put.Price;
        var analyticalParity = S0 * Math.Exp(-Q * T) - strike * Math.Exp(-R * T);

        var seDiff = call.StandardError + put.StandardError;
        Assert.True(Math.Abs(mcParity - analyticalParity) < 5.0 * seDiff,
            $"Strike={strike}: MC={mcParity:F6}  Analytical={analyticalParity:F6}  5σ={5.0 * seDiff:F6}");
    }

    // ── Zero-vol edge cases ───────────────────────────────────────────────────

    [Fact]
    public void ZeroVol_ItmCall_PricesAtDiscountedIntrinsic()
    {
        // σ=0: S(T) = S·exp((r−q)·T) deterministic; call = e^{-r·T}·max(F−K,0)
        var forward  = S0 * Math.Exp((R - Q) * T);
        var expected = Math.Exp(-R * T) * Math.Max(forward - K, 0.0);

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, R, Q);
        var mc     = BuildPricer(S0, K, T, sigma: 0.0, R, Q, isCall: true, cube).Price(market);

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_OtmCall_PricesAtZero()
    {
        // Strike above forward → max(F−K,0) = 0
        var forward = S0 * Math.Exp((R - Q) * T);
        Assert.True(forward < 120.0, "Precondition: forward must be below 120 for this test");

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, R, Q);
        var mc     = BuildPricer(S0, strike: 120.0, T, sigma: 0.0, R, Q, isCall: true, cube).Price(market);

        Assert.Equal(0.0, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_MatchesBlackScholesLimit()
    {
        // BS with σ→0 equals discounted intrinsic; MC with σ=0 should agree
        const double tinyVol = 1e-8;
        var bs   = BlackScholes.Price(S0, K, T, tinyVol, R, Q, isCall: true);
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, R, Q);
        var mc     = BuildPricer(S0, K, T, sigma: 0.0, R, Q, isCall: true, cube).Price(market);

        Assert.Equal(bs, mc.Price, precision: 6);
    }

    // ── Variance reduction ────────────────────────────────────────────────────

    [Fact]
    public void Antithetics_ReduceStandardError()
    {
        const int halfPaths = 50_000;
        var cubeNo = SimulationCube.GenerateIndependent(halfPaths * 2, 1, 1, seed: 7, useAntithetics: false);
        var cubeAV = SimulationCube.GenerateIndependent(halfPaths * 2, 1, 1, seed: 7, useAntithetics: true);

        var market = MakeMarket(S0, R, Q);
        var seNo   = BuildPricer(S0, K, T, Sigma, R, Q, true, cubeNo).Price(market).StandardError;
        var seAV   = BuildPricer(S0, K, T, Sigma, R, Q, true, cubeAV).Price(market).StandardError;

        Assert.True(seAV < seNo, $"Antithetic SE {seAV:F6} must be < plain SE {seNo:F6}");
    }

    // ── Greeks: Black-Scholes analytical comparison ───────────────────────────

    [Fact]
    public void Greeks_Delta_MatchesBlackScholes()
    {
        // BS analytical delta (call): e^{-q·T}·N(d₁)
        var sqrtT = Math.Sqrt(T);
        var d1    = (Math.Log(S0 / K) + (R - Q + 0.5 * Sigma * Sigma) * T) / (Sigma * sqrtT);
        var analyticalDelta = Math.Exp(-Q * T) * NormalDistribution.Cdf(d1);

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, R, Q);
        using var pricer = BuildPricer(S0, K, T, Sigma, R, Q, isCall: true, cube);
        var greeks = pricer.ComputeGreeks(market, spotEps: 1.0);   // spot units are ~100, so eps=1 is 1%

        Assert.True(Math.Abs(greeks.Delta - analyticalDelta) < 0.01,
            $"MC delta={greeks.Delta:F4}  Analytical={analyticalDelta:F4}");
    }

    [Fact]
    public void Greeks_Vega_MatchesBlackScholes()
    {
        // BS vega (call = put): S·e^{-q·T}·N'(d₁)·√T
        var sqrtT  = Math.Sqrt(T);
        var d1     = (Math.Log(S0 / K) + (R - Q + 0.5 * Sigma * Sigma) * T) / (Sigma * sqrtT);
        var nprime = NormalDistribution.Pdf(d1);
        var analyticalVega = S0 * Math.Exp(-Q * T) * nprime * sqrtT;

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, R, Q);
        using var pricer = BuildPricer(S0, K, T, Sigma, R, Q, isCall: true, cube);
        var greeks = pricer.ComputeGreeks(market, volEps: 0.001);

        Assert.True(Math.Abs(greeks.Vega - analyticalVega) < 1.0,   // vega units ≈ 40 for S0=100
            $"MC vega={greeks.Vega:F4}  Analytical={analyticalVega:F4}");
    }

    [Fact]
    public void Greeks_Theta_IsNaN()
    {
        // Theta cannot be computed via cube bump; must be NaN
        var cube   = SimulationCube.GenerateIndependent(1_000, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, R, Q);
        using var pricer = BuildPricer(S0, K, T, Sigma, R, Q, isCall: true, cube);
        var greeks = pricer.ComputeGreeks(market);

        Assert.True(double.IsNaN(greeks.Theta));
    }

    [Fact]
    public void Greeks_PutCallParityInDelta()
    {
        // Call_delta - Put_delta = e^{-q·T}  (model-free parity)
        var cube     = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market   = MakeMarket(S0, R, Q);
        using var callPricer = BuildPricer(S0, K, T, Sigma, R, Q, isCall: true,  cube);
        using var putPricer  = BuildPricer(S0, K, T, Sigma, R, Q, isCall: false, cube);

        var callDelta = callPricer.ComputeGreeks(market, spotEps: 1.0).Delta;
        var putDelta  = putPricer.ComputeGreeks(market, spotEps: 1.0).Delta;
        var expected  = Math.Exp(-Q * T);

        Assert.True(Math.Abs((callDelta - putDelta) - expected) < 0.01,
            $"Call Δ={callDelta:F4}  Put Δ={putDelta:F4}  Δ diff={callDelta - putDelta:F4}  expected={expected:F4}");
    }

    // ── Input validation ──────────────────────────────────────────────────────

    [Fact]
    public void NegativeVolatility_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            BuildPricer(S0, K, T, sigma: -0.01, R, Q, true, cube));
    }

    [Fact]
    public void ZeroStrike_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        Assert.Throws<ArgumentException>(() =>
            BuildPricer(S0, strike: 0.0, T, Sigma, R, Q, true, cube));
    }

    [Fact]
    public void WrongOptionKind_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        var barrierOpt = new Option
        {
            Underlying    = "AAPL",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Barrier       = new BarrierOption
            {
                BarrierLevel = 120.0,
                BarrierType  = BarrierType.UpAndOut,
                Observation  = BarrierObservation.Discrete
            }
        };
        Assert.Throws<ArgumentException>(() =>
            new EQVanillaOptionMCPricer(barrierOpt, ValuationDate, Sigma, cube));
    }

    // ── Parametric sweep ─────────────────────────────────────────────────────

    [Theory]
    [InlineData(80.0,  100.0, 1.0,  0.15, 0.05, 0.02, true,  "OTM call deep")]
    [InlineData(120.0, 100.0, 1.0,  0.15, 0.05, 0.02, true,  "ITM call deep")]
    [InlineData(100.0, 100.0, 0.25, 0.20, 0.03, 0.01, true,  "ATM short T")]
    [InlineData(100.0, 100.0, 2.0,  0.20, 0.04, 0.02, false, "ATM put long T")]
    [InlineData(100.0, 100.0, 1.0,  0.30, 0.00, 0.00, true,  "High vol, zero rates")]
    [InlineData(100.0, 100.0, 1.0,  0.20, 0.02, 0.08, true,  "High dividend (q > r)")]
    public void ParametricSweep_MatchesBlackScholes(
        double spot, double strike, double expiry, double sigma,
        double r, double q, bool isCall, string _)
    {
        var (mc, bs) = RunAndCompare(spot, strike, expiry, sigma, r, q, isCall);
        AssertWithinMcBounds(mc, bs, sigma: 5.0);
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private (PricingResult mc, double bs) RunAndCompare(
        double spot, double strike, double expiry, double sigma,
        double r, double q, bool isCall)
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(spot, r, q);
        var mc     = BuildPricer(spot, strike, expiry, sigma, r, q, isCall, cube).Price(market);
        var bs   = BlackScholes.Price(spot, strike, expiry, sigma, r, q, isCall);
        return (mc, bs);
    }

    private static EQVanillaOptionMCPricer BuildPricer(
        double spot, double strike, double expiry, double sigma,
        double r, double q, bool isCall, SimulationCube cube)
    {
        var option = new Option
        {
            Underlying    = "AAPL",
            Strike        = strike,
            ExpiryYears   = expiry,
            OptionType    = isCall ? OptionType.Call : OptionType.Put,
            ExerciseStyle = ExerciseStyle.European,
            Vanilla       = new VanillaOption()
        };
        return new EQVanillaOptionMCPricer(option, ValuationDate, sigma, cube);
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
            $"Diff={Math.Abs(mc.Price - analytical):F6}  {sigma}σ tol={tolerance:F6}  SE={mc.StandardError:F6}");
    }
}
