using Aegis.Instruments;
using AnalyticalPricers;
using MCPricer.Equity;
using RandomSimulator;

namespace MCPricer.Tests;

/// <summary>
/// Validates EQAsianOptionMCPricer against the Kemna-Vorst closed form
/// (geometric averaging) and structural no-arbitrage properties (arithmetic
/// averaging has no closed form). Structurally identical to
/// FXAsianOptionMCPricerTests with r_d → r, r_f → q.
///
/// See <see cref="GeometricAsian"/> for the analytical benchmark and
/// FXAsianOptionMCPricerTests for the Jensen's-inequality (AM ≥ GM) rationale.
/// </summary>
public sealed class EQAsianOptionMCPricerTests
{
    // Equity: spot = 100 (ATM), σ = 20%, r = 5%, q = 2%
    private const double S0    = 100.0;
    private const double K     = 100.0;
    private const double T     = 1.0;
    private const double Sigma = 0.20;
    private const double R     = 0.05;
    private const double Q     = 0.02;

    private const int Paths       = 100_000;
    private const int Steps       = 12;      // monthly monitoring
    private const int DefaultSeed = 42;

    private static readonly DateOnly ValuationDate = new(2026, 1, 1);
    private static readonly DateOnly CurveMaturity = new(2036, 1, 1);

    // ── Geometric average-price vs. Kemna-Vorst closed form ───────────────────

    [Fact]
    public void GeometricAveragePriceCall_MatchesKemnaVorst()
    {
        var (mc, kv) = RunAndCompareGeometric(S0, K, T, Sigma, R, Q, isCall: true);
        AssertWithinMcBounds(mc, kv, sigma: 5.0);
    }

    [Fact]
    public void GeometricAveragePricePut_MatchesKemnaVorst()
    {
        var (mc, kv) = RunAndCompareGeometric(S0, K, T, Sigma, R, Q, isCall: false);
        AssertWithinMcBounds(mc, kv, sigma: 5.0);
    }

    [Fact]
    public void GeometricAveragePriceCall_Otm_MatchesKemnaVorst()
    {
        var (mc, kv) = RunAndCompareGeometric(S0, strike: 120.0, T, Sigma, R, Q, isCall: true);
        AssertWithinMcBounds(mc, kv, sigma: 5.0);
    }

    [Fact]
    public void GeometricAveragePriceCall_Itm_MatchesKemnaVorst()
    {
        var (mc, kv) = RunAndCompareGeometric(S0, strike: 80.0, T, Sigma, R, Q, isCall: true);
        AssertWithinMcBounds(mc, kv, sigma: 5.0);
    }

    // ── Jensen's inequality: arithmetic vs geometric average-price ────────────

    [Fact]
    public void ArithmeticAveragePriceCall_AtLeastAsExpensiveAs_GeometricCall_ByJensensInequality()
    {
        var cube  = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var arith = BuildPricer(S0, K, T, Sigma, R, Q, AsianAveragingMethod.Arithmetic, AsianStrikeStyle.AveragePrice, true, cube).Price();
        var geo   = BuildPricer(S0, K, T, Sigma, R, Q, AsianAveragingMethod.Geometric,  AsianStrikeStyle.AveragePrice, true, cube).Price();

        Assert.True(arith.Price >= geo.Price - 1e-9,
            $"Arithmetic call price {arith.Price:F6} must be ≥ geometric call price {geo.Price:F6} " +
            "(AM ≥ GM pathwise ⇒ max(Ā−K,0) ≥ max(Ḡ−K,0) pathwise).");
    }

    [Fact]
    public void ArithmeticAveragePricePut_AtMostAsExpensiveAs_GeometricPut_ByJensensInequality()
    {
        var cube  = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var arith = BuildPricer(S0, K, T, Sigma, R, Q, AsianAveragingMethod.Arithmetic, AsianStrikeStyle.AveragePrice, false, cube).Price();
        var geo   = BuildPricer(S0, K, T, Sigma, R, Q, AsianAveragingMethod.Geometric,  AsianStrikeStyle.AveragePrice, false, cube).Price();

        Assert.True(arith.Price <= geo.Price + 1e-9,
            $"Arithmetic put price {arith.Price:F6} must be ≤ geometric put price {geo.Price:F6} " +
            "(AM ≥ GM pathwise ⇒ max(K−Ā,0) ≤ max(K−Ḡ,0) pathwise).");
    }

    // ── Zero volatility edge cases (deterministic path ⇒ exact match) ─────────

    [Fact]
    public void ZeroVol_ArithmeticAveragePriceCall_PricesAtDeterministicIntrinsic()
    {
        var path     = DeterministicEqPath(S0, R, Q, T, Steps);
        var average  = path.Average();
        var expected = Math.Exp(-R * T) * Math.Max(average - K, 0.0);

        var cube = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var mc   = BuildPricer(S0, K, T, sigma: 0.0, R, Q, AsianAveragingMethod.Arithmetic, AsianStrikeStyle.AveragePrice, isCall: true, cube).Price();

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_GeometricAveragePriceCall_PricesAtDeterministicIntrinsic()
    {
        var path     = DeterministicEqPath(S0, R, Q, T, Steps);
        var average  = Math.Exp(path.Select(x => Math.Log(x)).Average());
        var expected = Math.Exp(-R * T) * Math.Max(average - K, 0.0);

        var cube = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var mc   = BuildPricer(S0, K, T, sigma: 0.0, R, Q, AsianAveragingMethod.Geometric, AsianStrikeStyle.AveragePrice, isCall: true, cube).Price();

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_ArithmeticAverageStrikeCall_PricesAtDeterministicIntrinsic()
    {
        // Floating-strike payoff: max(φ·(S(T) − Ā), 0). Base Option.Strike is unused.
        var path     = DeterministicEqPath(S0, R, Q, T, Steps);
        var average  = path.Average();
        var terminal = path[^1];
        var expected = Math.Exp(-R * T) * Math.Max(terminal - average, 0.0);

        var cube = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var mc   = BuildPricer(S0, strike: 0.0, T, sigma: 0.0, R, Q, AsianAveragingMethod.Arithmetic, AsianStrikeStyle.AverageStrike, isCall: true, cube).Price();

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    // ── Input validation ───────────────────────────────────────────────────────

    [Fact]
    public void WrongOptionKind_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, Steps, 1);
        var vanillaOpt = new Option
        {
            Underlying    = "AAPL",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Vanilla       = new VanillaOption()
        };
        var market = MakeMarket(S0, R, Q);
        Assert.Throws<ArgumentException>(() =>
            new EQAsianOptionMCPricer(vanillaOpt, ValuationDate, market, Sigma, cube));
    }

    [Fact]
    public void UnspecifiedAveragingMethod_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, Steps, 1);
        var option = new Option
        {
            Underlying    = "AAPL",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Asian         = new AsianOption { StrikeStyle = AsianStrikeStyle.AveragePrice }
        };
        var market = MakeMarket(S0, R, Q);
        Assert.Throws<ArgumentException>(() =>
            new EQAsianOptionMCPricer(option, ValuationDate, market, Sigma, cube));
    }

    [Fact]
    public void UnspecifiedStrikeStyle_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, Steps, 1);
        var option = new Option
        {
            Underlying    = "AAPL",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Asian         = new AsianOption { AveragingMethod = AsianAveragingMethod.Arithmetic }
        };
        var market = MakeMarket(S0, R, Q);
        Assert.Throws<ArgumentException>(() =>
            new EQAsianOptionMCPricer(option, ValuationDate, market, Sigma, cube));
    }

    [Fact]
    public void ZeroStrike_AveragePrice_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, Steps, 1);
        Assert.Throws<ArgumentException>(() =>
            BuildPricer(S0, strike: 0.0, T, Sigma, R, Q, AsianAveragingMethod.Arithmetic, AsianStrikeStyle.AveragePrice, true, cube));
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private (PricingResult mc, double kv) RunAndCompareGeometric(
        double spot, double strike, double expiry, double sigma,
        double r, double q, bool isCall)
    {
        var cube = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var mc = BuildPricer(spot, strike, expiry, sigma, r, q,
            AsianAveragingMethod.Geometric, AsianStrikeStyle.AveragePrice, isCall, cube).Price();
        var kv = GeometricAsian.Price(spot, strike, expiry, sigma, r, q, observations: Steps, isCall);
        return (mc, kv);
    }

    private static EQAsianOptionMCPricer BuildPricer(
        double spot, double strike, double expiry, double sigma,
        double r, double q, AsianAveragingMethod averagingMethod, AsianStrikeStyle strikeStyle,
        bool isCall, SimulationCube cube)
    {
        var option = new Option
        {
            Underlying    = "AAPL",
            Strike        = strike,
            ExpiryYears   = expiry,
            OptionType    = isCall ? OptionType.Call : OptionType.Put,
            ExerciseStyle = ExerciseStyle.European,
            Asian         = new AsianOption { AveragingMethod = averagingMethod, StrikeStyle = strikeStyle }
        };
        var market = MakeMarket(spot, r, q);
        return new EQAsianOptionMCPricer(option, ValuationDate, market, sigma, cube);
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

    /// <summary>
    /// Replicates EQMCPricer.SimulateSpotPath's zero-volatility branch exactly:
    /// S(t) = S·exp((r − q)·t) sampled at t_i = i·Δt, Δt = T/steps.
    /// </summary>
    private static double[] DeterministicEqPath(double spot, double r, double q, double expiry, int steps)
    {
        var dt         = expiry / steps;
        var stepFactor = Math.Exp((r - q) * dt);
        var path       = new double[steps];
        var s          = spot;
        for (var i = 0; i < steps; i++)
        {
            s      *= stepFactor;
            path[i] = s;
        }
        return path;
    }

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
