using Aegis.Instruments;
using AnalyticalPricers;
using MCPricer.FX;
using RandomSimulator;
using NodaTime;

namespace MCPricer.Tests;

/// <summary>
/// Validates FXAsianOptionMCPricer against the Kemna-Vorst closed form
/// (geometric averaging) and structural no-arbitrage properties (arithmetic
/// averaging has no closed form — see <see cref="GeometricAsian"/> XML doc).
///
/// ── Analytical benchmark: geometric average-price ────────────────────────────
/// <see cref="GeometricAsian.Price"/> — Kemna-Vorst (1990) closed form for the
/// discrete geometric average G = exp((1/n)Σ ln S(t_i)), which is exactly
/// lognormal under GBM. n = the number of MC monitoring steps.
///
/// ── Structural check: Jensen's inequality (AM ≥ GM) ──────────────────────────
/// The arithmetic mean dominates the geometric mean pathwise (AM-GM
/// inequality, with equality only on the measure-zero event of a constant
/// path), and max(·,0) is monotone non-decreasing, so PATHWISE:
///   average-price call:  max(Ā−K,0) ≥ max(Ḡ−K,0)
///   average-price put:   max(K−Ā,0) ≤ max(K−Ḡ,0)
/// Running both pricers on the *same* cube preserves these inequalities
/// exactly in the sample averages (every path contributes a same-signed
/// difference), giving a model-free, noise-free ordering check.
///
/// ── Confidence interval methodology ─────────────────────────────────────────
/// Same 5σ MC-vs-analytical bound as FXVanillaOptionMCPricerTests.
/// </summary>
public sealed class FXAsianOptionMCPricerTests
{
    // EURUSD: spot = 1.10, K = 1.10 (ATM), T = 1y, σ = 20%, r_d (USD) = 5%, r_f (EUR) = 2%
    private const double S0    = 1.10;
    private const double K     = 1.10;
    private const double T     = 1.0;
    private const double Sigma = 0.20;
    private const double Rd    = 0.05;
    private const double Rf    = 0.02;

    private const int Paths       = 100_000;
    private const int Steps       = 12;      // monthly monitoring — enough to make AM ≠ GM meaningfully
    private const int DefaultSeed = 42;

    private static readonly LocalDate ValuationDate = new(2026, 1, 1);
    private static readonly LocalDate CurveMaturity = new(2036, 1, 1);

    // ── Geometric average-price vs. Kemna-Vorst closed form ───────────────────

    [Fact]
    public void GeometricAveragePriceCall_MatchesKemnaVorst()
    {
        var (mc, kv) = RunAndCompareGeometric(S0, K, T, Sigma, Rd, Rf, isCall: true);
        AssertWithinMcBounds(mc, kv, sigma: 5.0);
    }

    [Fact]
    public void GeometricAveragePricePut_MatchesKemnaVorst()
    {
        var (mc, kv) = RunAndCompareGeometric(S0, K, T, Sigma, Rd, Rf, isCall: false);
        AssertWithinMcBounds(mc, kv, sigma: 5.0);
    }

    [Fact]
    public void GeometricAveragePriceCall_Otm_MatchesKemnaVorst()
    {
        var (mc, kv) = RunAndCompareGeometric(S0, strike: 1.20, T, Sigma, Rd, Rf, isCall: true);
        AssertWithinMcBounds(mc, kv, sigma: 5.0);
    }

    [Fact]
    public void GeometricAveragePriceCall_Itm_MatchesKemnaVorst()
    {
        var (mc, kv) = RunAndCompareGeometric(S0, strike: 1.00, T, Sigma, Rd, Rf, isCall: true);
        AssertWithinMcBounds(mc, kv, sigma: 5.0);
    }

    // ── Jensen's inequality: arithmetic vs geometric average-price ────────────

    [Fact]
    public void ArithmeticAveragePriceCall_AtLeastAsExpensiveAs_GeometricCall_ByJensensInequality()
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var arith  = BuildPricer(S0, K, T, Sigma, Rd, Rf, AsianAveragingMethod.Arithmetic, AsianStrikeStyle.AveragePrice, true, cube).Price(market);
        var geo    = BuildPricer(S0, K, T, Sigma, Rd, Rf, AsianAveragingMethod.Geometric,  AsianStrikeStyle.AveragePrice, true, cube).Price(market);

        Assert.True(arith.Price >= geo.Price - 1e-9,
            $"Arithmetic call price {arith.Price:F6} must be ≥ geometric call price {geo.Price:F6} " +
            "(AM ≥ GM pathwise ⇒ max(Ā−K,0) ≥ max(Ḡ−K,0) pathwise).");
    }

    [Fact]
    public void ArithmeticAveragePricePut_AtMostAsExpensiveAs_GeometricPut_ByJensensInequality()
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var arith  = BuildPricer(S0, K, T, Sigma, Rd, Rf, AsianAveragingMethod.Arithmetic, AsianStrikeStyle.AveragePrice, false, cube).Price(market);
        var geo    = BuildPricer(S0, K, T, Sigma, Rd, Rf, AsianAveragingMethod.Geometric,  AsianStrikeStyle.AveragePrice, false, cube).Price(market);

        Assert.True(arith.Price <= geo.Price + 1e-9,
            $"Arithmetic put price {arith.Price:F6} must be ≤ geometric put price {geo.Price:F6} " +
            "(AM ≥ GM pathwise ⇒ max(K−Ā,0) ≤ max(K−Ḡ,0) pathwise).");
    }

    // ── Zero volatility edge cases (deterministic path ⇒ exact match) ─────────

    [Fact]
    public void ZeroVol_ArithmeticAveragePriceCall_PricesAtDeterministicIntrinsic()
    {
        var path     = DeterministicFxPath(S0, Rd, Rf, T, Steps);
        var average  = path.Average();
        var expected = Math.Exp(-Rd * T) * Math.Max(average - K, 0.0);

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var mc     = BuildPricer(S0, K, T, sigma: 0.0, Rd, Rf, AsianAveragingMethod.Arithmetic, AsianStrikeStyle.AveragePrice, isCall: true, cube).Price(market);

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_GeometricAveragePriceCall_PricesAtDeterministicIntrinsic()
    {
        var path     = DeterministicFxPath(S0, Rd, Rf, T, Steps);
        var average  = Math.Exp(path.Select(x => Math.Log(x)).Average());
        var expected = Math.Exp(-Rd * T) * Math.Max(average - K, 0.0);

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var mc     = BuildPricer(S0, K, T, sigma: 0.0, Rd, Rf, AsianAveragingMethod.Geometric, AsianStrikeStyle.AveragePrice, isCall: true, cube).Price(market);

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_ArithmeticAverageStrikeCall_PricesAtDeterministicIntrinsic()
    {
        // Floating-strike payoff: max(φ·(S(T) − Ā), 0). Base Option.Strike is unused.
        var path     = DeterministicFxPath(S0, Rd, Rf, T, Steps);
        var average  = path.Average();
        var terminal = path[^1];
        var expected = Math.Exp(-Rd * T) * Math.Max(terminal - average, 0.0);

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var mc     = BuildPricer(S0, strike: 0.0, T, sigma: 0.0, Rd, Rf, AsianAveragingMethod.Arithmetic, AsianStrikeStyle.AverageStrike, isCall: true, cube).Price(market);

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    // ── Input validation ───────────────────────────────────────────────────────

    [Fact]
    public void WrongOptionKind_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, Steps, 1);
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
            new FXAsianOptionMCPricer(vanillaOpt, ValuationDate, Sigma, cube));
    }

    [Fact]
    public void UnspecifiedAveragingMethod_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, Steps, 1);
        var option = new Option
        {
            Underlying    = "EURUSD",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Asian         = new AsianOption { StrikeStyle = AsianStrikeStyle.AveragePrice }
        };
        Assert.Throws<ArgumentException>(() =>
            new FXAsianOptionMCPricer(option, ValuationDate, Sigma, cube));
    }

    [Fact]
    public void UnspecifiedStrikeStyle_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, Steps, 1);
        var option = new Option
        {
            Underlying    = "EURUSD",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Asian         = new AsianOption { AveragingMethod = AsianAveragingMethod.Arithmetic }
        };
        Assert.Throws<ArgumentException>(() =>
            new FXAsianOptionMCPricer(option, ValuationDate, Sigma, cube));
    }

    [Fact]
    public void ZeroStrike_AveragePrice_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, Steps, 1);
        Assert.Throws<ArgumentException>(() =>
            BuildPricer(S0, strike: 0.0, T, Sigma, Rd, Rf, AsianAveragingMethod.Arithmetic, AsianStrikeStyle.AveragePrice, true, cube));
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private (PricingResult mc, double kv) RunAndCompareGeometric(
        double spot, double strike, double expiry, double sigma,
        double rd, double rf, bool isCall)
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(spot, rd, rf);
        var mc     = BuildPricer(spot, strike, expiry, sigma, rd, rf,
            AsianAveragingMethod.Geometric, AsianStrikeStyle.AveragePrice, isCall, cube).Price(market);
        var kv = GeometricAsian.Price(spot, strike, expiry, sigma, rd, rf, observations: Steps, isCall);
        return (mc, kv);
    }

    private static FXAsianOptionMCPricer BuildPricer(
        double spot, double strike, double expiry, double sigma,
        double rd, double rf, AsianAveragingMethod averagingMethod, AsianStrikeStyle strikeStyle,
        bool isCall, SimulationCube cube)
    {
        var option = new Option
        {
            Underlying    = "EURUSD",
            Strike        = strike,
            ExpiryYears   = expiry,
            OptionType    = isCall ? OptionType.Call : OptionType.Put,
            ExerciseStyle = ExerciseStyle.European,
            Asian         = new AsianOption { AveragingMethod = averagingMethod, StrikeStyle = strikeStyle }
        };
        return new FXAsianOptionMCPricer(option, ValuationDate, sigma, cube);
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

    /// <summary>
    /// Replicates GbmOptionMCPricer.SimulateSpotPath's zero-volatility branch exactly:
    /// S(t) = S·exp((r_d − r_f)·t) sampled at t_i = i·Δt, Δt = T/steps.
    /// </summary>
    private static double[] DeterministicFxPath(double spot, double rd, double rf, double expiry, int steps)
    {
        var dt         = expiry / steps;
        var stepFactor = Math.Exp((rd - rf) * dt);
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
