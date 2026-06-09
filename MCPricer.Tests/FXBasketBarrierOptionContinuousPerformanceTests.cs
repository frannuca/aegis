using System.Diagnostics;
using Aegis.Instruments;
using MCPricer.FX;
using RandomSimulator;
using Xunit.Abstractions;

namespace MCPricer.Tests;

/// <summary>
/// Performance benchmark for continuous-monitoring *basket*-level barrier
/// options over a realistic daily observation schedule.
///
/// ── Why this is the expensive case ───────────────────────────────────────────
/// FXBasketBarrierOptionMCPricer.SimulatePath does strictly more work per step
/// than the single-asset FXBarrierOptionMCPricer (see
/// FXBarrierOptionContinuousPerformanceTests for that baseline): it simulates
/// N correlated GBM legs, aggregates them into a basket level, recomputes the
/// local-effective-volatility proxy σ_eff from the realized leg spots at every
/// step (an O(N²) leg-correlation sum for WEIGHTED_SUM), and — like the
/// single-asset pricer — applies a Brownian-bridge crossing correction (fresh
/// System.Random per path-step) whenever both endpoints sit on the safe side of
/// the barrier. A 3-leg basket observed daily over a 1-year trade (250 trading
/// days — the standard convention) is the realistic worst case for this code
/// path's throughput.
///
/// ── What this test is (and isn't) ────────────────────────────────────────────
/// This is a throughput benchmark, not a pricing-correctness test — see
/// FXBasketBarrierOptionMCPricerTests for the single-leg degeneracy / KI+KO
/// parity / zero-vol / far-barrier correctness coverage of the same pricer. To
/// keep CLAUDE.md's "numerical sanity check alongside every benchmark" rule
/// satisfied without duplicating that coverage, it adds one model-free sanity
/// bound: a knock-out can only remove value relative to the otherwise-identical
/// basket vanilla, so 0 <= barrier price <= vanilla price (+ MC slack).
///
/// ── Timing methodology ───────────────────────────────────────────────────────
/// Wall-clock time for a single Price() call (which internally parallelises
/// across all cores via Parallel.For — see MCBasePricer) is measured with
/// Stopwatch and reported as throughput (paths/s, path-steps/s and
/// path-step-legs/s) via test output. The pass/fail bound is a deliberately
/// generous ceiling: it exists to catch gross regressions (e.g. accidentally
/// quadratic behaviour in the bridge sampling or the σ_eff recomputation)
/// rather than to enforce a tight performance SLA, so it should not flake on
/// slower or contended CI hardware.
/// </summary>
public sealed class FXBasketBarrierOptionContinuousPerformanceTests
{
    private readonly ITestOutputHelper _output;

    public FXBasketBarrierOptionContinuousPerformanceTests(ITestOutputHelper output) => _output = output;

    // Three-leg correlated FX basket (weighted sum), each leg an ATM-ish call:
    //   EURUSD: spot=1.10, σ=20%, r_d=5%, r_f=2%   weight=0.40
    //   GBPUSD: spot=1.25, σ=25%, r_d=5%, r_f=1%   weight=0.35
    //   USDJPY: spot=1.50, σ=18%, r_d=5%, r_f=3%   weight=0.25
    // (USDJPY here is just a third synthetic leg sharing the USD domestic rate,
    // scaled to a level comparable to the others — not a real-world FX quote.)
    private const double T = 1.0;

    private const string Leg1 = "EURUSD", Leg2 = "GBPUSD", Leg3 = "USDJPY";
    private const double S1 = 1.10, Vol1 = 0.20, Rd1 = 0.05, Rf1 = 0.02, W1 = 0.40;
    private const double S2 = 1.25, Vol2 = 0.25, Rd2 = 0.05, Rf2 = 0.01, W2 = 0.35;
    private const double S3 = 1.50, Vol3 = 0.18, Rd3 = 0.05, Rf3 = 0.03, W3 = 0.25;

    // Initial level = 0.40*1.10 + 0.35*1.25 + 0.25*1.50 = 1.2525 ⇒ ATM-ish strike/barrier
    private const double Strike = 1.25;
    private const double H_Up   = 1.45;   // comfortably above the initial level — up-and-out call

    private const int Paths       = 50_000;
    private const int Steps       = 250;     // 250 daily observations over a 1-year trade
    private const int DefaultSeed = 42;

    private static readonly DateOnly ValuationDate = new(2026, 1, 1);
    private static readonly DateOnly CurveMaturity = new(2036, 1, 1);

    private static readonly double[,] Correlation =
    {
        { 1.00, 0.30, -0.20 },
        { 0.30, 1.00,  0.10 },
        { -0.20, 0.10, 1.00 }
    };

    // Deliberately generous: a regression-detection ceiling, not an SLA.
    private static readonly TimeSpan MaxElapsed = TimeSpan.FromSeconds(120);

    [Fact]
    public void ContinuousUpAndOut_ThreeLegBasket_250DailyObservations_CompletesWithinBudget()
    {
        var cube = SimulationCube.Generate(Paths, Steps, assets: 3, Correlation, DefaultSeed);
        var (markets, vols) = MakeMarkets();
        var legSpec = new (string, double)[] { (Leg1, W1), (Leg2, W2), (Leg3, W3) };

        using var pricer = new FXBasketBarrierOptionMCPricer(
            MakeBasketBarrierOption(legSpec, BasketAggregationMethod.WeightedSum, Strike, isCall: true,
                H_Up, BarrierType.UpAndOut, BarrierObservation.Continuous),
            ValuationDate, markets, vols, cube);

        var stopwatch = Stopwatch.StartNew();
        var result = pricer.Price();
        stopwatch.Stop();

        var elapsedSeconds = stopwatch.Elapsed.TotalSeconds;
        var pathSteps     = (double)Paths * Steps;
        var pathStepLegs  = pathSteps * legSpec.Length;

        _output.WriteLine($"Continuous basket-barrier MC: {Paths:N0} paths x {Steps} daily steps x {legSpec.Length} legs " +
                          $"= {pathStepLegs:N0} path-step-legs");
        _output.WriteLine($"Elapsed:     {elapsedSeconds:F3} s");
        _output.WriteLine($"Throughput:  {Paths / elapsedSeconds:N0} paths/s, {pathSteps / elapsedSeconds:N0} path-steps/s, " +
                          $"{pathStepLegs / elapsedSeconds:N0} path-step-legs/s");
        _output.WriteLine($"Price={result.Price:F6}  StdErr={result.StandardError:F6}  " +
                          $"95% CI=[{result.ConfidenceIntervalLower:F6}, {result.ConfidenceIntervalUpper:F6}]");

        Assert.True(stopwatch.Elapsed < MaxElapsed,
            $"Continuous basket-barrier pricing over {Steps} daily observations on a {legSpec.Length}-leg basket " +
            $"took {elapsedSeconds:F1}s, exceeding the {MaxElapsed.TotalSeconds:F0}s budget — investigate a possible " +
            "performance regression (e.g. accidentally quadratic behaviour in the Brownian-bridge sampling or the " +
            "σ_eff recomputation).");

        // ── Numerical sanity check (CLAUDE.md: every benchmark needs one) ────
        // Model-free, regardless of MC noise: an up-and-out knock-out can only
        // ever remove value relative to the otherwise-identical basket vanilla,
        // so its price must lie in [0, vanilla price] up to MC slack.
        Assert.True(double.IsFinite(result.Price), $"Price must be finite, got {result.Price}");
        Assert.True(result.Price >= 0.0, $"Price must be non-negative, got {result.Price:F6}");

        var vanilla = new FXBasketVanillaMCPricer(
            MakeBasketOption(legSpec, BasketAggregationMethod.WeightedSum, Strike, isCall: true),
            ValuationDate, markets, vols, cube).Price();

        var slack = 5.0 * (result.StandardError + vanilla.StandardError);
        Assert.True(result.Price <= vanilla.Price + slack,
            $"Knock-out price {result.Price:F6} must not exceed basket vanilla price {vanilla.Price:F6} " +
            $"(5σ slack={slack:F6}) — knocking out can only remove value.");
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private static Option MakeBasketBarrierOption(
        (string name, double weight)[] legs,
        BasketAggregationMethod method,
        double strike, bool isCall,
        double barrierLevel, BarrierType barrierType, BarrierObservation observation,
        double rebate = 0.0)
    {
        var basket = new Basket { AggregationMethod = method };
        foreach (var (name, weight) in legs)
            basket.UnderlyingToWeight[name] = weight;

        return new Option
        {
            Basket        = basket,
            Strike        = strike,
            ExpiryYears   = T,
            OptionType    = isCall ? OptionType.Call : OptionType.Put,
            ExerciseStyle = ExerciseStyle.European,
            Barrier = new BarrierOption
            {
                BarrierLevel = barrierLevel,
                BarrierType  = barrierType,
                Observation  = observation,
                Rebate       = rebate
            }
        };
    }

    private static Option MakeBasketOption(
        (string name, double weight)[] legs,
        BasketAggregationMethod method,
        double strike, bool isCall)
    {
        var basket = new Basket { AggregationMethod = method };
        foreach (var (name, weight) in legs)
            basket.UnderlyingToWeight[name] = weight;

        return new Option
        {
            Basket        = basket,
            Strike        = strike,
            ExpiryYears   = T,
            OptionType    = isCall ? OptionType.Call : OptionType.Put,
            ExerciseStyle = ExerciseStyle.European,
            Vanilla       = new VanillaOption()
        };
    }

    private static (Dictionary<string, FxMarketData> markets, Dictionary<string, double> vols) MakeMarkets() =>
        (new Dictionary<string, FxMarketData>
        {
            [Leg1] = MakeMarket(Leg1, S1, Rd1, Rf1),
            [Leg2] = MakeMarket(Leg2, S2, Rd2, Rf2),
            [Leg3] = MakeMarket(Leg3, S3, Rd3, Rf3)
        },
        new Dictionary<string, double> { [Leg1] = Vol1, [Leg2] = Vol2, [Leg3] = Vol3 });

    private static FxMarketData MakeMarket(string pair, double spot, double rd, double rf) =>
        new()
        {
            CurrencyPair = pair,
            Spot         = spot,
            DomesticRate = ZeroCurve.Flat(rd, CurveMaturity),
            ForeignRate  = ZeroCurve.Flat(rf, CurveMaturity),
            VolSurface   = new VolatilitySurface()
        };
}
