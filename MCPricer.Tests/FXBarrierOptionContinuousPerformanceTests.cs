using System.Diagnostics;
using Aegis.Instruments;
using MCPricer.FX;
using RandomSimulator;
using Xunit.Abstractions;

namespace MCPricer.Tests;

/// <summary>
/// Performance benchmark for continuous-monitoring barrier options over a
/// realistic daily observation schedule.
///
/// ── Why continuous barriers are the expensive case ───────────────────────────
/// FXBarrierOptionMCPricer.CheckContinuousBarrier (see its XML doc) applies a
/// Brownian-bridge crossing correction at every step: whenever both path
/// endpoints S(t), S(t+dt) lie on the safe side of the barrier, it computes a
/// crossing probability and seeds a fresh System.Random to sample it. Unlike
/// discrete monitoring (an O(1) comparison per step), this makes per-step cost
/// materially higher — and the total cost scales with Paths x Steps. A 1-year
/// trade observed on every trading day (250 days — the standard trading-year
/// convention) is the realistic worst case for this code path's throughput.
///
/// ── What this test is (and isn't) ────────────────────────────────────────────
/// This is a throughput benchmark, not a pricing-correctness test — see
/// FXBarrierOptionMCPricerTests for the KI+KO parity / analytical / zero-vol
/// correctness coverage of the same pricer. To keep CLAUDE.md's "numerical
/// sanity check alongside every benchmark" rule satisfied without duplicating
/// that coverage, it adds one model-free sanity bound: a knock-out can only
/// remove value relative to the otherwise-identical vanilla, so
/// 0 <= barrier price <= vanilla price (+ MC slack).
///
/// ── Timing methodology ───────────────────────────────────────────────────────
/// Wall-clock time for a single Price() call (which internally parallelises
/// across all cores via Parallel.For — see MCBasePricer) is measured with
/// Stopwatch and reported as throughput (paths/s and path-steps/s) via test
/// output. The pass/fail bound is a deliberately generous ceiling: it exists to
/// catch gross regressions (e.g. accidentally quadratic behaviour in the bridge
/// sampling) rather than to enforce a tight performance SLA, so it should not
/// flake on slower or contended CI hardware.
/// </summary>
public sealed class FXBarrierOptionContinuousPerformanceTests
{
    private readonly ITestOutputHelper _output;

    public FXBarrierOptionContinuousPerformanceTests(ITestOutputHelper output) => _output = output;

    // EURUSD: spot = 1.10, K = 1.10 (ATM), T = 1y, σ = 20%, r_d (USD) = 5%, r_f (EUR) = 2%
    private const double S0    = 1.10;
    private const double K     = 1.10;
    private const double T     = 1.0;
    private const double Sigma = 0.20;
    private const double Rd    = 0.05;
    private const double Rf    = 0.02;
    private const double H_Up  = 1.30;   // above spot — up-and-out call

    private const int Paths       = 50_000;
    private const int Steps       = 250;     // 250 daily observations over a 1-year trade
    private const int DefaultSeed = 42;

    private static readonly DateOnly ValuationDate = new(2026, 1, 1);
    private static readonly DateOnly CurveMaturity = new(2036, 1, 1);

    // Deliberately generous: a regression-detection ceiling, not an SLA.
    // Typical runs on development hardware finish in low single-digit seconds.
    private static readonly TimeSpan MaxElapsed = TimeSpan.FromSeconds(60);

    [Fact]
    public void ContinuousUpAndOut_250DailyObservations_CompletesWithinBudget()
    {
        var cube = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        using var pricer = BuildContinuousBarrier(cube);

        var stopwatch = Stopwatch.StartNew();
        var result = pricer.Price();
        stopwatch.Stop();

        var elapsedSeconds = stopwatch.Elapsed.TotalSeconds;
        var pathSteps      = (double)Paths * Steps;

        _output.WriteLine($"Continuous barrier MC: {Paths:N0} paths x {Steps} daily steps = {pathSteps:N0} path-steps");
        _output.WriteLine($"Elapsed:     {elapsedSeconds:F3} s");
        _output.WriteLine($"Throughput:  {Paths / elapsedSeconds:N0} paths/s, {pathSteps / elapsedSeconds:N0} path-steps/s");
        _output.WriteLine($"Price={result.Price:F6}  StdErr={result.StandardError:F6}  " +
                          $"95% CI=[{result.ConfidenceIntervalLower:F6}, {result.ConfidenceIntervalUpper:F6}]");

        Assert.True(stopwatch.Elapsed < MaxElapsed,
            $"Continuous barrier pricing over {Steps} daily observations took {elapsedSeconds:F1}s, " +
            $"exceeding the {MaxElapsed.TotalSeconds:F0}s budget — investigate a possible performance regression " +
            "(e.g. accidentally quadratic behaviour in the Brownian-bridge sampling).");

        // ── Numerical sanity check (CLAUDE.md: every benchmark needs one) ────
        // Model-free, regardless of MC noise: an up-and-out knock-out can only
        // ever remove value relative to the otherwise-identical vanilla, so its
        // price must lie in [0, vanilla price] up to MC slack.
        Assert.True(double.IsFinite(result.Price), $"Price must be finite, got {result.Price}");
        Assert.True(result.Price >= 0.0, $"Price must be non-negative, got {result.Price:F6}");

        var vanilla = BuildVanilla(cube).Price();
        var slack   = 5.0 * (result.StandardError + vanilla.StandardError);
        Assert.True(result.Price <= vanilla.Price + slack,
            $"Knock-out price {result.Price:F6} must not exceed vanilla price {vanilla.Price:F6} " +
            $"(5σ slack={slack:F6}) — knocking out can only remove value.");
    }

    private static FXBarrierOptionMCPricer BuildContinuousBarrier(SimulationCube cube)
    {
        var option = new Option
        {
            Underlying    = "EURUSD",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Barrier = new BarrierOption
            {
                BarrierLevel = H_Up,
                BarrierType  = BarrierType.UpAndOut,
                Observation  = BarrierObservation.Continuous,
                Rebate       = 0.0
            }
        };
        return new FXBarrierOptionMCPricer(option, ValuationDate, MakeMarket(), Sigma, cube);
    }

    private static FXVanillaOptionMCPricer BuildVanilla(SimulationCube cube)
    {
        var option = new Option
        {
            Underlying    = "EURUSD",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Vanilla       = new VanillaOption()
        };
        return new FXVanillaOptionMCPricer(option, ValuationDate, MakeMarket(), Sigma, cube);
    }

    private static FxMarketData MakeMarket() =>
        new()
        {
            CurrencyPair = "EURUSD",
            Spot         = S0,
            DomesticRate = ZeroCurve.Flat(Rd, CurveMaturity),
            ForeignRate  = ZeroCurve.Flat(Rf, CurveMaturity),
            VolSurface   = new VolatilitySurface()
        };
}
