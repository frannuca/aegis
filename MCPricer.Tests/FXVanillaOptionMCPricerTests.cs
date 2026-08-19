using Aegis.Instruments;
using AnalyticalPricers;
using MCPricer.FX;
using RandomSimulator;
using NodaTime;

namespace MCPricer.Tests;

/// <summary>
/// Validates FXVanillaOptionMCPricer against the Garman-Kohlhagen analytical formula.
///
/// Garman-Kohlhagen (1983) — European FX option under Q^d (domestic numeraire):
///
///   Call = S·e^{-r_f·T}·N(d₁) − K·e^{-r_d·T}·N(d₂)
///   Put  = K·e^{-r_d·T}·N(−d₂) − S·e^{-r_f·T}·N(−d₁)
///
///   d₁ = [log(S/K) + (r_d − r_f + ½σ²)·T] / (σ√T)
///   d₂ = d₁ − σ√T
///
/// Put-call parity (GK):
///   Call − Put = S·e^{-r_f·T} − K·e^{-r_d·T}    (model-free from no-arbitrage)
///
/// ── Confidence interval methodology ─────────────────────────────────────────
/// All MC vs analytical comparisons use a 5σ bound:
///   |MC_price − GK_price| < 5 × SE
///
/// where SE is the sample standard error of the mean returned by PricingResult.
/// At 5σ, the probability of a spurious failure per test is < 6×10⁻⁷.
/// Deterministic seeds guarantee that failures are reproducible.
///
/// ── Path count choice ────────────────────────────────────────────────────────
/// 100,000 paths give SE ≈ 0.05 for typical ATM payoffs, so the 5σ bound is ≈ 0.25.
/// This catches real pricing errors (which would be much larger) while running
/// in well under a second per test.
/// </summary>
public sealed class FXVanillaOptionMCPricerTests
{
    // ── Reference parameters (used in most tests) ──────────────────────────────
    // EURUSD: spot = 1.10, K = 1.10 (ATM), T = 1y, σ = 20%, r_d (USD) = 5%, r_f (EUR) = 2%
    private const double S0    = 1.10;
    private const double K     = 1.10;
    private const double T     = 1.0;
    private const double Sigma = 0.20;
    private const double Rd    = 0.05;
    private const double Rf    = 0.02;

    private const int Paths      = 100_000;
    private const int Steps      = 1;       // single step suffices for European payoffs
    private const int DefaultSeed = 42;

    private static readonly LocalDate ValuationDate = new(2026, 1, 1);
    private static readonly LocalDate CurveMaturity = new(2036, 1, 1);

    // ── Garman-Kohlhagen price tests ──────────────────────────────────────────

    [Fact]
    public void AtmCall_MatchesGarmanKohlhagen()
    {
        var (mc, gk) = RunAndCompare(S0, K, T, Sigma, Rd, Rf, isCall: true);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void AtmPut_MatchesGarmanKohlhagen()
    {
        var (mc, gk) = RunAndCompare(S0, K, T, Sigma, Rd, Rf, isCall: false);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void OtmCall_MatchesGarmanKohlhagen()
    {
        var (mc, gk) = RunAndCompare(S0, strike: 1.20, T, Sigma, Rd, Rf, isCall: true);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void ItmCall_MatchesGarmanKohlhagen()
    {
        var (mc, gk) = RunAndCompare(S0, strike: 1.00, T, Sigma, Rd, Rf, isCall: true);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void OtmPut_MatchesGarmanKohlhagen()
    {
        var (mc, gk) = RunAndCompare(S0, strike: 1.00, T, Sigma, Rd, Rf, isCall: false);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void HighVolatility_MatchesGarmanKohlhagen()
    {
        var (mc, gk) = RunAndCompare(S0, K, T, sigma: 0.50, Rd, Rf, isCall: true);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void LowVolatility_MatchesGarmanKohlhagen()
    {
        var (mc, gk) = RunAndCompare(S0, K, T, sigma: 0.05, Rd, Rf, isCall: true);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void ShortExpiry_MatchesGarmanKohlhagen()
    {
        var (mc, gk) = RunAndCompare(S0, K, expiry: 1.0 / 52, Sigma, Rd, Rf, isCall: true);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void LongExpiry_MatchesGarmanKohlhagen()
    {
        var (mc, gk) = RunAndCompare(S0, K, expiry: 5.0, Sigma, Rd, Rf, isCall: true);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void ZeroForeignRate_MatchesGarmanKohlhagen()
    {
        var (mc, gk) = RunAndCompare(S0, K, T, Sigma, Rd, rf: 0.0, isCall: true);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void ZeroDomesticRate_MatchesGarmanKohlhagen()
    {
        var (mc, gk) = RunAndCompare(S0, K, T, Sigma, rd: 0.0, Rf, isCall: true);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void EqualRates_MatchesGarmanKohlhagen()
    {
        var (mc, gk) = RunAndCompare(S0, K, T, Sigma, rd: 0.03, rf: 0.03, isCall: true);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    // ── Multi-step path: same answer as single-step for European ──────────────

    [Fact]
    public void DailySteps_MatchesSingleStep_WithinMcNoise()
    {
        const int dailySteps = 252;

        var cube1Step    = SimulationCube.GenerateIndependent(Paths, 1,          1, DefaultSeed);
        var cube252Steps = SimulationCube.GenerateIndependent(Paths, dailySteps, 1, DefaultSeed + 1);

        var (mc1, _)   = RunAndCompareWithCube(S0, K, T, Sigma, Rd, Rf, true, cube1Step);
        var (mc252, _) = RunAndCompareWithCube(S0, K, T, Sigma, Rd, Rf, true, cube252Steps);
        var gk         = GarmanKohlhagen.Price(S0, K, T, Sigma, Rd, Rf, isCall: true);

        AssertWithinMcBounds(mc1,   gk, sigma: 5.0);
        AssertWithinMcBounds(mc252, gk, sigma: 5.0);
    }

    // ── Put-call parity ───────────────────────────────────────────────────────

    [Theory]
    [InlineData(1.00, "ITM call / OTM put")]
    [InlineData(1.10, "ATM")]
    [InlineData(1.20, "OTM call / ITM put")]
    public void PutCallParity_HoldsWithinMcNoise(double strike, string _)
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var mcCall = BuildPricer(S0, strike, T, Sigma, Rd, Rf, true,  cube).Price(market);
        var mcPut  = BuildPricer(S0, strike, T, Sigma, Rd, Rf, false, cube).Price(market);

        var mcParity         = mcCall.Price - mcPut.Price;
        var analyticalParity = S0 * Math.Exp(-Rf * T) - strike * Math.Exp(-Rd * T);

        var seDiff = mcCall.StandardError + mcPut.StandardError;

        Assert.True(Math.Abs(mcParity - analyticalParity) < 5.0 * seDiff,
            $"Strike={strike}: MC parity={mcParity:F6}  analytical={analyticalParity:F6}  " +
            $"5σ tolerance={5.0 * seDiff:F6}");
    }

    // ── Zero volatility edge case ─────────────────────────────────────────────

    [Fact]
    public void ZeroVol_ItmCall_PricesAtDiscountedIntrinsic()
    {
        var forward  = S0 * Math.Exp((Rd - Rf) * T);
        var expected = Math.Exp(-Rd * T) * Math.Max(forward - K, 0.0);

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var mc     = BuildPricer(S0, K, T, sigma: 0.0, Rd, Rf, isCall: true, cube).Price(market);

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_OtmCall_PricesAtZero()
    {
        const double otmStrike = 1.30;
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, rd: 0.02, rf: 0.05);
        var mc     = BuildPricer(S0, otmStrike, T, sigma: 0.0, rd: 0.02, rf: 0.05, isCall: true, cube).Price(market);

        Assert.Equal(0.0, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_GarmanKohlhagen_Agrees()
    {
        const double tinyVol = 1e-8;
        var gk     = GarmanKohlhagen.Price(S0, K, T, tinyVol, Rd, Rf, isCall: true);
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var mc     = BuildPricer(S0, K, T, sigma: tinyVol, Rd, Rf, isCall: true, cube).Price(market);

        Assert.Equal(gk, mc.Price, precision: 6);
    }

    // ── Convergence: Sobol beats pseudo-random at low path count ─────────────

    [Fact]
    public void Sobol_ConvergesFasterThanPseudo_ForAtmCall()
    {
        const int lowPaths = 2048;
        var gk     = GarmanKohlhagen.Price(S0, K, T, Sigma, Rd, Rf, isCall: true);
        var market = MakeMarket(S0, Rd, Rf);

        var sobolCube  = SimulationCube.GenerateIndependent(lowPaths, 1, 1, method: SamplingMethod.QuasiRandom);
        var sobolPrice = BuildPricer(S0, K, T, Sigma, Rd, Rf, true, sobolCube).Price(market).Price;
        var sobolError = Math.Abs(sobolPrice - gk);

        var pseudoErrors = new double[101];
        for (var seed = 0; seed < pseudoErrors.Length; seed++)
        {
            var cube  = SimulationCube.GenerateIndependent(lowPaths, 1, 1, seed: seed, method: SamplingMethod.PseudoRandom);
            var price = BuildPricer(S0, K, T, Sigma, Rd, Rf, true, cube).Price(market).Price;
            pseudoErrors[seed] = Math.Abs(price - gk);
        }
        Array.Sort(pseudoErrors);
        var pseudoMedian = pseudoErrors[pseudoErrors.Length / 2];

        Assert.True(sobolError < pseudoMedian,
            $"Sobol error {sobolError:F6} must be < pseudo median error {pseudoMedian:F6} " +
            $"(GK={gk:F6}, Sobol={sobolPrice:F6})");
    }

    // ── Antithetics reduce variance ───────────────────────────────────────────

    [Fact]
    public void Antithetics_ReduceStandardError()
    {
        const int halfPaths = 50_000;

        var cubeNo = SimulationCube.GenerateIndependent(halfPaths * 2, 1, 1, seed: 77, useAntithetics: false);
        var cubeAV = SimulationCube.GenerateIndependent(halfPaths * 2, 1, 1, seed: 77, useAntithetics: true);
        var market = MakeMarket(S0, Rd, Rf);

        var seNo = BuildPricer(S0, K, T, Sigma, Rd, Rf, true, cubeNo).Price(market).StandardError;
        var seAV = BuildPricer(S0, K, T, Sigma, Rd, Rf, true, cubeAV).Price(market).StandardError;

        Assert.True(seAV < seNo,
            $"Antithetic SE {seAV:F6} must be less than plain SE {seNo:F6}");
    }

    // ── Input validation ──────────────────────────────────────────────────────

    [Fact]
    public void NegativeVolatility_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            BuildPricer(S0, K, T, sigma: -0.01, Rd, Rf, true, cube));
    }

    [Fact]
    public void ZeroStrike_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        Assert.Throws<ArgumentException>(() =>
            BuildPricer(S0, strike: 0.0, T, Sigma, Rd, Rf, true, cube));
    }

    [Fact]
    public void ZeroSpot_Throws()
    {
        var cube   = SimulationCube.GenerateIndependent(100, 1, 1);
        var pricer = BuildPricer(spot: 0.0, K, T, Sigma, Rd, Rf, true, cube);
        Assert.Throws<ArgumentException>(() => pricer.Price(MakeMarket(spot: 0.0, Rd, Rf)));
    }

    [Fact]
    public void ZeroExpiry_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        Assert.Throws<ArgumentException>(() =>
            BuildPricer(S0, K, expiry: 0.0, Sigma, Rd, Rf, true, cube));
    }

    [Fact]
    public void WrongOptionKind_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, 1, 1);
        var barrierOpt = new Option
        {
            Underlying   = "EURUSD",
            Strike       = K,
            ExpiryYears  = T,
            OptionType   = OptionType.Call,
            Barrier      = new BarrierOption
            {
                BarrierLevel = 1.20,
                BarrierType  = BarrierType.UpAndOut,
                Observation  = BarrierObservation.Discrete
            }
        };
        Assert.Throws<ArgumentException>(() =>
            new FXVanillaOptionMCPricer(barrierOpt, ValuationDate, Sigma, cube));
    }

    // ── Async pricing ─────────────────────────────────────────────────────────

    [Fact]
    public async Task PriceAsync_NoToken_MatchesGarmanKohlhagen()
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        using var pricer = BuildPricer(S0, K, T, Sigma, Rd, Rf, true, cube);
        var market = MakeMarket(S0, Rd, Rf);

        var result = await pricer.PriceAsync(market);
        var gk     = GarmanKohlhagen.Price(S0, K, T, Sigma, Rd, Rf, isCall: true);

        AssertWithinMcBounds(result, gk, sigma: 5.0);
    }

    [Fact]
    public async Task PriceAsync_DefaultToken_MatchesSyncPrice()
    {
        // Two separate cubes initialised from the same seed produce identical
        // random streams, so the sync and async pricers consume equivalent
        // inputs and must produce bit-for-bit equal prices.
        var cubeSync  = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var cubeAsync = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);

        using var syncPricer  = BuildPricer(S0, K, T, Sigma, Rd, Rf, true, cubeSync);
        using var asyncPricer = BuildPricer(S0, K, T, Sigma, Rd, Rf, true, cubeAsync);

        var gk     = GarmanKohlhagen.Price(S0, K, T, Sigma, Rd, Rf, isCall: true);
        var market = MakeMarket(S0, Rd, Rf);

        var sync   = syncPricer.Price(market);
        var async_ = await asyncPricer.PriceAsync(market);

        // Both pricers used identical random streams, so their prices must agree
        // to full double precision, verifying the "MatchesSyncPrice" contract.
        Assert.Equal(sync.Price, async_.Price, precision: 10);

        AssertWithinMcBounds(sync,   gk, sigma: 5.0);
        AssertWithinMcBounds(async_, gk, sigma: 5.0);
    }

    // ── Cancellation ──────────────────────────────────────────────────────────

    [Fact]
    public async Task PriceAsync_PreCancelledToken_ThrowsImmediately()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        using var pricer = BuildPricer(S0, K, T, Sigma, Rd, Rf, true, cube);
        var market = MakeMarket(S0, Rd, Rf);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pricer.PriceAsync(market, cts.Token));
    }

    [Fact]
    public async Task PriceAsync_CancelledDuringRun_ThrowsOperationCanceledException()
    {
        using var cts    = new CancellationTokenSource(TimeSpan.FromMilliseconds(10));
        using var pricer = new SlowMCPricer(paths: 1000, cts.Token);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pricer.PriceAsync(cts.Token));
    }

    [Fact]
    public async Task PriceAsync_Cancelled_ExceptionCarriesToken()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        using var pricer = BuildPricer(S0, K, T, Sigma, Rd, Rf, true, cube);
        var market = MakeMarket(S0, Rd, Rf);

        var ex = await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pricer.PriceAsync(market, cts.Token));

        Assert.Equal(cts.Token, ex.CancellationToken);
    }

    [Fact]
    public async Task PriceAsync_MultipleConcurrentCalls_AllSucceed()
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        using var pricer = BuildPricer(S0, K, T, Sigma, Rd, Rf, true, cube);
        var gk     = GarmanKohlhagen.Price(S0, K, T, Sigma, Rd, Rf, isCall: true);
        var market = MakeMarket(S0, Rd, Rf);

        var tasks   = Enumerable.Range(0, 3).Select(_ => pricer.PriceAsync(market)).ToArray();
        var results = await Task.WhenAll(tasks);

        foreach (var r in results)
            AssertWithinMcBounds(r, gk, sigma: 5.0);
    }

    // ── PricingResult contract ────────────────────────────────────────────────

    [Fact]
    public void PricingResult_ConfidenceIntervalContainsPrice()
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var result = BuildPricer(S0, K, T, Sigma, Rd, Rf, true, cube).Price(market);

        Assert.True(result.ConfidenceIntervalLower <= result.Price);
        Assert.True(result.Price <= result.ConfidenceIntervalUpper);
        Assert.True(result.StandardError > 0.0);
        Assert.Equal(Paths, result.Paths);
    }

    [Fact]
    public void PricingResult_ConfidenceIntervalWidth_MatchesSe()
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        var market = MakeMarket(S0, Rd, Rf);
        var result = BuildPricer(S0, K, T, Sigma, Rd, Rf, true, cube).Price(market);

        // Full double-precision value of the 97.5th-percentile standard-normal quantile
        // (i.e. z for a two-sided 95% confidence interval).  The previously used
        // truncated literal 1.959964 differs from this in the 7th significant digit,
        // which caused the precision: 6 equality check to fail spuriously.
        const double z95 = 1.9599639845400536;
        Assert.Equal(result.ConfidenceIntervalWidth, 2.0 * z95 * result.StandardError, precision: 6);
    }

    // ── Parametric GK sweep ───────────────────────────────────────────────────

    [Theory]
    [InlineData(0.80, 1.10, 1.0,  0.15, 0.05, 0.02, true,  "OTM call deep")]
    [InlineData(1.30, 1.10, 1.0,  0.15, 0.05, 0.02, true,  "ITM call deep")]
    [InlineData(1.10, 1.10, 0.25, 0.20, 0.03, 0.01, true,  "ATM call short T")]
    [InlineData(1.10, 1.10, 2.0,  0.20, 0.04, 0.02, false, "ATM put long T")]
    [InlineData(1.10, 1.10, 1.0,  0.30, 0.00, 0.00, true,  "High vol, zero rates")]
    [InlineData(1.10, 1.10, 1.0,  0.20, 0.05, 0.07, true,  "Negative carry (rf > rd)")]
    public void ParametricSweep_MatchesGarmanKohlhagen(
        double spot, double strike, double expiry, double sigma,
        double rd, double rf, bool isCall, string _)
    {
        var (mc, gk) = RunAndCompare(spot, strike, expiry, sigma, rd, rf, isCall);
        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private (PricingResult mc, double gk) RunAndCompare(
        double spot, double strike, double expiry, double sigma,
        double rd, double rf, bool isCall)
    {
        var cube = SimulationCube.GenerateIndependent(Paths, Steps, 1, DefaultSeed);
        return RunAndCompareWithCube(spot, strike, expiry, sigma, rd, rf, isCall, cube);
    }

    private (PricingResult mc, double gk) RunAndCompareWithCube(
        double spot, double strike, double expiry, double sigma,
        double rd, double rf, bool isCall, SimulationCube cube)
    {
        var market = MakeMarket(spot, rd, rf);
        var mc     = BuildPricer(spot, strike, expiry, sigma, rd, rf, isCall, cube).Price(market);
        var gk     = GarmanKohlhagen.Price(spot, strike, expiry, sigma, rd, rf, isCall);
        return (mc, gk);
    }

    private static FXVanillaOptionMCPricer BuildPricer(
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
        return new FXVanillaOptionMCPricer(option, ValuationDate, sigma, cube);
    }

    private static FxMarketData MakeMarket(double spot, double rd, double rf) =>
        new()
        {
            CurrencyPair  = "EURUSD",
            Spot          = spot,
            DomesticRate  = ZeroCurve.Flat(rd, CurveMaturity),
            ForeignRate   = ZeroCurve.Flat(rf, CurveMaturity),
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

/// <summary>
/// MCBasePricer subclass whose SimulatePath sleeps for a fixed duration,
/// honouring the supplied CancellationToken so that cancellation tests
/// are not hardware-speed-dependent.
///
/// The token is captured at construction time and stored in a field so
/// that SimulatePath — which receives no token parameter from the base
/// class — can observe cancellation on every iteration.
///
/// Thread.Sleep(1) blocks for up to 1 ms without touching the token's
/// WaitHandle, making each path slow while remaining safe even if the
/// CancellationTokenSource is disposed concurrently. ThrowIfCancellationRequested()
/// is then called to surface the cancellation as OperationCanceledException,
/// which the base-class pricing loop propagates to the caller.
/// </summary>
file sealed class SlowMCPricer : MCBasePricer
{
    private readonly CancellationToken _token;

    public SlowMCPricer(int paths, CancellationToken token)
        : base(SimulationCube.GenerateIndependent(paths, steps: 1, assets: 1, seed: 0))
    {
        _token = token;
    }

    protected override double SimulatePath(int pathIndex)
    {
        // Block for ~1 ms without accessing the token's WaitHandle.
        // This avoids ObjectDisposedException if the CancellationTokenSource
        // is disposed while paths are still executing on background threads.
        Thread.Sleep(1);
        // Propagate cancellation as OperationCanceledException.
        // ThrowIfCancellationRequested() reads an internal volatile flag and
        // is safe to call even after the source has been disposed.
        _token.ThrowIfCancellationRequested();
        return 0.0;
    }
}
