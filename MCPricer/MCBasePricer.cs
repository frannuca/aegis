using Aegis.Instruments;
using RandomSimulator;

namespace MCPricer;

/// <summary>
/// Abstract base class for all Monte Carlo pricers.
///
/// Pricing framework:
///   Price = E_Q[ B(0,T) · payoff(ω) ]
///
/// Derived classes implement the SDE dynamics of the underlying(s) and compute
/// the discounted payoff per path. The base class aggregates paths and returns
/// a 95% confidence interval.
///
/// Numeraire convention: risk-neutral measure Q. Each derived class multiplies
/// the path payoff by the appropriate discount factor before returning from
/// SimulatePath.
///
/// ── Concurrency ──────────────────────────────────────────────────────────────
/// Both Price() and PriceAsync() dispatch path evaluation via Parallel.For,
/// using all available cores. SimulatePath must therefore be thread-safe.
/// Derived classes achieve this through ThreadLocal scratch storage
/// (see FXMCPricer, EQMCPricer).
///
/// Floating-point note: parallel summation changes the order of additions.
/// Results are statistically equivalent but not bit-for-bit reproducible
/// across runs with different thread counts or OS scheduling.
///
/// ── Cancellation ─────────────────────────────────────────────────────────────
/// PriceAsync(ct) honours cancellation at two levels:
///   1. ParallelOptions.CancellationToken — the framework stops dispatching new
///      work items when ct is signalled.
///   2. ct.ThrowIfCancellationRequested() at the start of each path — ensures
///      cancellation is observed promptly even during a long burst of paths on
///      one thread, not just between work chunks.
///
/// When cancelled, PriceAsync() propagates OperationCanceledException with the
/// original token, and the returned Task transitions to the Canceled state.
///
/// Convergence: O(1/√N) pseudo-random, O((log N)^d / N) Sobol quasi-random.
/// Antithetic variates (built into SimulationCube) add variance reduction.
/// </summary>
public abstract class MCBasePricer : IDisposable
{
    protected readonly SimulationCube Cube;
    private bool _disposed;

    protected MCBasePricer(SimulationCube cube)
    {
        ArgumentNullException.ThrowIfNull(cube);
        Cube = cube;
    }

    /// <summary>
    /// Computes the discount(T) × payoff for a single Monte Carlo path.
    /// Called concurrently from multiple threads — must be thread-safe.
    /// </summary>
    protected abstract double SimulatePath(int pathIndex);

    /// <summary>
    /// Prices the instrument synchronously using all available cores.
    /// Blocks the calling thread until all paths have been evaluated.
    /// For non-blocking execution use PriceAsync.
    /// </summary>
    public PricingResult Price() => RunParallel(CancellationToken.None);

    /// <summary>
    /// Prices the instrument using the supplied market data snapshot.
    /// Override in derived classes; the base implementation throws NotSupportedException.
    /// </summary>
    public virtual PricingResult Price(IMarketData market)
        => throw new NotSupportedException(
            $"{GetType().Name} does not support Price(IMarketData). " +
            "Call the type-specific Price overload instead.");

    /// <summary>
    /// Prices the instrument asynchronously using the supplied market data snapshot.
    /// Override in derived classes; the base implementation throws NotSupportedException.
    /// </summary>
    public virtual Task<PricingResult> PriceAsync(IMarketData market, CancellationToken ct = default)
        => throw new NotSupportedException(
            $"{GetType().Name} does not support PriceAsync(IMarketData). " +
            "Call the type-specific PriceAsync overload instead.");

    /// <summary>
    /// Prices the instrument asynchronously. Offloads the parallel path
    /// evaluation to the thread pool so the caller is not blocked.
    ///
    /// Cancellation: passing a signalled or subsequently cancelled token
    /// causes the task to throw OperationCanceledException and transition
    /// to the Canceled state. Cancellation is checked both by the Parallel.For
    /// infrastructure and at the start of every individual path.
    /// </summary>
    /// <param name="ct">
    /// Token used to cancel the pricing run. Defaults to CancellationToken.None.
    /// </param>
    public Task<PricingResult> PriceAsync(CancellationToken ct = default)
        => Task.Run(() => RunParallel(ct), ct);

    // ── Internal parallel runner ───────────────────────────────────────────────

    private PricingResult RunParallel(CancellationToken ct)
    {
        var n       = Cube.Paths;
        var sum     = 0.0;
        var sumSq   = 0.0;
        var syncObj = new object();

        Parallel.For(
            0, n,
            new ParallelOptions { CancellationToken = ct },
            localInit: () => (sum: 0.0, sumSq: 0.0),
            body: (i, _, local) =>
            {
                // Per-path check: ensures cancellation is observed on the very
                // next path boundary, not only between Parallel.For work chunks.
                ct.ThrowIfCancellationRequested();
                var v = SimulatePath(i);
                return (local.sum + v, local.sumSq + v * v);
            },
            localFinally: local =>
            {
                // Fires exactly once per thread at completion (or cancellation).
                // The lock is uncontended in practice — one acquire per thread.
                lock (syncObj)
                {
                    sum   += local.sum;
                    sumSq += local.sumSq;
                }
            });

        return BuildResult(n, sum, sumSq);
    }

    /// <summary>
    /// Computes PricingResult from accumulated sum and sum-of-squares.
    ///
    /// Standard error of the mean (one-pass formula):
    ///   SE² = (sumSq/N − mean²) / (N−1)   [= unbiased sample variance / N]
    ///
    /// 95% confidence interval: mean ± 1.96 × SE.
    /// </summary>
    private static PricingResult BuildResult(int n, double sum, double sumSq)
    {
        var mean      = sum / n;
        var sampleVar = (sumSq / n - mean * mean) / (n - 1);
        var stderr    = Math.Sqrt(Math.Max(0.0, sampleVar));
        const double z95 = 1.959964;
        return new PricingResult(
            Price:                   mean,
            StandardError:           stderr,
            ConfidenceIntervalLower: mean - z95 * stderr,
            ConfidenceIntervalUpper: mean + z95 * stderr,
            Paths:                   n);
    }

    // ── Greeks ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Computes first- and second-order price sensitivities via bump-and-reprice.
    ///
    /// Each Greek requires one or two additional full MC pricing runs on a bumped
    /// copy of the pricer. Using the same SimulationCube for the bumped pricers
    /// (common random numbers) reduces variance in the finite-difference estimator.
    ///
    /// Bump sizes (all absolute, not relative):
    ///   spotEps  — spot bump in the same units as the market spot (default 0.01)
    ///   volEps   — vol bump in annual vol units (default 0.001 = 10 bps)
    ///   rateEps  — rate bump in annual rate units (default 0.0001 = 1 bp)
    ///
    /// Theta is set to NaN because bumping T requires rebuilding the SimulationCube
    /// with a different step count, destroying the common-random-number variance
    /// reduction. Use an analytical formula for Theta, or override ComputeGreeks.
    ///
    /// Rho convention: domestic rate for FX, risk-free rate for equity.
    ///
    /// Requires derived class to override CreateBumped.
    /// </summary>
    public GreekResult ComputeGreeks(
        double spotEps = 0.01,
        double volEps  = 0.001,
        double rateEps = 0.0001)
    {
        var mid  = Price().Price;
        var sUp  = CreateBumped(BumpType.SpotUp,   spotEps).Price().Price;
        var sDn  = CreateBumped(BumpType.SpotDown,  spotEps).Price().Price;
        var vUp  = CreateBumped(BumpType.VolUp,     volEps ).Price().Price;
        var vDn  = CreateBumped(BumpType.VolDown,   volEps ).Price().Price;
        var rUp  = CreateBumped(BumpType.RateUp,    rateEps).Price().Price;
        var rDn  = CreateBumped(BumpType.RateDown,  rateEps).Price().Price;

        return new GreekResult(
            Delta: (sUp - sDn)            / (2.0 * spotEps),
            Gamma: (sUp - 2.0 * mid + sDn) / (spotEps * spotEps),
            Vega:  (vUp - vDn)            / (2.0 * volEps),
            Theta: double.NaN,
            Rho:   (rUp - rDn)            / (2.0 * rateEps));
    }

    /// <summary>
    /// Returns a new pricer of the same concrete type with one market parameter
    /// shifted by epsilon. The same SimulationCube is reused (common random numbers).
    ///
    /// Override in each concrete pricer. The base implementation throws
    /// NotSupportedException so that ComputeGreeks fails clearly rather than silently
    /// returning garbage if a derived class forgets to override.
    ///
    /// SABR note: override this in SABR-based pricers so that BumpType.VolUp
    /// bumps the SABR alpha parameter instead of a non-existent flat sigma.
    /// Vol-of-vol (nu) and correlation sensitivity require additional BumpType
    /// values or a fully custom ComputeGreeks override.
    /// </summary>
    protected virtual MCBasePricer CreateBumped(BumpType bump, double epsilon)
        => throw new NotSupportedException(
            $"{GetType().Name} does not support bump-and-reprice Greeks. " +
            $"Override CreateBumped to enable ComputeGreeks.");

    // ── Disposal ──────────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        Dispose(disposing: true);
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing) { }
}
