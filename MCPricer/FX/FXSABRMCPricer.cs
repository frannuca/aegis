using Aegis.Instruments;
using NodaTime;
using RandomSimulator;

namespace MCPricer.FX;

/// <summary>
/// Abstract MC pricer for FX options under the SABR stochastic volatility model
/// (Hagan, Kumar, Lesniewski, Woodward 2002), simulated under the T-forward measure.
///
/// ── Model (T-forward measure Q^T) ────────────────────────────────────────────
///
///   dF  = σ · C(F) · dW₁^T          F = forward FX rate, martingale under Q^T
///   dσ  = ν · σ · dW₂^T             lognormal vol-of-vol
///   dW₁^T · dW₂^T = ρ dt
///
/// where the CEV function C(F) depends on β:
///   β = 0   → C(F) = 1              (normal / additive diffusion)
///   β = 1   → C(F) = F              (lognormal / multiplicative diffusion)
///   0 < β < 1 → C(F) = F^β          (CEV; absorbing boundary at F = 0)
///
/// Initial conditions:
///   F(0) = S₀ · exp((r_d − r_f) · T)    (FX forward under covered-interest parity)
///   σ(0) = α                             (SABR initial vol)
///
/// Option price:
///   Price = P(0,T) · E^T[payoff(F(T), σ(T))]
///   P(0,T) = exp(−r_d · T)
///
/// Under the T-forward measure, the forward F is a martingale — no drift term.
/// Rates are interpolated once at T (not per-step), since the SABR SDE has no
/// drift in the forward process.
///
/// ── Discretization ────────────────────────────────────────────────────────────
///
/// Vol (σ): exact log-Euler (no bias for lognormal SDE with fixed F):
///   σ(t+dt) = σ(t) · exp(ν·√dt·W₂ − ½ν²·dt)
///
/// Forward (F), using beginning-of-step σ:
///
///   β = 1 (lognormal):
///     F(t+dt) = F(t) · exp(−½σ²·dt + σ·√dt·Z₁)     [log-Euler, no negative F]
///
///   β = 0 (normal):
///     F(t+dt) = F(t) + σ·√dt·Z₁                      [exact for constant σ]
///
///   0 < β < 1 (CEV Euler with absorbing boundary):
///     F(t+dt) = max(F(t) + σ · max(F(t),0)^β · √dt · Z₁, 0)
///
/// ── Brownian decomposition ────────────────────────────────────────────────────
/// The cube provides TWO independent N(0,1) factors per step:
///   Z₁ = Cube[path, t, 0]   → drives the forward F
///   Z₂ = Cube[path, t, 1]   → independent noise
///
/// The correlated vol Brownian is reconstructed inside the pricer:
///   W₂ = ρ · Z₁ + √(1−ρ²) · Z₂
///
/// This keeps ρ as a model parameter: bumping ρ in CreateBumped never requires
/// rebuilding the SimulationCube.
///
/// ── Convergence ───────────────────────────────────────────────────────────────
/// SABR vol (log-Euler) has no time-step bias.
/// Forward (Euler) has O(√dt) weak error; 50–252 steps recommended.
/// For β = 1, log-Euler eliminates the Euler bias on F.
/// Benchmark: compare to Hagan (2002) analytical approximation.
///
/// ── Thread safety ─────────────────────────────────────────────────────────────
/// _buffers is ThreadLocal; each thread owns its (fwdBuf, volBuf) pair.
/// SimulationCube is read-only after construction. Safe for Parallel.For.
/// Sequential repricing with different IMarketData is supported; concurrent
/// repricing on the same instance with different markets is NOT safe.
/// </summary>
public abstract class FXSABRMCPricer : MCBasePricer
{
    protected readonly Option          OptionDef;
    protected readonly LocalDate       ValuationDate;
    protected readonly SabrParameters  Sabr;
    protected readonly double          Dt;

    // Set by PrepareFromMarket; valid during and after the most recent Price(market) call.
    protected double InitialForward  { get; private set; }
    protected double DiscountFactor  { get; private set; }
    protected double DomesticRate    { get; private set; }
    protected double ForeignRate     { get; private set; }

    // Precomputed constants for the hot path (depend only on Sabr, not market)
    private readonly double _sqrtDt;
    private readonly double _volDriftAdj;    // −½ν²·dt  (exact log-Euler drift for σ)
    private readonly double _rhoComplement;  // √(1−ρ²)
    private readonly bool   _isLognormal;    // β = 1 exactly
    private readonly bool   _isNormal;       // β = 0 exactly

    // One (fwdBuf, volBuf) pair per thread, reused across paths
    private readonly ThreadLocal<(double[] fwd, double[] vol)> _buffers;
    private readonly ThreadLocal<int> _currentPathIndex = new();

    /// <summary>
    /// The index of the path currently being evaluated on the calling thread.
    /// Valid only during EvaluatePayoff; undefined between paths.
    /// </summary>
    protected int CurrentPathIndex => _currentPathIndex.Value;

    protected FXSABRMCPricer(
        Option          option,
        LocalDate       valuationDate,
        SabrParameters  sabr,
        SimulationCube  cube) : base(cube)
    {
        ArgumentNullException.ThrowIfNull(option);
        ArgumentNullException.ThrowIfNull(sabr);

        if (cube.Assets < 2)
            throw new ArgumentException(
                "SABR simulation requires a cube with at least 2 assets " +
                "(asset 0 = forward factor Z₁, asset 1 = independent vol factor Z₂).",
                nameof(cube));
        if (option.ExpiryYears <= 0)
            throw new ArgumentException("ExpiryYears must be positive.", nameof(option));
        if (sabr.Alpha <= 0)
            throw new ArgumentOutOfRangeException(nameof(sabr), "Alpha must be > 0.");
        if (sabr.Beta is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(sabr), "Beta must be in [0, 1].");
        if (sabr.Nu < 0)
            throw new ArgumentOutOfRangeException(nameof(sabr), "Nu must be ≥ 0.");
        if (sabr.Rho is <= -1 or >= 1)
            throw new ArgumentOutOfRangeException(nameof(sabr), "Rho must be in (−1, 1).");

        OptionDef     = option;
        ValuationDate = valuationDate;
        Sabr          = sabr;
        Dt            = option.ExpiryYears / cube.Steps;

        _sqrtDt        = Math.Sqrt(Dt);
        _volDriftAdj   = -0.5 * sabr.Nu * sabr.Nu * Dt;
        _rhoComplement = Math.Sqrt(1.0 - sabr.Rho * sabr.Rho);
        _isLognormal   = sabr.Beta == 1.0;
        _isNormal      = sabr.Beta == 0.0;

        var steps = cube.Steps;
        _buffers = new ThreadLocal<(double[], double[])>(() =>
            (new double[steps], new double[steps]));
    }

    // ── Market-aware pricing entry points ─────────────────────────────────────

    public override PricingResult Price(IMarketData market)
    {
        PrepareFromMarket(market);
        return base.Price();
    }

    public override Task<PricingResult> PriceAsync(IMarketData market, CancellationToken ct = default)
    {
        PrepareFromMarket(market);
        return base.PriceAsync(ct);
    }

    private void PrepareFromMarket(IMarketData market)
    {
        if (market.Spot <= 0)
            throw new ArgumentException("Spot must be positive.", nameof(market));

        var T = OptionDef.ExpiryYears;
        DomesticRate   = ZeroCurve.InterpolateRate(market.DiscountCurve, ValuationDate, T);
        ForeignRate    = ZeroCurve.InterpolateRate(market.CarryCurve,    ValuationDate, T);
        DiscountFactor = Math.Exp(-DomesticRate * T);
        InitialForward = market.Spot * Math.Exp((DomesticRate - ForeignRate) * T);
    }

    // ── Market-aware Greeks ────────────────────────────────────────────────────

    /// <summary>
    /// Computes first- and second-order price sensitivities via bump-and-reprice.
    /// Spot and rate bumps reprice on this instance with bumped market data.
    /// Vol bumps (alpha) create a new pricer instance with bumped Alpha.
    /// Theta = NaN.
    /// </summary>
    public GreekResult ComputeGreeks(
        IMarketData market,
        double spotEps = 0.01,
        double volEps  = 0.001,
        double rateEps = 0.0001)
    {
        var mid = Price(market).Price;
        var sUp = Price(MarketDataBumps.BumpSpot(market, +spotEps)).Price;
        var sDn = Price(MarketDataBumps.BumpSpot(market, -spotEps)).Price;
        var vUp = CreateBumped(BumpType.VolUp,   volEps).Price(market).Price;
        var vDn = CreateBumped(BumpType.VolDown, volEps).Price(market).Price;
        var rUp = Price(MarketDataBumps.BumpDiscountCurve(market, +rateEps)).Price;
        var rDn = Price(MarketDataBumps.BumpDiscountCurve(market, -rateEps)).Price;

        return new GreekResult(
            Delta: (sUp - sDn)             / (2.0 * spotEps),
            Gamma: (sUp - 2.0 * mid + sDn) / (spotEps * spotEps),
            Vega:  (vUp - vDn)             / (2.0 * volEps),
            Theta: double.NaN,
            Rho:   (rUp - rDn)             / (2.0 * rateEps));
    }

    // ── SABR-specific Greek helpers ────────────────────────────────────────────

    /// <summary>
    /// Sensitivity to vol-of-vol ν (dV/dν), computed by central finite differences.
    /// Reuses the same SimulationCube (common random numbers).
    ///
    /// nuEps: absolute bump in ν (default 0.01 = 1 vol-of-vol point).
    /// Returns NaN if the bumped ν would go below 0.
    /// </summary>
    public double ComputeVolOfVolSensitivity(IMarketData market, double nuEps = 0.01)
    {
        var up  = CreateBumped(BumpType.SabrNuUp,   nuEps).Price(market).Price;
        var dn  = CreateBumped(BumpType.SabrNuDown, nuEps).Price(market).Price;
        return (up - dn) / (2.0 * nuEps);
    }

    /// <summary>
    /// Sensitivity to Brownian correlation ρ (dV/dρ), computed by central finite
    /// differences. Bumped ρ is clamped to (−1+ε, 1−ε) to stay in the valid range.
    ///
    /// rhoEps: absolute bump in ρ (default 0.01).
    /// </summary>
    public double ComputeCorrelSensitivity(IMarketData market, double rhoEps = 0.01)
    {
        var up  = CreateBumped(BumpType.SabrRhoUp,   rhoEps).Price(market).Price;
        var dn  = CreateBumped(BumpType.SabrRhoDown, rhoEps).Price(market).Price;
        return (up - dn) / (2.0 * rhoEps);
    }

    // ── Path simulation ────────────────────────────────────────────────────────

    /// <summary>
    /// Simulates one SABR path (F, σ) into the calling thread's buffers.
    /// Access results via GetPathBuffers() immediately after calling this method.
    ///
    /// fwdBuf[t] = F at end of step t;  fwdBuf[Steps-1] = F(T)
    /// volBuf[t] = σ at end of step t;  volBuf[Steps-1] = σ(T)
    ///
    /// Valid only until the next SimulateSabrPath call on the same thread.
    /// </summary>
    protected void SimulateSabrPath(int pathIndex)
    {
        var (fwdBuf, volBuf) = _buffers.Value!;

        var f = InitialForward;
        var v = Sabr.Alpha;

        for (var t = 0; t < Cube.Steps; t++)
        {
            var z1 = Cube[pathIndex, t, 0];   // drives forward
            var z2 = Cube[pathIndex, t, 1];   // independent
            var wv = Sabr.Rho * z1 + _rhoComplement * z2;  // correlated vol Brownian

            // Forward step (beginning-of-step vol v):
            f = StepForward(f, v, z1);
            fwdBuf[t] = f;

            // Vol step — exact log-Euler for lognormal vol SDE:
            //   σ(t+dt) = σ(t)·exp(ν·√dt·W₂ − ½ν²·dt)
            v *= Math.Exp(Sabr.Nu * _sqrtDt * wv + _volDriftAdj);
            volBuf[t] = v;
        }
    }

    /// <summary>Returns the thread-local (fwdBuf, volBuf) filled by SimulateSabrPath.</summary>
    protected (double[] fwdBuf, double[] volBuf) GetPathBuffers() => _buffers.Value!;

    private double StepForward(double f, double v, double z1)
    {
        if (_isLognormal)
        {
            // Log-Euler for β=1: exact simulation for GBM with fixed v.
            // Prevents negative F. F=0 is absorbing (exp(−∞) = 0).
            return f > 0.0
                ? f * Math.Exp(-0.5 * v * v * Dt + v * _sqrtDt * z1)
                : 0.0;
        }

        if (_isNormal)
        {
            // β=0: additive diffusion; F can be negative (appropriate for rates).
            return f + v * _sqrtDt * z1;
        }

        // General CEV Euler with absorbing boundary at 0:
        //   F(t+dt) = max(F(t) + v · max(F(t),0)^β · √dt · Z₁, 0)
        var fPlus = Math.Max(f, 0.0);
        return Math.Max(f + v * Math.Pow(fPlus, Sabr.Beta) * _sqrtDt * z1, 0.0);
    }

    protected sealed override double SimulatePath(int pathIndex)
    {
        if (InitialForward == 0 && DiscountFactor == 0)
            throw new InvalidOperationException(
                "Market data has not been supplied. Call Price(IMarketData market) instead of Price().");
        _currentPathIndex.Value = pathIndex;
        SimulateSabrPath(pathIndex);
        var (fwdBuf, volBuf) = GetPathBuffers();
        return DiscountFactor * EvaluatePayoff(
            fwdBuf.AsSpan(0, Cube.Steps),
            volBuf.AsSpan(0, Cube.Steps));
    }

    /// <summary>
    /// Evaluates the undiscounted payoff for one simulated path.
    ///
    /// fwdPath[t] = F at end of step t;  fwdPath[^1] = terminal forward F(T).
    /// volPath[t] = σ at end of step t;  volPath[^1] = terminal vol σ(T).
    ///
    /// volPath is useful for exotic payoffs (variance swaps, cliquets, vol targets).
    /// For vanilla options, only fwdPath[^1] is needed.
    ///
    /// Must not write to shared mutable state.
    /// </summary>
    protected abstract double EvaluatePayoff(
        ReadOnlySpan<double> fwdPath,
        ReadOnlySpan<double> volPath);

    // ── Disposal ──────────────────────────────────────────────────────────────

    protected override void Dispose(bool disposing)
    {
        if (disposing) { _buffers.Dispose(); _currentPathIndex.Dispose(); }
        base.Dispose(disposing);
    }
}
