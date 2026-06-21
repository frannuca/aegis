using Aegis.Instruments;
using RandomSimulator;
using NodaTime;

namespace MCPricer.FX;

/// <summary>
/// FX single-barrier option pricer under SABR stochastic volatility,
/// simulated under the T-forward measure.
///
/// ── Barrier semantics ────────────────────────────────────────────────────────
/// Supported barrier types (Merton-Reiner-Rubinstein notation):
///   UP_AND_IN    payoff = vanilla if ∃t: F(t) ≥ H, else rebate
///   UP_AND_OUT   payoff = vanilla if ∀t: F(t) < H,  else rebate
///   DOWN_AND_IN  payoff = vanilla if ∃t: F(t) ≤ H, else rebate
///   DOWN_AND_OUT payoff = vanilla if ∀t: F(t) > H,  else rebate
///
/// The barrier is monitored on the **forward F** (not the spot S), because
/// SABR is simulated under the T-forward measure where F is a martingale.
/// This is consistent with how FX barrier options are quoted on the forward.
///
/// ── Observation modes ────────────────────────────────────────────────────────
///
///   DISCRETE:
///     Barrier is tested only at the MC time steps (monitoring dates).
///
///   CONTINUOUS (Brownian bridge correction):
///     At each step, given F(t) and F(t+dt) both on the safe side of H,
///     the probability that a continuous path crossed H is estimated via:
///
///       P(cross | F(t)=a, F(t+dt)=b, same side of H) =
///           exp(−2 · log(H/a) · log(H/b) / (v_beg² · dt))
///
///     where v_beg is the SABR vol σ at the beginning of the step.
///
///     This formula is exact for β=1 (lognormal SABR), where the forward
///     follows dF = σ·F·dW and log(F) is a Brownian bridge conditioned on
///     its endpoints. For β≠1 it is an approximation that treats the CEV
///     diffusion as locally lognormal over the step.
///
///     Reference: Baldi, Caramellino, Iovino (1999) "Pricing General Barrier
///     Options: A Numerical Approach Using Sharp Large Deviations."
///
/// ── KI + KO = Vanilla identity ───────────────────────────────────────────────
/// For a fixed barrier H and zero rebates:
///   KI price + KO price = SABR vanilla price
/// This holds path-by-path (not just in expectation) and provides a model-free
/// numerical sanity check. It is verified in the test suite.
///
/// ── ν = 0 degeneracy ─────────────────────────────────────────────────────────
/// When ν = 0, vol-of-vol is zero and SABR reduces to a GBM with constant
/// vol σ = α. The SABR barrier price must then agree with FXBarrierOptionMCPricer
/// priced at the same cube with σ = α (to within MC sampling noise). This is
/// tested with the same-cube construction and is the strongest correctness check.
/// </summary>
public sealed class FXSABRBarrierOptionMCPricer : FXSABRMCPricer
{
    private readonly double             _strike;
    private readonly double             _phi;          // +1 call, −1 put
    private readonly double             _barrierLevel;
    private readonly BarrierType        _barrierType;
    private readonly BarrierObservation _observation;
    private readonly double             _rebate;

    public FXSABRBarrierOptionMCPricer(
        Option          option,
        LocalDate       valuationDate,
        SabrParameters  sabr,
        SimulationCube  cube)
        : base(option, valuationDate, sabr, cube)
    {
        if (option.KindCase != Option.KindOneofCase.Barrier)
            throw new ArgumentException("Option.Kind must be Barrier.", nameof(option));

        var barrier = option.Barrier;
        if (barrier.BarrierType == BarrierType.Unspecified)
            throw new ArgumentException("BarrierType must be specified.", nameof(option));
        if (barrier.Observation == BarrierObservation.Unspecified)
            throw new ArgumentException("BarrierObservation must be specified.", nameof(option));
        if (barrier.BarrierLevel <= 0)
            throw new ArgumentException("BarrierLevel must be positive.", nameof(option));
        if (option.Strike <= 0)
            throw new ArgumentException("Strike must be positive.", nameof(option));

        _strike       = option.Strike;
        _phi          = option.OptionType == OptionType.Call ? 1.0 : -1.0;
        _barrierLevel = barrier.BarrierLevel;
        _barrierType  = barrier.BarrierType;
        _observation  = barrier.Observation;
        _rebate       = barrier.Rebate;
    }

    protected override double EvaluatePayoff(
        ReadOnlySpan<double> fwdPath,
        ReadOnlySpan<double> volPath)
    {
        var barrierHit = _observation == BarrierObservation.Discrete
            ? CheckDiscreteBarrier(fwdPath)
            : CheckContinuousBarrier(fwdPath, volPath);

        var terminal      = fwdPath[^1];
        var vanillaPayoff = Math.Max(_phi * (terminal - _strike), 0.0);

        return _barrierType switch
        {
            BarrierType.UpAndIn  or BarrierType.DownAndIn
                => barrierHit ? vanillaPayoff : _rebate,
            BarrierType.UpAndOut or BarrierType.DownAndOut
                => barrierHit ? _rebate : vanillaPayoff,
            _ => throw new InvalidOperationException($"Unknown barrier type {_barrierType}.")
        };
    }

    // ── Discrete monitoring ───────────────────────────────────────────────────

    private bool CheckDiscreteBarrier(ReadOnlySpan<double> fwdPath)
    {
        for (var i = 0; i < fwdPath.Length; i++)
            if (IsBreached(fwdPath[i])) return true;
        return false;
    }

    // ── Continuous monitoring (Brownian bridge) ───────────────────────────────

    private bool CheckContinuousBarrier(ReadOnlySpan<double> fwdPath, ReadOnlySpan<double> volPath)
    {
        // Salt from path index for reproducible bridge draws, one fresh Random per (path, step).
        var pathSeed = CurrentPathIndex ^ unchecked((int)0x9E3779B9u);

        // Starting point of the first interval is the initial forward F(0).
        var prev    = InitialForward;
        var prevVol = Sabr.Alpha;   // beginning-of-step vol for step t=0

        for (var t = 0; t < fwdPath.Length; t++)
        {
            var curr = fwdPath[t];

            if (IsBreached(curr))
                return true;

            // Sample bridge crossing probability when both endpoints are on the safe side.
            if (!IsBreached(prev) && prevVol > 0.0)
            {
                var sigSqDt = prevVol * prevVol * Dt;
                var pCross  = BridgeCrossingProb(prev, curr, _barrierLevel, sigSqDt);
                if (pCross > 0.0)
                {
                    var u = new Random(pathSeed ^ (t * 1000003)).NextDouble();
                    if (u < pCross) return true;
                }
            }

            prev    = curr;
            prevVol = volPath[t];  // end-of-step vol = beginning-of-step vol for next step
        }
        return false;
    }

    /// <summary>
    /// Log-space Brownian bridge crossing probability.
    ///
    /// P = exp(−2 · log(H/a) · log(H/b) / (σ²·dt))
    ///
    /// Returns 1.0 if a and b are on opposite sides of H.
    /// Returns 0.0 if sigSqDt = 0.
    /// </summary>
    private static double BridgeCrossingProb(double a, double b, double h, double sigSqDt)
    {
        if (sigSqDt <= 0.0) return 0.0;
        var logHa = Math.Log(h / a);
        var logHb = Math.Log(h / b);
        if (logHa * logHb <= 0.0) return 1.0;   // opposite sides — certain crossing
        return Math.Exp(-2.0 * logHa * logHb / sigSqDt);
    }

    private bool IsBreached(double f) =>
        _barrierType is BarrierType.UpAndIn or BarrierType.UpAndOut
            ? f >= _barrierLevel
            : f <= _barrierLevel;

    /// <summary>
    /// Creates a new pricer with one SABR parameter bumped by epsilon.
    /// Reuses the same SimulationCube for common-random-number variance reduction.
    ///
    /// BumpType mapping:
    ///   VolUp/Down     → Sabr.Alpha
    ///   SabrNuUp/Down  → Sabr.Nu
    ///   SabrRhoUp/Down → Sabr.Rho (clamped to (−1,1))
    ///
    /// Spot and rate bumps are handled by ComputeGreeks(IMarketData) via
    /// MarketDataBumps on the same instance.
    /// </summary>
    protected override MCBasePricer CreateBumped(BumpType bump, double epsilon)
    {
        var sabr = Sabr.Clone();

        switch (bump)
        {
            case BumpType.VolUp:       sabr.Alpha += epsilon;                                     break;
            case BumpType.VolDown:     sabr.Alpha  = Math.Max(1e-8, sabr.Alpha - epsilon);        break;
            case BumpType.SabrNuUp:    sabr.Nu    += epsilon;                                     break;
            case BumpType.SabrNuDown:  sabr.Nu     = Math.Max(0.0, sabr.Nu - epsilon);           break;
            case BumpType.SabrRhoUp:   sabr.Rho    = Math.Min(1.0 - 1e-6, sabr.Rho + epsilon);  break;
            case BumpType.SabrRhoDown: sabr.Rho    = Math.Max(-1.0 + 1e-6, sabr.Rho - epsilon); break;
            default: throw new NotSupportedException(
                $"Bump {bump} not supported by {GetType().Name}. " +
                "Use ComputeGreeks(market, ...) for spot and rate bumps.");
        }

        return new FXSABRBarrierOptionMCPricer(OptionDef, ValuationDate, sabr, Cube);
    }
}
