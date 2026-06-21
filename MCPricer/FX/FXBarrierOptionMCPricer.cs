using Aegis.Instruments;
using RandomSimulator;
using NodaTime;

namespace MCPricer.FX;

/// <summary>
/// FX single-barrier option pricer (knock-in or knock-out, call or put).
///
/// Supported barrier types (Merton-Reiner-Rubinstein notation):
///   UP_AND_IN    payoff = vanilla if ∃t: S(t) ≥ H, else rebate
///   UP_AND_OUT   payoff = vanilla if ∀t: S(t) < H,  else rebate
///   DOWN_AND_IN  payoff = vanilla if ∃t: S(t) ≤ H, else rebate
///   DOWN_AND_OUT payoff = vanilla if ∀t: S(t) > H,  else rebate
///
/// Barrier observation modes:
///
///   DISCRETE:
///     Barrier is checked only at the MC time steps (monitoring dates).
///     Convergence in barrier accuracy requires many steps.
///
///   CONTINUOUS (Brownian bridge correction):
///     At each step, given S(t) and S(t+dt) on the same side of H, the
///     probability that a continuous Brownian path crossed H between steps is:
///
///       P(cross | S(t)=a, S(t+dt)=b, same side of H) =
///           exp(−2 · log(H/a) · log(H/b) / (σ²·dt))
///
///     Reference: Baldi, Caramellino, Iovino (1999) "Pricing General Barrier
///     Options: A Numerical Approach Using Sharp Large Deviations."
///
///     For reproducibility, bridge draws are seeded deterministically from
///     the path index (xor with a salt). A different seed per step index is
///     achieved by seeding a new Random per (path, step) pair.
///
/// Model: Garman-Kohlhagen GBM — see GbmOptionMCPricer class header.
/// </summary>
public sealed class FXBarrierOptionMCPricer : GbmOptionMCPricer
{
    private readonly double             _strike;
    private readonly double             _phi;          // +1 call, −1 put
    private readonly double             _barrierLevel;
    private readonly BarrierType        _barrierType;
    private readonly BarrierObservation _observation;
    private readonly double             _rebate;

    public FXBarrierOptionMCPricer(
        Option         option,
        LocalDate       valuationDate,
        double         volatility,
        SimulationCube cube)
        : base(option, valuationDate, volatility, cube)
    {
        if (option.KindCase != Option.KindOneofCase.Barrier)
            throw new ArgumentException("Option.Kind must be Barrier.", nameof(option));

        var barrier = option.Barrier;
        if (barrier.BarrierLevel <= 0)
            throw new ArgumentException("BarrierLevel must be positive.");
        if (option.Strike <= 0)
            throw new ArgumentException("Strike must be positive.", nameof(option));

        _strike       = option.Strike;
        _phi          = option.OptionType == OptionType.Call ? 1.0 : -1.0;
        _barrierLevel = barrier.BarrierLevel;
        _barrierType  = barrier.BarrierType;
        _observation  = barrier.Observation;
        _rebate       = barrier.Rebate;
    }

    protected override double EvaluatePayoff(ReadOnlySpan<double> spotPath)
    {
        var barrierHit = _observation == BarrierObservation.Discrete
            ? CheckDiscreteBarrier(spotPath)
            : CheckContinuousBarrier(spotPath);

        var terminal      = spotPath[^1];
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

    private bool CheckDiscreteBarrier(ReadOnlySpan<double> spotPath)
    {
        for (var i = 0; i < spotPath.Length; i++)
            if (IsBreached(spotPath[i])) return true;
        return false;
    }

    // ── Continuous monitoring (Brownian bridge) ───────────────────────────────

    private bool CheckContinuousBarrier(ReadOnlySpan<double> spotPath)
    {
        // Bridge draws are seeded per (path, step) for reproducibility.
        // Salt 0x9E3779B9 distributes the seed space; step is mixed in below.
        var pathSeed = CurrentPathIndex ^ unchecked((int)0x9E3779B9u);
        var sigSqDt  = SigSqDt;  // σ²·dt from base class (set by PrepareFromMarket)

        var prev = InitialSpot;   // was Market.Spot in old FXMCPricer
        for (var t = 0; t < spotPath.Length; t++)
        {
            var curr = spotPath[t];

            if (IsBreached(curr))
                return true;

            // If both endpoints are on the safe side, sample the crossing probability.
            if (!IsBreached(prev) && sigSqDt > 0.0)
            {
                var pCross = BridgeCrossingProb(prev, curr, _barrierLevel, sigSqDt);
                if (pCross > 0.0)
                {
                    // Deterministic per (path, step) — new Random is cheap for a single draw.
                    var u = new Random(pathSeed ^ (t * 1000003)).NextDouble();
                    if (u < pCross) return true;
                }
            }

            prev = curr;
        }
        return false;
    }

    /// <summary>
    /// Probability that a GBM path crosses barrier H between two monitored
    /// endpoints a = S(t) and b = S(t+dt), both on the same side of H.
    ///
    /// Formula (log-space Brownian bridge):
    ///   P = exp(−2 · log(H/a) · log(H/b) / (σ²·dt))
    ///
    /// Returns 1.0 if the endpoints are on opposite sides (crossing certain).
    /// Returns 0.0 if σ²·dt = 0 (no diffusion; deterministic path cannot cross).
    /// </summary>
    private static double BridgeCrossingProb(double a, double b, double h, double sigSqDt)
    {
        var logHa = Math.Log(h / a);
        var logHb = Math.Log(h / b);

        // Opposite sides: certain crossing.
        if (logHa * logHb <= 0.0)
            return 1.0;

        return Math.Exp(-2.0 * logHa * logHb / sigSqDt);
    }

    private bool IsBreached(double s) =>
        _barrierType is BarrierType.UpAndIn or BarrierType.UpAndOut
            ? s >= _barrierLevel
            : s <= _barrierLevel;

    /// <summary>
    /// Limitation: delta and gamma computed via bump-and-reprice have elevated
    /// variance when spot is near the barrier level. A spot bump can flip the
    /// barrier-hit status of marginal paths, creating a noisy finite-difference
    /// estimator. The result is mathematically correct in expectation but may
    /// require far more paths for a reliable Greek near the barrier.
    /// Pathwise differentiation or the likelihood ratio method give lower-variance
    /// barrier Greeks but require payoff-specific derivations.
    /// </summary>
    protected override MCBasePricer CreateBumped(BumpType bump, double epsilon)
    {
        var vol = Volatility;
        switch (bump)
        {
            case BumpType.VolUp:   vol += epsilon;                          break;
            case BumpType.VolDown: vol  = Math.Max(0.0, vol - epsilon);    break;
            default: throw new NotSupportedException($"Bump {bump} not supported by {GetType().Name}. Use ComputeGreeks(market, ...) for spot and rate bumps.");
        }
        return new FXBarrierOptionMCPricer(OptionDef, ValuationDate, vol, Cube);
    }
}
