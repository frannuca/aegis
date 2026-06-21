using Aegis.Instruments;
using RandomSimulator;
using NodaTime;

namespace MCPricer.FX;

/// <summary>
/// Abstract MC pricer for FX options under the Garman-Kohlhagen model.
///
/// Stochastic process (domestic risk-neutral measure, Q^d):
///   dS/S = (r_d(t) − r_f(t)) dt + σ dW^d
///
/// where:
///   S       = spot FX rate (domestic per unit of foreign, e.g. USD/EUR)
///   r_d(t)  = instantaneous domestic forward rate at time t
///   r_f(t)  = instantaneous foreign forward rate at time t (acts as carry)
///   σ       = flat implied volatility (lognormal Black-Scholes convention)
///   dW^d    = Brownian motion under Q^d
///
/// ── Rates from term structures ────────────────────────────────────────────────
/// Market.DomesticRate / Market.ForeignRate are zero-rate curves (Pillars).
/// At construction, the instantaneous forward rate over each simulation step
/// [t_i, t_{i+1}] is extracted via ZeroCurve.ForwardRate:
///
///   f_d(t_i, t_{i+1}) = (r_d(t_{i+1})·t_{i+1} − r_d(t_i)·t_i) / dt
///
/// The drift varies step-by-step, correctly reflecting the shape of the curve
/// for path-dependent products (barriers, Asians, etc.). For flat curves this
/// degenerates to the constant drift used previously.
///
/// DomesticRate / ForeignRate remain accessible as the terminal zero rates
/// r_d(T), r_f(T) — used for the discount factor and available to subclasses.
///
/// Exact log-Euler discretization per step i:
///   S(t_{i+1}) = S(t_i) · exp( (f_d(t_i,t_{i+1}) − f_f(t_i,t_{i+1}) − ½σ²)·dt + σ·√dt · Z_i )
///
/// where Z_i ~ N(0,1) is drawn from SimulationCube[path, i, 0].
///
/// Discount factor:
///   B(0,T) = exp(−r_d(T) · T)
///
/// ── Thread safety ────────────────────────────────────────────────────────────
/// SimulatePath is called concurrently by MCBasePricer.Price(). Thread safety
/// is achieved without locking via two ThreadLocal fields:
///
///   _spotBuffer       — one double[] per OS thread; written then immediately
///                       read within the same SimulatePath call on the same thread.
///
///   _currentPathIndex — one int per OS thread; set at the top of SimulatePath
///                       and read by EvaluatePayoff (e.g. for bridge RNG seeds)
///                       on the same thread before SimulatePath returns.
///
/// SimulationCube is read-only after construction and is safe to read from many
/// threads simultaneously.
///
/// Zero-vol edge case:
///   When σ = 0, _drifts[t] = (f_d − f_f)·dt and the path is deterministic.
///   SimulateSpotPath handles this without accessing the cube.
/// </summary>
public abstract class FXMCPricer : MCBasePricer
{
    protected readonly FxMarketData Market;
    protected readonly Option       OptionDef;
    protected readonly LocalDate     ValuationDate;
    protected readonly double       Dt;
    protected readonly double       Volatility;
    protected readonly double       DiscountFactor;

    /// <summary>Effective domestic zero rate r_d(T), interpolated from Market.DomesticRate at expiry T.</summary>
    protected readonly double DomesticRate;

    /// <summary>Effective foreign zero rate r_f(T), interpolated from Market.ForeignRate at expiry T.</summary>
    protected readonly double ForeignRate;

    // One spot buffer per thread: allocated once per thread on first use,
    // then reused for every path that thread processes. Avoids per-path allocation
    // while being fully thread-safe.
    private readonly ThreadLocal<double[]> _spotBuffer;

    // Stores the path index for the path currently being evaluated on each thread,
    // making it accessible to EvaluatePayoff overrides (e.g. for seeding bridge RNGs)
    // without passing it through the EvaluatePayoff signature.
    private readonly ThreadLocal<int> _currentPathIndex = new();

    private readonly double[] _drifts;    // per-step (f_d − f_f − ½σ²)·dt, length = cube.Steps
    private readonly double   _diffusion; // σ·√dt

    /// <summary>
    /// The index of the path currently being evaluated on the calling thread.
    /// Valid only during EvaluatePayoff; undefined between paths.
    /// </summary>
    protected int CurrentPathIndex => _currentPathIndex.Value;

    protected FXMCPricer(
        Option         option,
        LocalDate       valuationDate,
        FxMarketData   market,
        double         volatility,
        SimulationCube cube) : base(cube)
    {
        ArgumentNullException.ThrowIfNull(option);
        ArgumentNullException.ThrowIfNull(market);
        if (volatility < 0)
            throw new ArgumentOutOfRangeException(nameof(volatility), "Volatility must be ≥ 0.");
        if (market.Spot <= 0)
            throw new ArgumentException("Spot must be positive.", nameof(market));
        if (option.ExpiryYears <= 0)
            throw new ArgumentException("ExpiryYears must be positive.", nameof(option));

        Market         = market;
        OptionDef      = option;
        ValuationDate  = valuationDate;
        Volatility     = volatility;
        Dt             = option.ExpiryYears / cube.Steps;

        DomesticRate = ZeroCurve.InterpolateRate(market.DomesticRate, valuationDate, option.ExpiryYears);
        ForeignRate  = ZeroCurve.InterpolateRate(market.ForeignRate,  valuationDate, option.ExpiryYears);

        DiscountFactor = Math.Exp(-DomesticRate * option.ExpiryYears);

        _diffusion = volatility * Math.Sqrt(Dt);

        var halfSigSqDt = 0.5 * volatility * volatility * Dt;
        _drifts = new double[cube.Steps];
        for (var i = 0; i < cube.Steps; i++)
        {
            var t0 = i       * Dt;
            var t1 = (i + 1) * Dt;
            var fD = ZeroCurve.ForwardRate(market.DomesticRate, valuationDate, t0, t1);
            var fF = ZeroCurve.ForwardRate(market.ForeignRate,  valuationDate, t0, t1);
            _drifts[i] = (fD - fF) * Dt - halfSigSqDt;
        }

        var steps = cube.Steps;   // capture for the lambda (avoids closing over `cube`)
        _spotBuffer = new ThreadLocal<double[]>(() => new double[steps]);
    }

    /// <summary>
    /// Simulates a discretized GBM spot path and writes it into the calling
    /// thread's spot buffer.
    ///
    /// The returned span is valid only until the next SimulateSpotPath call on
    /// the same thread. Do not store it across calls.
    ///
    /// _spotBuffer.Value[t] = S at end of step t; last element = S(T).
    /// </summary>
    protected ReadOnlySpan<double> SimulateSpotPath(int pathIndex)
    {
        var buf = _spotBuffer.Value!;
        var s   = Market.Spot;

        if (Volatility == 0.0)
        {
            for (var t = 0; t < Cube.Steps; t++)
            {
                s      *= Math.Exp(_drifts[t]);
                buf[t]  = s;
            }
        }
        else
        {
            for (var t = 0; t < Cube.Steps; t++)
            {
                s      *= Math.Exp(_drifts[t] + _diffusion * Cube[pathIndex, t, 0]);
                buf[t]  = s;
            }
        }

        return buf.AsSpan(0, Cube.Steps);
    }

    protected sealed override double SimulatePath(int pathIndex)
    {
        _currentPathIndex.Value = pathIndex;
        var spotPath            = SimulateSpotPath(pathIndex);
        return DiscountFactor * EvaluatePayoff(spotPath);
    }

    /// <summary>
    /// Evaluates the undiscounted payoff given the realized spot path.
    ///
    /// spotPath[t] = S at the end of step t (1-indexed).
    /// spotPath[^1] = terminal spot S(T).
    ///
    /// Called from SimulatePath on whichever thread is processing this path.
    /// Must not write to shared mutable state. Access CurrentPathIndex (thread-local)
    /// for the current path index when needed (e.g. for deterministic RNG seeding).
    /// </summary>
    protected abstract double EvaluatePayoff(ReadOnlySpan<double> spotPath);

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _spotBuffer.Dispose();
            _currentPathIndex.Dispose();
        }
        base.Dispose(disposing);
    }
}
