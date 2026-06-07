using Aegis.Instruments;
using RandomSimulator;

namespace MCPricer.Equity;

/// <summary>
/// Abstract MC pricer for equity options under the Black-Scholes GBM model.
///
/// Stochastic process (risk-neutral measure Q):
///   dS/S = (r − q) dt + σ dW
///
/// where:
///   S   = equity spot price
///   r   = continuously compounded risk-free rate
///   q   = continuously compounded dividend yield
///   σ   = flat implied volatility (lognormal Black-Scholes convention)
///   dW  = Brownian motion under Q
///
/// ── Rates from term structures ────────────────────────────────────────────────
/// Market.RiskFreeRate / Market.DividendYield are term structures (Pillars), not
/// flat scalars. At construction, each curve is interpolated once — at the
/// option's expiry T, from the supplied valuation date, via
/// ZeroCurve.InterpolateRate — to obtain single effective values r = r(T),
/// q = q(T). The simulated process is then exactly the flat-rate GBM above
/// over [0, T]: the term structure only determines *which* flat rate/yield
/// applies to this trade.
///
/// Exact log-Euler discretization (no time-step bias for GBM):
///   S(t+dt) = S(t) · exp( (r − q − ½σ²)·dt + σ·√dt · Z_t )
///
/// where Z_t ~ N(0,1) is drawn from SimulationCube[path, step, 0].
///
/// Discount factor:
///   B(0,T) = exp(−r · T)
///
/// ── Thread safety ────────────────────────────────────────────────────────────
/// Mirrors FXMCPricer exactly. _spotBuffer and _currentPathIndex are ThreadLocal
/// so SimulatePath is safe to call concurrently from MCBasePricer.Price().
///
/// Note: structurally isomorphic to FXMCPricer (r_d = r, r_f = q). Kept as a
/// separate class to preserve clear domain semantics and allow independent
/// specialisation.
///
/// Zero-vol edge case:
///   σ = 0: path is deterministic. SimulateSpotPath handles this without
///   reading from the SimulationCube.
/// </summary>
public abstract class EQMCPricer : MCBasePricer
{
    protected readonly EqMarketData Market;
    protected readonly Option       OptionDef;
    protected readonly DateOnly     ValuationDate;
    protected readonly double       Dt;
    protected readonly double       Volatility;
    protected readonly double       DiscountFactor;

    /// <summary>Effective risk-free rate r(T), interpolated from Market.RiskFreeRate at expiry T.</summary>
    protected readonly double RiskFreeRate;

    /// <summary>Effective dividend yield q(T), interpolated from Market.DividendYield at expiry T.</summary>
    protected readonly double DividendYield;

    private readonly ThreadLocal<double[]> _spotBuffer;
    private readonly ThreadLocal<int>      _currentPathIndex = new();

    private readonly double _drift;      // (r − q − ½σ²)·dt
    private readonly double _diffusion;  // σ·√dt

    /// <summary>
    /// The index of the path currently being evaluated on the calling thread.
    /// Valid only during EvaluatePayoff; undefined between paths.
    /// </summary>
    protected int CurrentPathIndex => _currentPathIndex.Value;

    protected EQMCPricer(
        Option         option,
        DateOnly       valuationDate,
        EqMarketData   market,
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

        RiskFreeRate  = ZeroCurve.InterpolateRate(market.RiskFreeRate,  valuationDate, option.ExpiryYears);
        DividendYield = ZeroCurve.InterpolateRate(market.DividendYield, valuationDate, option.ExpiryYears);

        DiscountFactor = Math.Exp(-RiskFreeRate * option.ExpiryYears);

        _drift     = (RiskFreeRate - DividendYield - 0.5 * volatility * volatility) * Dt;
        _diffusion = volatility * Math.Sqrt(Dt);

        var steps = cube.Steps;
        _spotBuffer = new ThreadLocal<double[]>(() => new double[steps]);
    }

    /// <summary>
    /// Simulates a discretized GBM spot path into the calling thread's buffer.
    /// The returned span is valid only until the next SimulateSpotPath call on
    /// the same thread.
    /// </summary>
    protected ReadOnlySpan<double> SimulateSpotPath(int pathIndex)
    {
        var buf = _spotBuffer.Value!;
        var s   = Market.Spot;

        if (Volatility == 0.0)
        {
            var deterministicStep = Math.Exp((RiskFreeRate - DividendYield) * Dt);
            for (var t = 0; t < Cube.Steps; t++)
            {
                s      *= deterministicStep;
                buf[t]  = s;
            }
        }
        else
        {
            for (var t = 0; t < Cube.Steps; t++)
            {
                s      *= Math.Exp(_drift + _diffusion * Cube[pathIndex, t, 0]);
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
    /// spotPath[^1] = terminal spot S(T).
    /// Must not write to shared mutable state.
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
