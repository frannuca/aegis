using Aegis.Instruments;
using NodaTime;
using NodaTime.Text;
using RandomSimulator;

namespace MCPricer;

/// <summary>
/// Abstract GBM option pricer shared by FX (Garman-Kohlhagen) and equity
/// (Black-Scholes) pricers. Replaces the structurally isomorphic FXMCPricer
/// and EQMCPricer with a single class parameterised by IMarketData.
///
/// ── Stochastic process ────────────────────────────────────────────────────────
/// Under the risk-neutral measure Q:
///   dS/S = (r_discount − r_carry) dt + σ dW
///
/// where the rate mapping depends on the concrete market data:
///   FxMarketData  r_discount = domestic rate   r_carry = foreign rate
///   EqMarketData  r_discount = risk-free rate  r_carry = dividend yield
///
/// ── Per-step forward-rate drift ───────────────────────────────────────────────
/// Drift is computed step-by-step from the full term structures using
/// ZeroCurve.ForwardRate, so a non-flat curve correctly varies the drift
/// across the simulation grid. For flat curves the computation degenerates
/// to the same scalar used before, preserving bit-for-bit results.
///
/// Exact log-Euler discretization (no time-step error for GBM):
///   S(t+dt) = S(t) · exp( (f_d(t,t+dt) − f_c(t,t+dt) − ½σ²)·dt + σ·√dt · Z_t )
///
/// ── Discrete cash dividends ───────────────────────────────────────────────────
/// Model: escrowed / jump model. After the multiplicative GBM step, the
/// cash dividend ex-dividing during [t, t+dt] is subtracted from the spot:
///   S(t+dt) = max(S(t)·exp(·) − D_t, 0)
///
/// Discrete dividends coexist with the continuous carry rate (additive).
/// Validated at prepare-time: Σ D_i (ex-date ≤ T) < S(0) — hard check,
/// no silent fallback.
///
/// ── Model assumptions ─────────────────────────────────────────────────────────
///   • Flat volatility σ over [0, T] (one σ per pricer instance).
///   • Continuously-compounded rates from Pillars zero-rate curves.
///   • Cash dividends are subtracted from the spot path on their ex-date step.
///   • Risk-neutral measure Q (domestic for FX, physical numeraire for equity).
///
/// ── Thread safety ────────────────────────────────────────────────────────────
/// SimulatePath is called concurrently by MCBasePricer.Price(). Thread safety
/// is achieved without locking via ThreadLocal scratch buffers.
/// Sequential repricing with different IMarketData is supported; concurrent
/// repricing on the same instance with different markets is NOT safe.
/// </summary>
public abstract class GbmOptionMCPricer : MCBasePricer
{
    protected readonly Option     OptionDef;
    protected readonly LocalDate  ValuationDate;
    protected readonly double     Volatility;

    // Set by PrepareFromMarket, valid during and after the most recent Price(market) call.
    protected double InitialSpot    { get; private set; }
    protected double DiscountFactor { get; private set; }
    protected double SigSqDt        { get; private set; }   // σ²·dt, for barrier bridge formula

    private double[]? _drift;           // per-step drift (f_d − f_c − ½σ²)·dt, length Steps
    private double    _diffusion;       // σ·√dt (uniform across steps)
    private double[]? _dividendAtStep;  // per-step cash dividend sum, or null if no dividends

    private readonly ThreadLocal<double[]> _spotBuffer;
    private readonly ThreadLocal<int>      _currentPathIndex = new();

    /// <summary>
    /// The index of the path currently being evaluated on the calling thread.
    /// Valid only during EvaluatePayoff; undefined between paths.
    /// </summary>
    protected int CurrentPathIndex => _currentPathIndex.Value;

    protected GbmOptionMCPricer(
        Option         option,
        LocalDate      valuationDate,
        double         volatility,
        SimulationCube cube)
        : base(cube)
    {
        ArgumentNullException.ThrowIfNull(option);
        if (volatility < 0)
            throw new ArgumentOutOfRangeException(nameof(volatility), "Volatility must be ≥ 0.");
        if (option.ExpiryYears <= 0)
            throw new ArgumentException("ExpiryYears must be positive.", nameof(option));

        OptionDef     = option;
        ValuationDate = valuationDate;
        Volatility    = volatility;

        var steps = cube.Steps;
        _spotBuffer = new ThreadLocal<double[]>(() => new double[steps]);
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

    // ── Market-aware Greeks ────────────────────────────────────────────────────

    /// <summary>
    /// Computes first- and second-order price sensitivities via bump-and-reprice,
    /// using the supplied market data. Same-cube repricing (common random numbers)
    /// reduces finite-difference variance.
    ///
    /// Spot / rate bumps reprice on this instance with bumped market data.
    /// Vol bumps create a new pricer instance with a different Volatility and reprice.
    /// Theta = NaN (bumping T requires a different SimulationCube step count).
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
        var vUp = CreateBumped(BumpType.VolUp,   volEps ).Price(market).Price;
        var vDn = CreateBumped(BumpType.VolDown, volEps ).Price(market).Price;
        var rUp = Price(MarketDataBumps.BumpDiscountCurve(market, +rateEps)).Price;
        var rDn = Price(MarketDataBumps.BumpDiscountCurve(market, -rateEps)).Price;

        return new GreekResult(
            Delta: (sUp - sDn)             / (2.0 * spotEps),
            Gamma: (sUp - 2.0 * mid + sDn) / (spotEps * spotEps),
            Vega:  (vUp - vDn)             / (2.0 * volEps),
            Theta: double.NaN,
            Rho:   (rUp - rDn)             / (2.0 * rateEps));
    }

    /// <summary>
    /// Sensitivity to the carry curve (FX: foreign rate; equity: dividend yield),
    /// computed by central finite differences.
    /// </summary>
    public double ComputeCarryRho(IMarketData market, double eps = 0.0001)
    {
        var up = Price(MarketDataBumps.BumpCarryCurve(market, +eps)).Price;
        var dn = Price(MarketDataBumps.BumpCarryCurve(market, -eps)).Price;
        return (up - dn) / (2.0 * eps);
    }

    // ── Internal: prepare per-step arrays from market data ────────────────────

    private void PrepareFromMarket(IMarketData market)
    {
        if (market.Spot <= 0)
            throw new ArgumentException("Spot must be positive.", nameof(market));

        InitialSpot = market.Spot;
        var T  = OptionDef.ExpiryYears;
        var dt = T / Cube.Steps;

        DiscountFactor = ZeroCurve.DiscountFactor(market.DiscountCurve, ValuationDate, T);
        _diffusion     = Volatility * Math.Sqrt(dt);
        SigSqDt        = Volatility * Volatility * dt;

        var steps = Cube.Steps;
        if (_drift is null || _drift.Length != steps)
            _drift = new double[steps];

        for (var i = 0; i < steps; i++)
        {
            var t0    = i * dt;
            var t1    = (i + 1) * dt;
            var fDisc  = ZeroCurve.ForwardRate(market.DiscountCurve, ValuationDate, t0, t1);
            var fCarry = ZeroCurve.ForwardRate(market.CarryCurve,    ValuationDate, t0, t1);
            _drift[i] = (fDisc - fCarry - 0.5 * Volatility * Volatility) * dt;
        }

        _dividendAtStep = PrepareDividends(market.Dividends, dt, T, market.Spot);
    }

    private double[]? PrepareDividends(
        IReadOnlyList<CashDividend> dividends, double dt, double T, double spot)
    {
        if (dividends.Count == 0)
            return null;

        var steps   = Cube.Steps;
        var perStep = new double[steps];
        var total   = 0.0;

        foreach (var div in dividends)
        {
            var exDate = LocalDatePattern.Iso.Parse(div.ExDate).Value;
            var tau    = Period.Between(ValuationDate, exDate, PeriodUnits.Days).Days / 365.0;
            if (tau <= 0 || tau > T)
                continue;

            // Bucket rule: t_{i} < tau ≤ t_{i+1}  →  step index i
            var step = (int)Math.Ceiling(tau / dt) - 1;
            step = Math.Clamp(step, 0, steps - 1);
            perStep[step] += div.Amount;
            total         += div.Amount;
        }

        if (total >= spot)
            throw new ArgumentException(
                $"Sum of discrete dividends with ex-date ≤ T ({total:G}) must be less than the current spot ({spot:G}).",
                nameof(dividends));

        return perStep;
    }

    // ── Path simulation ────────────────────────────────────────────────────────

    protected ReadOnlySpan<double> SimulateSpotPath(int pathIndex)
    {
        var buf = _spotBuffer.Value!;
        var s   = InitialSpot;
        var n   = Cube.Steps;

        for (var t = 0; t < n; t++)
        {
            s *= Math.Exp(_drift![t] + _diffusion * Cube[pathIndex, t, 0]);
            if (_dividendAtStep is not null)
                s = Math.Max(0.0, s - _dividendAtStep[t]);
            buf[t] = s;
        }

        return buf.AsSpan(0, n);
    }

    protected sealed override double SimulatePath(int pathIndex)
    {
        if (_drift is null)
            throw new InvalidOperationException(
                "Market data has not been supplied. Call Price(IMarketData market) instead of Price().");
        _currentPathIndex.Value = pathIndex;
        return DiscountFactor * EvaluatePayoff(SimulateSpotPath(pathIndex));
    }

    /// <summary>
    /// Evaluates the undiscounted payoff given the realized spot path.
    ///
    /// spotPath[t] = S at end of step t (1-indexed); spotPath[^1] = terminal spot S(T).
    ///
    /// Called from SimulatePath on whichever thread is processing this path.
    /// Must not write to shared mutable state. Access CurrentPathIndex (thread-local)
    /// for the current path index when needed (e.g. for deterministic RNG seeding).
    /// </summary>
    protected abstract double EvaluatePayoff(ReadOnlySpan<double> spotPath);

    // ── Disposal ──────────────────────────────────────────────────────────────

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
