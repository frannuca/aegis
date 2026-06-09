using Aegis.Instruments;
using RandomSimulator;

namespace MCPricer.FX;

/// <summary>
/// Single-barrier (knock-in / knock-out) option on a basket of correlated FX
/// currency pairs. Combines FXBasketVanillaMCPricer's multi-leg correlated-GBM
/// simulation with FXBarrierOptionMCPricer's barrier-monitoring logic, applied
/// to the *aggregated basket level* path rather than to a single spot.
///
/// ── Model ─────────────────────────────────────────────────────────────────────
/// Each leg i follows correlated Garman-Kohlhagen GBM (see FXBasketVanillaMCPricer
/// for the full leg-setup convention — sorted leg names, term-structure rates
/// interpolated once at expiry, single domestic discount factor):
///   dS_i/S_i = (r_d − r_f,i) dt + σ_i dW_i,    corr(dW_i, dW_j) = ρ_ij
///
/// At every monitoring date t_k the basket level is recomputed from the live leg
/// spots:
///   level(t_k) = Aggregate(method, [w_1·S_1(t_k), …, w_N·S_N(t_k)])     (BasketAggregator)
///
/// and the barrier is monitored on this *level path* level(t_0), …, level(t_n)
/// exactly as FXBarrierOptionMCPricer monitors a single spot path — the same
/// four barrier types (up/down × in/out), the same rebate-on-extinguishment
/// convention, and the same terminal vanilla-on-the-level payoff:
///
///   UP_AND_IN/DOWN_AND_IN    payoff = vanilla(level(T)) if barrier hit, else rebate
///   UP_AND_OUT/DOWN_AND_OUT  payoff = vanilla(level(T)) if never hit,   else rebate
///   Price = P(0,T) · E[payoff],   vanilla(L) = max(φ·(L − K), 0)
///
/// ── Barrier observation modes ─────────────────────────────────────────────────
///
///   DISCRETE:
///     The level is compared to the barrier only at the n monitoring dates —
///     an O(1) comparison per step, exactly like the single-asset case.
///
///   CONTINUOUS (Brownian-bridge correction with a local-volatility proxy):
///     For a single lognormal asset, the probability that a continuous path
///     crossed H between two same-side observations a, b has the closed form
///       P = exp(−2·log(H/a)·log(H/b) / (σ²·dt))     (Baldi-Caramellino-Iovino 1999)
///     — see FXBarrierOptionMCPricer. The basket level is *not* lognormal (a
///     sum, max, or min of correlated lognormals isn't itself lognormal), so no
///     exact closed-form bridge exists for it. This pricer reuses the identical
///     formula with a *local effective volatility* σ_eff(t_k) — the
///     instantaneous volatility of the level process over [t_k, t_k+dt],
///     evaluated from the live leg spots at t_k (recomputed at every monitoring
///     date, not frozen at inception):
///
///       WEIGHTED_SUM  (level = Σ w_i·S_i ⇒ dlevel = Σ w_i·dS_i to leading order):
///         σ_eff² = [ Σ_i Σ_j (w_i·S_i)·(w_j·S_j)·σ_i·σ_j·ρ_ij ] / level²
///         — the standard linearised ("Levy" / moment-matching) basket-volatility
///         proxy: Itô on level = Σ w_i S_i gives instantaneous variance
///         Var(dlevel)/level² = Σ_ij (w_i S_i)(w_j S_j) σ_i σ_j ρ_ij dt / level².
///
///       BEST_OF / WORST_OF  (level = max_i / min_i (w_i·S_i)):
///         σ_eff = σ_k,   k = argmax_i / argmin_i (w_i·S_i(t_k))
///         — almost surely, at any instant the running max/min of distinct
///         continuous diffusions coincides with exactly one leg (ties have
///         probability zero), so the level process locally *is* that leg's
///         GBM — constant instantaneous vol σ_k — to leading order.
///
///     Both branches collapse *exactly* to the single-asset formula for a
///     degenerate single-leg basket (N=1, w_1=1 ⇒ σ_eff ≡ σ_1, independent of
///     the aggregation method) — the strongest available check on the proxy,
///     exercised by the "single-leg basket ⇒ matches FXBarrierOptionMCPricer"
///     benchmark in the test suite.
///
///     For reproducibility, bridge draws are seeded deterministically from the
///     path index (xor with a salt) and step — exactly as FXBarrierOptionMCPricer.
///
/// ── Greeks ────────────────────────────────────────────────────────────────────
/// Not supported, for the same reason as FXBasketVanillaMCPricer: a basket
/// barrier has one delta/vega per leg plus barrier-proximity effects, which
/// doesn't fit MCBasePricer's single-spot/single-vol bump model. CreateBumped
/// is intentionally left unoverridden (base throws NotSupportedException).
///
/// ── Edge cases ────────────────────────────────────────────────────────────────
///   Single-leg basket (N=1, weight=1): degenerates exactly to
///   FXBarrierOptionMCPricer — for every aggregation method and observation mode.
///   Zero vol on all legs: every leg path — and hence the level path — is
///   deterministic, so the barrier-hit outcome and the price are exact.
/// </summary>
public sealed class FXBasketBarrierOptionMCPricer : MCBasePricer
{
    private readonly double                  _strike;
    private readonly double                  _phi;            // +1 call, −1 put
    private readonly BasketAggregationMethod _aggregation;
    private readonly double                  _discountFactor; // exp(-r_d·T)

    private readonly string[]  _legNames;     // sorted leg identifiers, e.g. ["EURUSD","GBPUSD"]
    private readonly double[]  _initialSpots;
    private readonly double[]  _weights;
    private readonly double[]  _drifts;       // (r_d - r_f,i - ½σ_i²)·dt
    private readonly double[]  _diffusions;   // σ_i·√dt
    private readonly double[]  _legVols;      // flat per-leg vols σ_i (for the σ_eff proxy)
    private readonly double[,] _correlation;  // leg correlation matrix ρ_ij (from the cube)

    private readonly double             _initialLevel;
    private readonly double             _barrierLevel;
    private readonly BarrierType        _barrierType;
    private readonly BarrierObservation _observation;
    private readonly double             _rebate;
    private readonly double             _dt;

    // Two scratch buffers per thread: running per-leg spots, and the weighted
    // values handed to BasketAggregator at each monitoring date.
    private readonly ThreadLocal<double[]> _spotScratch;
    private readonly ThreadLocal<double[]> _weightedScratch;

    /// <summary>Sorted basket leg identifiers (read-only view for inspection/tests).</summary>
    public IReadOnlyList<string> LegNames => _legNames;

    public FXBasketBarrierOptionMCPricer(
        Option                                     option,
        DateOnly                                   valuationDate,
        IReadOnlyDictionary<string, FxMarketData>  markets,
        IReadOnlyDictionary<string, double>        volatilities,
        SimulationCube                             cube)
        : base(cube)
    {
        ArgumentNullException.ThrowIfNull(option);
        ArgumentNullException.ThrowIfNull(markets);
        ArgumentNullException.ThrowIfNull(volatilities);

        if (option.UnderlyingKindCase != Option.UnderlyingKindOneofCase.Basket)
            throw new ArgumentException("Option.UnderlyingKind must be Basket.", nameof(option));
        if (option.KindCase != Option.KindOneofCase.Barrier)
            throw new ArgumentException("Option.Kind must be Barrier.", nameof(option));
        if (option.Strike <= 0)
            throw new ArgumentException("Strike must be positive.", nameof(option));
        if (option.ExpiryYears <= 0)
            throw new ArgumentException("ExpiryYears must be positive.", nameof(option));

        var basket = option.Basket;
        if (basket.UnderlyingToWeight.Count == 0)
            throw new ArgumentException("Basket.UnderlyingToWeight must be non-empty.", nameof(option));
        if (basket.AggregationMethod == BasketAggregationMethod.Unspecified)
            throw new ArgumentException("Basket.AggregationMethod must be specified.", nameof(option));

        var barrier = option.Barrier;
        if (barrier.BarrierLevel <= 0)
            throw new ArgumentException("BarrierLevel must be positive.", nameof(option));
        if (barrier.BarrierType == BarrierType.Unspecified)
            throw new ArgumentException("BarrierOption.BarrierType must be specified.", nameof(option));
        if (barrier.Observation == BarrierObservation.Unspecified)
            throw new ArgumentException("BarrierOption.Observation must be specified.", nameof(option));

        // Sort for a deterministic, reproducible mapping onto cube asset indices —
        // identical convention to FXBasketVanillaMCPricer.
        var legNames = basket.UnderlyingToWeight.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray();
        var n        = legNames.Length;

        if (cube.Assets != n)
            throw new ArgumentException(
                $"SimulationCube.Assets ({cube.Assets}) must equal the number of basket legs ({n}); " +
                "one correlated GBM factor per leg, asset index = sorted position of the leg identifier.",
                nameof(cube));

        var dt           = option.ExpiryYears / cube.Steps;
        var initialSpots = new double[n];
        var weights      = new double[n];
        var drifts       = new double[n];
        var diffusions   = new double[n];
        var legVols      = new double[n];
        double? domesticRate = null;

        for (var i = 0; i < n; i++)
        {
            var name = legNames[i];

            if (!FXBasketVanillaMCPricer.TryParseCurrencyPair(name, out _))
                throw new ArgumentException(
                    $"Basket leg '{name}' is not a valid FX currency pair " +
                    "(expected a 6-letter code like 'EURUSD' or separated form 'EUR/USD').",
                    nameof(option));

            if (!markets.TryGetValue(name, out var market) || market is null)
                throw new ArgumentException($"No market data supplied for basket leg '{name}'.", nameof(markets));
            if (!volatilities.TryGetValue(name, out var vol))
                throw new ArgumentException($"No volatility supplied for basket leg '{name}'.", nameof(volatilities));

            if (market.Spot <= 0)
                throw new ArgumentException($"Market spot for leg '{name}' must be positive.", nameof(markets));
            if (vol < 0)
                throw new ArgumentOutOfRangeException(nameof(volatilities), $"Volatility for leg '{name}' must be ≥ 0.");
            if (!string.Equals(market.CurrencyPair, name, StringComparison.Ordinal))
                throw new ArgumentException(
                    $"FxMarketData.CurrencyPair ('{market.CurrencyPair}') for leg '{name}' must equal the " +
                    "basket leg identifier — provide market data keyed and labelled consistently.",
                    nameof(markets));

            var legDomesticRate = ZeroCurve.InterpolateRate(market.DomesticRate, valuationDate, option.ExpiryYears);
            var legForeignRate  = ZeroCurve.InterpolateRate(market.ForeignRate,  valuationDate, option.ExpiryYears);

            // First leg (sorted order) sets the common settlement-currency discount rate.
            domesticRate ??= legDomesticRate;

            initialSpots[i] = market.Spot;
            weights[i]      = basket.UnderlyingToWeight[name];
            drifts[i]       = (legDomesticRate - legForeignRate - 0.5 * vol * vol) * dt;
            diffusions[i]   = vol * Math.Sqrt(dt);
            legVols[i]      = vol;
        }

        _legNames       = legNames;
        _initialSpots   = initialSpots;
        _weights        = weights;
        _drifts         = drifts;
        _diffusions     = diffusions;
        _legVols        = legVols;
        _correlation    = cube.Correlation;
        _strike         = option.Strike;
        _phi            = option.OptionType == OptionType.Call ? 1.0 : -1.0;
        _aggregation    = basket.AggregationMethod;
        _discountFactor = Math.Exp(-domesticRate!.Value * option.ExpiryYears);

        _barrierLevel = barrier.BarrierLevel;
        _barrierType  = barrier.BarrierType;
        _observation  = barrier.Observation;
        _rebate       = barrier.Rebate;
        _dt           = dt;

        var weightedInitial = new double[n];
        for (var i = 0; i < n; i++) weightedInitial[i] = weights[i] * initialSpots[i];
        _initialLevel = BasketAggregator.Aggregate(_aggregation, weightedInitial);

        var legCount = n;
        _spotScratch     = new ThreadLocal<double[]>(() => new double[legCount]);
        _weightedScratch = new ThreadLocal<double[]>(() => new double[legCount]);
    }

    protected override double SimulatePath(int pathIndex)
    {
        var spot     = _spotScratch.Value!;
        var weighted = _weightedScratch.Value!;
        var n        = _legNames.Length;

        for (var i = 0; i < n; i++) spot[i] = _initialSpots[i];

        // Bridge draws are seeded per (path, step) for reproducibility — same
        // salt/mixing convention as FXBarrierOptionMCPricer.
        var pathSeed = pathIndex ^ unchecked((int)0x9E3779B9u);

        var barrierHit = false;
        var prevLevel  = _initialLevel;

        for (var t = 0; t < Cube.Steps; t++)
        {
            // σ_eff must reflect the level process over [t, t+dt]; evaluate it
            // from the *pre-update* leg spots, before advancing to t+dt.
            var sigmaEff = (!barrierHit && _observation == BarrierObservation.Continuous)
                ? EffectiveVolatility(spot)
                : 0.0;

            var z = Cube.StepIncrements(pathIndex, t);
            for (var i = 0; i < n; i++)
                spot[i] *= Math.Exp(_drifts[i] + _diffusions[i] * z[i]);

            for (var i = 0; i < n; i++)
                weighted[i] = _weights[i] * spot[i];
            var currLevel = BasketAggregator.Aggregate(_aggregation, weighted.AsSpan(0, n));

            if (!barrierHit)
            {
                barrierHit = _observation == BarrierObservation.Discrete
                    ? IsBreached(currLevel)
                    : CheckContinuousStep(prevLevel, currLevel, sigmaEff, pathSeed, t);
            }

            prevLevel = currLevel;
        }

        var terminalLevel = prevLevel;
        var vanillaPayoff = Math.Max(_phi * (terminalLevel - _strike), 0.0);

        var payoff = _barrierType switch
        {
            BarrierType.UpAndIn  or BarrierType.DownAndIn
                => barrierHit ? vanillaPayoff : _rebate,
            BarrierType.UpAndOut or BarrierType.DownAndOut
                => barrierHit ? _rebate : vanillaPayoff,
            _ => throw new InvalidOperationException($"Unknown barrier type {_barrierType}.")
        };

        return _discountFactor * payoff;
    }

    // ── Continuous monitoring (Brownian bridge with local-volatility proxy) ──

    private bool CheckContinuousStep(double prevLevel, double currLevel, double sigmaEff, int pathSeed, int step)
    {
        if (IsBreached(currLevel))
            return true;

        if (IsBreached(prevLevel) || sigmaEff <= 0.0)
            return false;

        var sigSqDt = sigmaEff * sigmaEff * _dt;
        if (sigSqDt <= 0.0)
            return false;

        var pCross = BridgeCrossingProb(prevLevel, currLevel, _barrierLevel, sigSqDt);
        if (pCross <= 0.0)
            return false;

        // Deterministic per (path, step) — new Random is cheap for a single draw.
        var u = new Random(pathSeed ^ (step * 1000003)).NextDouble();
        return u < pCross;
    }

    /// <summary>
    /// Probability that a GBM-like path crosses barrier H between two monitored
    /// endpoints a, b on the same side of H, given local volatility σ such that
    /// σ²·dt = sigSqDt:
    ///   P = exp(−2·log(H/a)·log(H/b) / (σ²·dt))
    /// Returns 1.0 when the endpoints are on opposite sides (crossing certain).
    /// Identical formula to FXBarrierOptionMCPricer.BridgeCrossingProb.
    /// </summary>
    private static double BridgeCrossingProb(double a, double b, double h, double sigSqDt)
    {
        var logHa = Math.Log(h / a);
        var logHb = Math.Log(h / b);

        if (logHa * logHb <= 0.0)
            return 1.0;

        return Math.Exp(-2.0 * logHa * logHb / sigSqDt);
    }

    private bool IsBreached(double level) =>
        _barrierType is BarrierType.UpAndIn or BarrierType.UpAndOut
            ? level >= _barrierLevel
            : level <= _barrierLevel;

    // ── Local effective volatility of the basket level (continuous-mode proxy) ─

    /// <summary>
    /// Instantaneous volatility proxy σ_eff for the basket-level process over
    /// the next step, evaluated from the live (pre-step) leg spots — see the
    /// class XML doc for the derivation of both branches.
    /// </summary>
    private double EffectiveVolatility(ReadOnlySpan<double> spot) =>
        _aggregation switch
        {
            BasketAggregationMethod.WeightedSum => WeightedSumEffectiveVolatility(spot),
            BasketAggregationMethod.BestOf      => LeadingLegVolatility(spot, selectMax: true),
            BasketAggregationMethod.WorstOf     => LeadingLegVolatility(spot, selectMax: false),
            _ => throw new InvalidOperationException($"Unknown aggregation method {_aggregation}.")
        };

    /// <summary>
    /// Levy / moment-matching basket-volatility proxy:
    ///   σ_eff² = [ Σ_i Σ_j (w_i·S_i)(w_j·S_j)·σ_i·σ_j·ρ_ij ] / level²
    /// (instantaneous variance of d(level)/level under level = Σ w_i·S_i, via Itô).
    /// </summary>
    private double WeightedSumEffectiveVolatility(ReadOnlySpan<double> spot)
    {
        var n     = _legNames.Length;
        var level = 0.0;
        for (var i = 0; i < n; i++) level += _weights[i] * spot[i];
        if (level <= 0.0) return 0.0;

        var variance = 0.0;
        for (var i = 0; i < n; i++)
        {
            var ai = _weights[i] * spot[i] * _legVols[i];
            for (var j = 0; j < n; j++)
            {
                var aj = _weights[j] * spot[j] * _legVols[j];
                variance += ai * aj * _correlation[i, j];
            }
        }
        variance /= level * level;
        return Math.Sqrt(Math.Max(0.0, variance));
    }

    /// <summary>
    /// Locally, the running max/min of distinct continuous diffusions coincides
    /// with exactly one leg almost surely, so BEST_OF/WORST_OF's level process
    /// is — to leading order — that leg's own GBM, with its own constant vol.
    /// </summary>
    private double LeadingLegVolatility(ReadOnlySpan<double> spot, bool selectMax)
    {
        var n          = _legNames.Length;
        var leadingIdx = 0;
        var leadingVal = _weights[0] * spot[0];
        for (var i = 1; i < n; i++)
        {
            var val = _weights[i] * spot[i];
            if (selectMax ? val > leadingVal : val < leadingVal)
            {
                leadingVal = val;
                leadingIdx = i;
            }
        }
        return _legVols[leadingIdx];
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _spotScratch.Dispose();
            _weightedScratch.Dispose();
        }
        base.Dispose(disposing);
    }
}
