using Aegis.Instruments;
using RandomSimulator;
using NodaTime;

namespace MCPricer.FX;

/// <summary>
/// Single-barrier (knock-in / knock-out) option on a basket of correlated FX
/// currency pairs. Combines FXBasketVanillaMCPricer's multi-leg correlated-GBM
/// simulation with FXBarrierOptionMCPricer's barrier-monitoring logic, applied
/// to the *aggregated basket level* path rather than to a single spot.
///
/// ── Model ─────────────────────────────────────────────────────────────────────
/// Each leg i follows correlated Garman-Kohlhagen GBM (see FXBasketVanillaMCPricer
/// for the full leg-setup convention — sorted leg names, per-step forward-rate
/// drift, single domestic discount factor):
///   dS_i/S_i = (r_d − r_f,i) dt + σ_i dW_i,    corr(dW_i, dW_j) = ρ_ij
///
/// At every monitoring date t_k the basket level is recomputed from the live leg
/// spots:
///   level(t_k) = Aggregate(method, [w_1·S_1(t_k), …, w_N·S_N(t_k)])
///
/// Barrier is monitored on the *level path* exactly as FXBarrierOptionMCPricer
/// monitors a single spot path — same four barrier types, same rebate convention.
///
/// ── Barrier observation modes ─────────────────────────────────────────────────
///
///   DISCRETE: level compared to barrier only at monitoring dates.
///
///   CONTINUOUS (Brownian-bridge correction with a local-volatility proxy):
///     Reuses the single-asset bridge formula with σ_eff(t_k):
///
///       WEIGHTED_SUM (level = Σ w_i·S_i):
///         σ_eff² = [ Σ_i Σ_j (w_i·S_i)(w_j·S_j)·σ_i·σ_j·ρ_ij ] / level²
///
///       BEST_OF/WORST_OF (level = max/min w_i·S_i):
///         σ_eff = σ_k,   k = argmax/argmin (w_i·S_i)
///
///     Both collapse exactly to the single-asset formula for N=1.
///
/// ── Greeks ────────────────────────────────────────────────────────────────────
/// Not supported. CreateBumped is left unoverridden (base throws NotSupportedException).
///
/// ── Edge cases ────────────────────────────────────────────────────────────────
///   Single-leg basket (N=1, weight=1): degenerates exactly to
///   FXBarrierOptionMCPricer for every aggregation method and observation mode.
///   Zero vol on all legs: level path is deterministic; barrier outcome is exact.
/// </summary>
public sealed class FXBasketBarrierOptionMCPricer : MCBasePricer
{
    private readonly double                  _strike;
    private readonly double                  _phi;            // +1 call, −1 put
    private readonly BasketAggregationMethod _aggregation;
    private readonly string[]                _legNames;       // sorted leg identifiers
    private readonly double[]                _weights;

    private readonly LocalDate _valuationDate;
    private readonly double   _expiryYears;

    private readonly double             _barrierLevel;
    private readonly BarrierType        _barrierType;
    private readonly BarrierObservation _observation;
    private readonly double             _rebate;
    private readonly double             _dt;

    // Set by PrepareFromMarket; valid during and after the most recent Price(markets, vols) call.
    private double[]  _initialSpots  = null!;
    private double[,] _drift         = null!;   // [leg, step]
    private double[]  _diffusions    = null!;   // σ_i·√dt
    private double[]  _legVols       = null!;   // flat per-leg vols σ_i (for σ_eff proxy)
    private double    _discountFactor;
    private double    _initialLevel;

    private readonly double[,] _correlation;    // leg correlation matrix ρ_ij (from cube; fixed)

    // Two scratch buffers per thread: running per-leg spots, and the weighted
    // values handed to BasketAggregator at each monitoring date.
    private readonly ThreadLocal<double[]> _spotScratch;
    private readonly ThreadLocal<double[]> _weightedScratch;

    /// <summary>Sorted basket leg identifiers (read-only view for inspection/tests).</summary>
    public IReadOnlyList<string> LegNames => _legNames;

    public FXBasketBarrierOptionMCPricer(
        Option         option,
        LocalDate       valuationDate,
        SimulationCube cube)
        : base(cube)
    {
        ArgumentNullException.ThrowIfNull(option);

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

        var legNames = basket.UnderlyingToWeight.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray();
        var n        = legNames.Length;

        if (cube.Assets != n)
            throw new ArgumentException(
                $"SimulationCube.Assets ({cube.Assets}) must equal the number of basket legs ({n}); " +
                "one correlated GBM factor per leg, asset index = sorted position of the leg identifier.",
                nameof(cube));

        foreach (var name in legNames)
        {
            if (!FXBasketVanillaMCPricer.TryParseCurrencyPair(name, out _))
                throw new ArgumentException(
                    $"Basket leg '{name}' is not a valid FX currency pair " +
                    "(expected a 6-letter code like 'EURUSD' or separated form 'EUR/USD').",
                    nameof(option));
        }

        _legNames      = legNames;
        _weights       = legNames.Select(name => basket.UnderlyingToWeight[name]).ToArray();
        _strike        = option.Strike;
        _phi           = option.OptionType == OptionType.Call ? 1.0 : -1.0;
        _aggregation   = basket.AggregationMethod;
        _valuationDate  = valuationDate;
        _expiryYears    = option.ExpiryYears;

        _barrierLevel = barrier.BarrierLevel;
        _barrierType  = barrier.BarrierType;
        _observation  = barrier.Observation;
        _rebate       = barrier.Rebate;
        _dt           = option.ExpiryYears / cube.Steps;

        _correlation = cube.Correlation;

        var legCount = n;
        _spotScratch     = new ThreadLocal<double[]>(() => new double[legCount]);
        _weightedScratch = new ThreadLocal<double[]>(() => new double[legCount]);
    }

    /// <summary>
    /// Prices the basket barrier option for the given market data and per-leg volatilities.
    /// Per-step drift is computed using ZeroCurve.ForwardRate for each leg.
    /// </summary>
    public PricingResult Price(
        IReadOnlyDictionary<string, FxMarketData> markets,
        IReadOnlyDictionary<string, double>       volatilities)
    {
        PrepareFromMarket(markets, volatilities);
        return base.Price();
    }

    /// <summary>
    /// Prices the basket barrier option asynchronously for the given market data and per-leg volatilities.
    /// </summary>
    public Task<PricingResult> PriceAsync(
        IReadOnlyDictionary<string, FxMarketData> markets,
        IReadOnlyDictionary<string, double>       volatilities,
        CancellationToken                         ct = default)
    {
        PrepareFromMarket(markets, volatilities);
        return base.PriceAsync(ct);
    }

    private void PrepareFromMarket(
        IReadOnlyDictionary<string, FxMarketData> markets,
        IReadOnlyDictionary<string, double>       volatilities)
    {
        ArgumentNullException.ThrowIfNull(markets);
        ArgumentNullException.ThrowIfNull(volatilities);

        var n    = _legNames.Length;
        var T    = _expiryYears;
        var dt   = _dt;

        var initialSpots = new double[n];
        var drift        = new double[n, Cube.Steps];
        var diffusions   = new double[n];
        var legVols      = new double[n];
        double? domesticRate = null;

        for (var i = 0; i < n; i++)
        {
            var name = _legNames[i];

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

            if (i == 0)
                domesticRate = ZeroCurve.InterpolateRate(market.DomesticRate, _valuationDate, T);

            for (var t = 0; t < Cube.Steps; t++)
            {
                var t0 = t * dt;
                var t1 = (t + 1) * dt;
                var fD = ZeroCurve.ForwardRate(market.DomesticRate, _valuationDate, t0, t1);
                var fF = ZeroCurve.ForwardRate(market.ForeignRate,  _valuationDate, t0, t1);
                drift[i, t] = (fD - fF - 0.5 * vol * vol) * dt;
            }

            initialSpots[i] = market.Spot;
            diffusions[i]   = vol * Math.Sqrt(dt);
            legVols[i]      = vol;
        }

        _initialSpots   = initialSpots;
        _drift          = drift;
        _diffusions     = diffusions;
        _legVols        = legVols;
        _discountFactor = Math.Exp(-domesticRate!.Value * T);

        // Compute initial basket level for continuous-barrier bridge seeding.
        var weightedInitial = new double[n];
        for (var i = 0; i < n; i++) weightedInitial[i] = _weights[i] * initialSpots[i];
        _initialLevel = BasketAggregator.Aggregate(_aggregation, weightedInitial);
    }

    protected override double SimulatePath(int pathIndex)
    {
        if (_drift is null)
            throw new InvalidOperationException(
                "Market data has not been supplied. Call Price(markets, volatilities) instead of Price().");

        var spot     = _spotScratch.Value!;
        var weighted = _weightedScratch.Value!;
        var n        = _legNames.Length;

        for (var i = 0; i < n; i++) spot[i] = _initialSpots[i];

        var pathSeed = pathIndex ^ unchecked((int)0x9E3779B9u);

        var barrierHit = false;
        var prevLevel  = _initialLevel;

        for (var t = 0; t < Cube.Steps; t++)
        {
            var sigmaEff = (!barrierHit && _observation == BarrierObservation.Continuous)
                ? EffectiveVolatility(spot)
                : 0.0;

            var z = Cube.StepIncrements(pathIndex, t);
            for (var i = 0; i < n; i++)
                spot[i] *= Math.Exp(_drift[i, t] + _diffusions[i] * z[i]);

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

        var u = new Random(pathSeed ^ (step * 1000003)).NextDouble();
        return u < pCross;
    }

    /// <summary>
    /// Probability that a GBM-like path crosses barrier H between two monitored
    /// endpoints a, b on the same side of H, given σ²·dt = sigSqDt.
    ///   P = exp(−2·log(H/a)·log(H/b) / (σ²·dt))
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
