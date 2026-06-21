using Aegis.Instruments;
using RandomSimulator;
using NodaTime;

namespace MCPricer.FX;

/// <summary>
/// European vanilla option on a basket of FX currency pairs, priced under the
/// domestic risk-neutral measure with one correlated GBM per leg.
///
/// ── Model ─────────────────────────────────────────────────────────────────────
/// Each leg i follows Garman-Kohlhagen GBM (see GbmOptionMCPricer):
///   dS_i/S_i = (r_d − r_f,i) dt + σ_i dW_i,    corr(dW_i, dW_j) = ρ_ij
///
/// All legs must share the same domestic (settlement) currency, so a single
/// discount factor P(0,T) = exp(−r_d·T) applies to the whole basket — taken
/// from the first leg's domestic curve (sorted order). Per-step drift for each
/// leg uses ZeroCurve.ForwardRate on that leg's domestic/foreign curves.
///
/// Correlation between legs lives in the SimulationCube (Cholesky of the leg
/// correlation matrix); this pricer reads correlated increments per asset index.
///
/// ── Basket level and payoff ───────────────────────────────────────────────────
///   level = Aggregate(method, [w_1·S_1(T), …, w_N·S_N(T)])      (BasketAggregator)
///   Price = P(0,T) · E[max(φ·(level − K), 0)]
///
/// where φ = +1 (call) / −1 (put), and Aggregate implements:
///   WEIGHTED_SUM → Σ w_i·S_i(T)            (portfolio level)
///   BEST_OF      → max_i (w_i·S_i(T))      (best-performing weighted leg)
///   WORST_OF     → min_i (w_i·S_i(T))      (worst-performing weighted leg)
///
/// ── Leg identification — "currency pair object" check ────────────────────────
/// Basket.UnderlyingToWeight keys are the single source of truth for basket
/// composition. For an FX basket, every key MUST parse as a currency pair —
/// either a 6-letter code ("EURUSD") or separator form ("EUR/USD", "EUR-USD") —
/// yielding a well-formed CurrencyPair. Keys that don't parse are rejected.
///
/// ── Greeks ────────────────────────────────────────────────────────────────────
/// Not supported: a basket has one delta/vega per leg, which doesn't fit the
/// single-spot/single-vol bump model in MCBasePricer. CreateBumped is
/// intentionally left unoverridden (base throws NotSupportedException).
///
/// ── Edge cases ────────────────────────────────────────────────────────────────
///   Single-leg basket (N=1, weight=1): degenerates exactly to
///   FXVanillaOptionMCPricer pricing (all three aggregation methods reduce to
///   the single value). Useful as an exact analytical check (Garman-Kohlhagen).
///   Zero vol on all legs: deterministic terminal spots → exact intrinsic value.
/// </summary>
public sealed class FXBasketVanillaMCPricer : MCBasePricer
{
    private readonly double                  _strike;
    private readonly double                  _phi;            // +1 call, −1 put
    private readonly BasketAggregationMethod _aggregation;
    private readonly string[]                _legNames;       // sorted leg identifiers
    private readonly double[]                _weights;

    private readonly LocalDate  _valuationDate;
    private readonly double    _expiryYears;

    // Set by PrepareFromMarket; valid during and after the most recent Price(markets, vols) call.
    private double[]  _initialSpots  = null!;
    private double[,] _drift         = null!;   // [leg, step]
    private double[]  _diffusions    = null!;   // σ_i·√dt
    private double    _discountFactor;

    // One scratch buffer per thread: holds terminal spots, then overwritten
    // in-place with weighted values w_i·S_i(T) before aggregation.
    private readonly ThreadLocal<double[]> _scratch;

    /// <summary>Sorted basket leg identifiers (read-only view for inspection/tests).</summary>
    public IReadOnlyList<string> LegNames => _legNames;

    public FXBasketVanillaMCPricer(
        Option         option,
        LocalDate       valuationDate,
        SimulationCube cube)
        : base(cube)
    {
        ArgumentNullException.ThrowIfNull(option);

        if (option.UnderlyingKindCase != Option.UnderlyingKindOneofCase.Basket)
            throw new ArgumentException("Option.UnderlyingKind must be Basket.", nameof(option));
        if (option.KindCase != Option.KindOneofCase.Vanilla)
            throw new ArgumentException("Option.Kind must be Vanilla.", nameof(option));
        if (option.Strike <= 0)
            throw new ArgumentException("Strike must be positive.", nameof(option));
        if (option.ExpiryYears <= 0)
            throw new ArgumentException("ExpiryYears must be positive.", nameof(option));

        var basket = option.Basket;
        if (basket.UnderlyingToWeight.Count == 0)
            throw new ArgumentException("Basket.UnderlyingToWeight must be non-empty.", nameof(option));
        if (basket.AggregationMethod == BasketAggregationMethod.Unspecified)
            throw new ArgumentException("Basket.AggregationMethod must be specified.", nameof(option));

        var legNames = basket.UnderlyingToWeight.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray();
        var n        = legNames.Length;

        if (cube.Assets != n)
            throw new ArgumentException(
                $"SimulationCube.Assets ({cube.Assets}) must equal the number of basket legs ({n}); " +
                "one correlated GBM factor per leg, asset index = sorted position of the leg identifier.",
                nameof(cube));

        foreach (var name in legNames)
        {
            if (!TryParseCurrencyPair(name, out _))
                throw new ArgumentException(
                    $"Basket leg '{name}' is not a valid FX currency pair " +
                    "(expected a 6-letter code like 'EURUSD' or separated form 'EUR/USD').",
                    nameof(option));
        }

        _legNames     = legNames;
        _weights      = legNames.Select(name => basket.UnderlyingToWeight[name]).ToArray();
        _strike       = option.Strike;
        _phi          = option.OptionType == OptionType.Call ? 1.0 : -1.0;
        _aggregation  = basket.AggregationMethod;
        _valuationDate = valuationDate;
        _expiryYears   = option.ExpiryYears;

        var legCount = n;
        _scratch = new ThreadLocal<double[]>(() => new double[legCount]);
    }

    /// <summary>
    /// Prices the basket option for the given market data and per-leg volatilities.
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
    /// Prices the basket option asynchronously for the given market data and per-leg volatilities.
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
        var dt   = T / Cube.Steps;

        var initialSpots = new double[n];
        var drift        = new double[n, Cube.Steps];
        var diffusions   = new double[n];
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

            // First leg (sorted order) sets the settlement-currency discount rate.
            if (i == 0)
                domesticRate = ZeroCurve.InterpolateRate(market.DomesticRate, _valuationDate, T);

            // Per-step drift for leg i: (fDisc_i[t] - fCarry_i[t] - 0.5·σ²)·dt
            for (var t = 0; t < Cube.Steps; t++)
            {
                var t0   = t * dt;
                var t1   = (t + 1) * dt;
                var fD   = ZeroCurve.ForwardRate(market.DomesticRate, _valuationDate, t0, t1);
                var fF   = ZeroCurve.ForwardRate(market.ForeignRate,  _valuationDate, t0, t1);
                drift[i, t] = (fD - fF - 0.5 * vol * vol) * dt;
            }

            initialSpots[i] = market.Spot;
            diffusions[i]   = vol * Math.Sqrt(dt);
        }

        _initialSpots  = initialSpots;
        _drift         = drift;
        _diffusions    = diffusions;
        _discountFactor = Math.Exp(-domesticRate!.Value * T);
    }

    protected override double SimulatePath(int pathIndex)
    {
        if (_drift is null)
            throw new InvalidOperationException(
                "Market data has not been supplied. Call Price(markets, volatilities) instead of Price().");

        var buf = _scratch.Value!;
        var n   = _legNames.Length;

        for (var i = 0; i < n; i++) buf[i] = _initialSpots[i];

        for (var t = 0; t < Cube.Steps; t++)
        {
            var z = Cube.StepIncrements(pathIndex, t);
            for (var i = 0; i < n; i++)
                buf[i] *= Math.Exp(_drift[i, t] + _diffusions[i] * z[i]);
        }

        // Overwrite terminal spots in-place with weighted values w_i·S_i(T).
        for (var i = 0; i < n; i++) buf[i] *= _weights[i];

        var level = BasketAggregator.Aggregate(_aggregation, buf.AsSpan(0, n));
        return _discountFactor * Math.Max(_phi * (level - _strike), 0.0);
    }

    /// <summary>
    /// Parses an FX leg identifier into a CurrencyPair (base + quote, 3 letters
    /// each). Accepts a bare 6-letter code ("EURUSD") or separated forms
    /// ("EUR/USD", "EUR-USD"); case-insensitive. Returns false for anything else
    /// (e.g. equity tickers, malformed codes, identical base/quote).
    /// </summary>
    internal static bool TryParseCurrencyPair(string identifier, out CurrencyPair pair)
    {
        pair = null!;
        if (string.IsNullOrWhiteSpace(identifier))
            return false;

        var s   = identifier.Trim().ToUpperInvariant();
        var sep = s.IndexOfAny(['/', '-']);

        string baseCcy, quoteCcy;
        if (sep >= 0)
        {
            baseCcy  = s[..sep];
            quoteCcy = s[(sep + 1)..];
        }
        else if (s.Length == 6)
        {
            baseCcy  = s[..3];
            quoteCcy = s[3..];
        }
        else
        {
            return false;
        }

        if (baseCcy.Length != 3 || quoteCcy.Length != 3)
            return false;
        if (!baseCcy.All(char.IsAsciiLetter) || !quoteCcy.All(char.IsAsciiLetter))
            return false;
        if (baseCcy == quoteCcy)
            return false;

        pair = new CurrencyPair { BaseCurrency = baseCcy, QuoteCurrency = quoteCcy };
        return true;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) _scratch.Dispose();
        base.Dispose(disposing);
    }
}
