using Aegis.Instruments;
using AnalyticalPricers;
using MCPricer.FX;
using RandomSimulator;
using NodaTime;

namespace MCPricer.Tests;

/// <summary>
/// Validates FXBasketVanillaMCPricer — a European vanilla option on a basket of
/// correlated FX currency pairs (WEIGHTED_SUM / BEST_OF / WORST_OF aggregation).
///
/// ── Benchmark strategy ───────────────────────────────────────────────────────
///
///   1. Single-leg basket (N=1, weight=1): every aggregation method degenerates
///      to the bare leg value, so the basket price must equal the analytical
///      Garman-Kohlhagen price exactly (5σ MC bound). This is the strongest
///      check — it validates the multi-asset machinery against a known formula.
///
///   2. Zero-volatility legs: terminal spots are deterministic
///      (S_i(T) = S_i(0)·exp((r_d − r_f,i)·T)), so the basket level and the
///      price are exact closed-form numbers — no MC noise at all.
///
///   3. Cross-method ordering: writing a_i = w_i·S_i(T) ≥ 0, pointwise on every path
///        min_i(a_i) = WORST_OF  ≤  max_i(a_i) = BEST_OF  ≤  Σ_i a_i = WEIGHTED_SUM
///      (a sum of non-negative terms is never smaller than its largest term).
///      max(level−K, 0) is non-decreasing in level, so the ordering survives the
///      payoff and the expectation — i.e. WORST_OF ≤ BEST_OF ≤ WEIGHTED_SUM holds
///      at the price level too (using one shared cube to remove MC noise).
///
///   4. Put-call parity for WEIGHTED_SUM: Call − Put = P(0,T)·(E[level] − K).
///      For WEIGHTED_SUM, E[level] = Σ w_i · F_i(0,T) is known in closed form
///      (each leg's forward), so parity has an exact analytical reference.
/// </summary>
public sealed class FXBasketVanillaMCPricerTests
{
    private const double T  = 1.0;
    private const int    Paths = 200_000;
    private const int    Steps = 50;
    private const int    Seed  = 42;

    // Pricers now take rates as Pillars zero-rate curves (term structures), interpolated
    // at the option's expiry from this valuation date — see ZeroCurve.InterpolateRate.
    // Tests use single-pillar "flat" curves so the interpolated rate equals the scalar
    // reference rate (rd, rf) regardless of T, keeping the analytical comparisons exact.
    private static readonly LocalDate ValuationDate = new(2026, 1, 1);
    private static readonly LocalDate CurveMaturity = new(2036, 1, 1);

    // ── Single-leg basket ⇒ exact Garman-Kohlhagen ───────────────────────────

    [Theory]
    [InlineData(BasketAggregationMethod.WeightedSum)]
    [InlineData(BasketAggregationMethod.BestOf)]
    [InlineData(BasketAggregationMethod.WorstOf)]
    public void SingleLeg_AnyAggregation_MatchesGarmanKohlhagen(BasketAggregationMethod method)
    {
        // With exactly one leg and weight 1.0, w·S(T) is the only candidate for
        // every aggregation rule — they all collapse to the same scalar. The
        // basket price must therefore equal the plain GK vanilla price exactly.
        const double spot = 1.10, rd = 0.05, rf = 0.02, vol = 0.20, strike = 1.10;

        var option = MakeBasketOption(
            legs: [("EURUSD", 1.0)], method, strike, isCall: true);
        var markets = new Dictionary<string, FxMarketData>
        {
            ["EURUSD"] = MakeMarket("EURUSD", spot, rd, rf)
        };
        var vols = new Dictionary<string, double> { ["EURUSD"] = vol };

        var cube = SimulationCube.GenerateIndependent(Paths, Steps, assets: 1, Seed);
        using var pricer = new FXBasketVanillaMCPricer(option, ValuationDate, cube);
        var mc = pricer.Price(markets, vols);

        var gk = GarmanKohlhagen.Price(spot, strike, T, vol, rd, rf, isCall: true);

        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    [Fact]
    public void SingleLeg_Put_MatchesGarmanKohlhagen()
    {
        const double spot = 1.10, rd = 0.05, rf = 0.02, vol = 0.20, strike = 1.15;

        var option = MakeBasketOption(
            legs: [("EURUSD", 1.0)], BasketAggregationMethod.WeightedSum, strike, isCall: false);
        var markets = new Dictionary<string, FxMarketData> { ["EURUSD"] = MakeMarket("EURUSD", spot, rd, rf) };
        var vols    = new Dictionary<string, double> { ["EURUSD"] = vol };

        var cube = SimulationCube.GenerateIndependent(Paths, Steps, assets: 1, Seed);
        using var pricer = new FXBasketVanillaMCPricer(option, ValuationDate, cube);
        var mc = pricer.Price(markets, vols);
        var gk = GarmanKohlhagen.Price(spot, strike, T, vol, rd, rf, isCall: false);

        AssertWithinMcBounds(mc, gk, sigma: 5.0);
    }

    // ── Zero volatility ⇒ exact deterministic price ──────────────────────────

    [Fact]
    public void ZeroVol_TwoLegs_WeightedSum_PricesAtExactDiscountedIntrinsic()
    {
        // All terminal spots deterministic: S_i(T) = S_i(0)·exp((r_d − r_f,i)·T).
        // Basket level = Σ w_i·S_i(T); price = exp(-r_d·T)·max(φ(level-K),0) — exact.
        var legs = BuildTwoZeroVolLegs(out var spot1T, out var spot2T, out var rd);
        const double w1 = 0.6, w2 = 0.4, strike = 1.18;

        var option = MakeBasketOption(
            legs: [("EURUSD", w1), ("GBPUSD", w2)], BasketAggregationMethod.WeightedSum, strike, isCall: true);

        var cube = SimulationCube.GenerateIndependent(Paths, Steps, assets: 2, Seed);
        using var pricer = new FXBasketVanillaMCPricer(option, ValuationDate, cube);
        var mc = pricer.Price(legs.markets, legs.vols);

        var level    = w1 * spot1T + w2 * spot2T;
        var expected = Math.Exp(-rd * T) * Math.Max(level - strike, 0.0);

        Assert.Equal(expected, mc.Price, precision: 10);
        Assert.Equal(0.0, mc.StandardError, precision: 10);
    }

    [Fact]
    public void ZeroVol_TwoLegs_BestOf_PricesAtExactMax()
    {
        var legs = BuildTwoZeroVolLegs(out var spot1T, out var spot2T, out var rd);
        const double w1 = 0.6, w2 = 0.4, strike = 0.70;

        var option = MakeBasketOption(
            legs: [("EURUSD", w1), ("GBPUSD", w2)], BasketAggregationMethod.BestOf, strike, isCall: true);

        var cube = SimulationCube.GenerateIndependent(Paths, Steps, assets: 2, Seed);
        using var pricer = new FXBasketVanillaMCPricer(option, ValuationDate, cube);
        var mc = pricer.Price(legs.markets, legs.vols);

        var level    = Math.Max(w1 * spot1T, w2 * spot2T);
        var expected = Math.Exp(-rd * T) * Math.Max(level - strike, 0.0);

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_TwoLegs_WorstOf_PricesAtExactMin()
    {
        var legs = BuildTwoZeroVolLegs(out var spot1T, out var spot2T, out var rd);
        const double w1 = 0.6, w2 = 0.4, strike = 0.40;

        var option = MakeBasketOption(
            legs: [("EURUSD", w1), ("GBPUSD", w2)], BasketAggregationMethod.WorstOf, strike, isCall: true);

        var cube = SimulationCube.GenerateIndependent(Paths, Steps, assets: 2, Seed);
        using var pricer = new FXBasketVanillaMCPricer(option, ValuationDate, cube);
        var mc = pricer.Price(legs.markets, legs.vols);

        var level    = Math.Min(w1 * spot1T, w2 * spot2T);
        var expected = Math.Exp(-rd * T) * Math.Max(level - strike, 0.0);

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_DeepOtmCall_PricesAtZero()
    {
        var legs = BuildTwoZeroVolLegs(out _, out _, out _);
        var option = MakeBasketOption(
            legs: [("EURUSD", 0.5), ("GBPUSD", 0.5)], BasketAggregationMethod.WeightedSum,
            strike: 100.0, isCall: true);

        var cube = SimulationCube.GenerateIndependent(Paths, Steps, assets: 2, Seed);
        using var pricer = new FXBasketVanillaMCPricer(option, ValuationDate, cube);
        var mc = pricer.Price(legs.markets, legs.vols);

        Assert.Equal(0.0, mc.Price, precision: 10);
    }

    // ── Cross-method ordering (equal weights, same cube ⇒ pathwise sandwich) ──

    [Fact]
    public void EqualWeights_WorstOf_LessOrEqual_BestOf_LessOrEqual_WeightedSum()
    {
        // Pathwise, with a = w·S1(T) ≥ 0 and b = w·S2(T) ≥ 0:
        //   min(a,b) = WorstOf  ≤  max(a,b) = BestOf  ≤  a+b = WeightedSum
        // The right inequality holds because a sum of non-negative terms is never
        // smaller than its largest term (a+b ≥ max(a,b) whenever a,b ≥ 0).
        // max(level−K,0) is non-decreasing in level, so the ordering survives the
        // payoff and the expectation — i.e. it holds at the price level too.
        // Using the same cube for all three removes MC noise from the comparison
        // (paired paths / common random numbers).
        const double strike = 1.0;
        var legs = MakeTwoLegMarkets(
            spot1: 1.10, rd1: 0.05, rf1: 0.02, vol1: 0.20,
            spot2: 1.25, rd2: 0.05, rf2: 0.01, vol2: 0.25,
            name1: "EURUSD", name2: "GBPUSD");

        var cube = SimulationCube.Generate(Paths, Steps, assets: 2,
            correlation: new double[,] { { 1.0, 0.3 }, { 0.3, 1.0 } }, Seed);

        var worstOf = new FXBasketVanillaMCPricer(
            MakeBasketOption([("EURUSD", 0.5), ("GBPUSD", 0.5)], BasketAggregationMethod.WorstOf, strike, true),
            ValuationDate, cube).Price(legs.markets, legs.vols);

        var bestOf = new FXBasketVanillaMCPricer(
            MakeBasketOption([("EURUSD", 0.5), ("GBPUSD", 0.5)], BasketAggregationMethod.BestOf, strike, true),
            ValuationDate, cube).Price(legs.markets, legs.vols);

        var weightedSum = new FXBasketVanillaMCPricer(
            MakeBasketOption([("EURUSD", 0.5), ("GBPUSD", 0.5)], BasketAggregationMethod.WeightedSum, strike, true),
            ValuationDate, cube).Price(legs.markets, legs.vols);

        Assert.True(worstOf.Price <= bestOf.Price + 1e-9,
            $"WorstOf={worstOf.Price:F6} should be ≤ BestOf={bestOf.Price:F6}");
        Assert.True(bestOf.Price <= weightedSum.Price + 1e-9,
            $"BestOf={bestOf.Price:F6} should be ≤ WeightedSum={weightedSum.Price:F6}");
    }

    // ── Put-call parity for WEIGHTED_SUM ──────────────────────────────────────

    [Fact]
    public void WeightedSum_PutCallParity_MatchesAnalyticalForwardLevel()
    {
        // For WEIGHTED_SUM, level(T) = Σ w_i·S_i(T) is linear, so
        //   E[level(T)] = Σ w_i·F_i(0,T),   F_i(0,T) = S_i(0)·exp((r_d − r_f,i)·T)
        // Call − Put = P(0,T)·(E[level(T)] − K)   — exact, model-free identity.
        const double strike = 1.0;
        var legs = MakeTwoLegMarkets(
            spot1: 1.10, rd1: 0.05, rf1: 0.02, vol1: 0.20,
            spot2: 1.25, rd2: 0.05, rf2: 0.01, vol2: 0.25,
            name1: "EURUSD", name2: "GBPUSD");
        const double w1 = 0.5, w2 = 0.5;

        var cube = SimulationCube.Generate(Paths, Steps, assets: 2,
            correlation: new double[,] { { 1.0, -0.4 }, { -0.4, 1.0 } }, Seed);

        var call = new FXBasketVanillaMCPricer(
            MakeBasketOption([("EURUSD", w1), ("GBPUSD", w2)], BasketAggregationMethod.WeightedSum, strike, true),
            ValuationDate, cube).Price(legs.markets, legs.vols);
        var put = new FXBasketVanillaMCPricer(
            MakeBasketOption([("EURUSD", w1), ("GBPUSD", w2)], BasketAggregationMethod.WeightedSum, strike, false),
            ValuationDate, cube).Price(legs.markets, legs.vols);

        var f1 = 1.10 * Math.Exp((0.05 - 0.02) * T);
        var f2 = 1.25 * Math.Exp((0.05 - 0.01) * T);
        var expectedLevel    = w1 * f1 + w2 * f2;
        var discount         = Math.Exp(-0.05 * T);
        var analyticalParity = discount * (expectedLevel - strike);

        var mcParity = call.Price - put.Price;
        var seSum    = call.StandardError + put.StandardError;

        Assert.True(Math.Abs(mcParity - analyticalParity) < 5.0 * seSum,
            $"Parity diff={Math.Abs(mcParity - analyticalParity):F6}  5σ={5.0 * seSum:F6}");
    }

    // ── Correlation effects (qualitative, well above MC noise) ───────────────

    [Fact]
    public void WeightedSum_HigherCorrelation_IncreasesAtmCallPrice()
    {
        // For a basket call, higher pairwise correlation increases the variance
        // of the basket level (less diversification benefit) ⇒ higher option
        // value (Jensen / convexity). Compare low vs. high positive correlation
        // with identical legs and equal weights.
        var legs = MakeTwoLegMarkets(
            spot1: 1.10, rd1: 0.05, rf1: 0.02, vol1: 0.20,
            spot2: 1.10, rd2: 0.05, rf2: 0.02, vol2: 0.20,
            name1: "EURUSD", name2: "GBPUSD");

        var f  = 1.10 * Math.Exp(0.03 * T);
        var atmStrike = f; // ATM forward on the (identical) legs ⇒ ATM on the basket too

        var lowCorrCube  = SimulationCube.Generate(Paths, Steps, 2, new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }, Seed);
        var highCorrCube = SimulationCube.Generate(Paths, Steps, 2, new double[,] { { 1.0, 0.9 }, { 0.9, 1.0 } }, Seed);

        var lowCorr = new FXBasketVanillaMCPricer(
            MakeBasketOption([("EURUSD", 0.5), ("GBPUSD", 0.5)], BasketAggregationMethod.WeightedSum, atmStrike, true),
            ValuationDate, lowCorrCube).Price(legs.markets, legs.vols);
        var highCorr = new FXBasketVanillaMCPricer(
            MakeBasketOption([("EURUSD", 0.5), ("GBPUSD", 0.5)], BasketAggregationMethod.WeightedSum, atmStrike, true),
            ValuationDate, highCorrCube).Price(legs.markets, legs.vols);

        Assert.True(highCorr.Price > lowCorr.Price,
            $"High-correlation basket call={highCorr.Price:F6} should exceed low-correlation={lowCorr.Price:F6}");
    }

    // ── Input validation ──────────────────────────────────────────────────────

    [Fact]
    public void NonBasketOption_Throws()
    {
        var option = new Option
        {
            Underlying  = "EURUSD",
            Strike      = 1.10,
            ExpiryYears = T,
            OptionType  = OptionType.Call,
            Vanilla     = new VanillaOption()
        };
        var cube = SimulationCube.GenerateIndependent(1_000, Steps, 1, Seed);
        Assert.Throws<ArgumentException>(() =>
            new FXBasketVanillaMCPricer(option, ValuationDate, cube));
    }

    [Fact]
    public void EmptyWeightMap_Throws()
    {
        var option = MakeBasketOption(legs: [], BasketAggregationMethod.WeightedSum, strike: 1.0, isCall: true);
        var cube   = SimulationCube.GenerateIndependent(1_000, Steps, 1, Seed);
        Assert.Throws<ArgumentException>(() =>
            new FXBasketVanillaMCPricer(option, ValuationDate, cube));
    }

    [Fact]
    public void UnspecifiedAggregationMethod_Throws()
    {
        var option = MakeBasketOption(
            legs: [("EURUSD", 1.0)], BasketAggregationMethod.Unspecified, strike: 1.0, isCall: true);
        var cube   = SimulationCube.GenerateIndependent(1_000, Steps, 1, Seed);

        Assert.Throws<ArgumentException>(() => new FXBasketVanillaMCPricer(option, ValuationDate, cube));
    }

    [Fact]
    public void CubeAssetCountMismatch_Throws()
    {
        var option = MakeBasketOption(
            legs: [("EURUSD", 0.5), ("GBPUSD", 0.5)], BasketAggregationMethod.WeightedSum, strike: 1.0, isCall: true);

        var wrongCube = SimulationCube.GenerateIndependent(1_000, Steps, assets: 1, Seed); // should be 2
        Assert.Throws<ArgumentException>(() =>
            new FXBasketVanillaMCPricer(option, ValuationDate, wrongCube));
    }

    [Fact]
    public void NonCurrencyPairLegIdentifier_Throws()
    {
        // "AAPL" is a 4-letter equity ticker, not a 6-letter FX pair — must be rejected
        // for an FX basket (the "currency pair object" check).
        var option = MakeBasketOption(
            legs: [("AAPL", 1.0)], BasketAggregationMethod.WeightedSum, strike: 1.0, isCall: true);
        var cube   = SimulationCube.GenerateIndependent(1_000, Steps, 1, Seed);

        Assert.Throws<ArgumentException>(() => new FXBasketVanillaMCPricer(option, ValuationDate, cube));
    }

    [Fact]
    public void ZeroStrike_Throws()
    {
        var option = MakeBasketOption(legs: [("EURUSD", 1.0)], BasketAggregationMethod.WeightedSum, strike: 0.0, isCall: true);
        var cube   = SimulationCube.GenerateIndependent(1_000, Steps, 1, Seed);

        Assert.Throws<ArgumentException>(() => new FXBasketVanillaMCPricer(option, ValuationDate, cube));
    }

    [Fact]
    public void MissingMarketDataForLeg_Throws()
    {
        var option = MakeBasketOption(
            legs: [("EURUSD", 0.5), ("GBPUSD", 0.5)], BasketAggregationMethod.WeightedSum, strike: 1.0, isCall: true);
        // Only EURUSD provided — GBPUSD missing
        var markets = new Dictionary<string, FxMarketData> { ["EURUSD"] = MakeMarket("EURUSD", 1.10, 0.05, 0.02) };
        var vols    = new Dictionary<string, double> { ["EURUSD"] = 0.20, ["GBPUSD"] = 0.25 };
        var cube    = SimulationCube.GenerateIndependent(1_000, Steps, assets: 2, Seed);

        using var pricer = new FXBasketVanillaMCPricer(option, ValuationDate, cube);
        Assert.Throws<ArgumentException>(() => pricer.Price(markets, vols));
    }

    [Fact]
    public void MismatchedCurrencyPairLabel_Throws()
    {
        // Basket key says "EURUSD" but the supplied FxMarketData is labelled "EURGBP".
        var option = MakeBasketOption(
            legs: [("EURUSD", 1.0)], BasketAggregationMethod.WeightedSum, strike: 1.0, isCall: true);
        var markets = new Dictionary<string, FxMarketData> { ["EURUSD"] = MakeMarket("EURGBP", 1.10, 0.05, 0.02) };
        var vols    = new Dictionary<string, double> { ["EURUSD"] = 0.20 };
        var cube    = SimulationCube.GenerateIndependent(1_000, Steps, 1, Seed);

        using var pricer = new FXBasketVanillaMCPricer(option, ValuationDate, cube);
        Assert.Throws<ArgumentException>(() => pricer.Price(markets, vols));
    }

    [Fact]
    public void NegativeVolatility_Throws()
    {
        var option = MakeBasketOption(legs: [("EURUSD", 1.0)], BasketAggregationMethod.WeightedSum, strike: 1.0, isCall: true);
        var markets = new Dictionary<string, FxMarketData> { ["EURUSD"] = MakeMarket("EURUSD", 1.10, 0.05, 0.02) };
        var vols    = new Dictionary<string, double> { ["EURUSD"] = -0.1 };
        var cube    = SimulationCube.GenerateIndependent(1_000, Steps, 1, Seed);

        using var pricer = new FXBasketVanillaMCPricer(option, ValuationDate, cube);
        Assert.Throws<ArgumentOutOfRangeException>(() => pricer.Price(markets, vols));
    }

    // ── CurrencyPair parsing — exercised indirectly through the public ctor ───
    // (TryParseCurrencyPair is `internal`; no InternalsVisibleTo to the test
    // assembly, so we validate the parsing contract via the only public surface
    // that uses it — the constructor's per-leg validation.)

    [Theory]
    [InlineData("EUR/USD")]   // separator form, slash
    [InlineData("eur-usd")]   // separator form, dash, lower-case
    [InlineData("eurusd")]    // bare 6-letter code, lower-case
    public void SeparatedOrLowercaseCurrencyPairIdentifiers_AreAccepted(string id)
    {
        // Market data and the basket key must agree on the identifier (validated
        // by string equality), so we key everything consistently by `id` itself —
        // the point here is purely that construction succeeds, i.e. `id` parses
        // as a valid currency pair.
        var option = MakeBasketOption(legs: [(id, 1.0)], BasketAggregationMethod.WeightedSum, strike: 1.0, isCall: true);
        var cube   = SimulationCube.GenerateIndependent(1_000, Steps, 1, Seed);

        using var pricer = new FXBasketVanillaMCPricer(option, ValuationDate, cube);
        Assert.Single(pricer.LegNames);
    }

    [Theory]
    [InlineData("AAPL")]       // equity ticker, 4 letters
    [InlineData("EU1USD")]     // contains digit
    [InlineData("EURUSDX")]    // 7 letters, no separator
    [InlineData("USDUSD")]     // base == quote
    [InlineData("EU/USD")]     // base too short
    public void MalformedCurrencyPairIdentifiers_AreRejected(string id)
    {
        var option = MakeBasketOption(legs: [(id, 1.0)], BasketAggregationMethod.WeightedSum, strike: 1.0, isCall: true);
        var cube   = SimulationCube.GenerateIndependent(1_000, Steps, 1, Seed);

        Assert.Throws<ArgumentException>(() => new FXBasketVanillaMCPricer(option, ValuationDate, cube));
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private static Option MakeBasketOption(
        (string name, double weight)[] legs,
        BasketAggregationMethod method,
        double strike,
        bool isCall)
    {
        var basket = new Basket { AggregationMethod = method };
        foreach (var (name, weight) in legs)
            basket.UnderlyingToWeight[name] = weight;

        return new Option
        {
            Basket        = basket,
            Strike        = strike,
            ExpiryYears   = T,
            OptionType    = isCall ? OptionType.Call : OptionType.Put,
            ExerciseStyle = ExerciseStyle.European,
            Vanilla       = new VanillaOption()
        };
    }

    private static FxMarketData MakeMarket(string pair, double spot, double rd, double rf) =>
        new() { CurrencyPair = pair, Spot = spot,
                DomesticRate = ZeroCurve.Flat(rd, CurveMaturity),
                ForeignRate  = ZeroCurve.Flat(rf, CurveMaturity),
                VolSurface = new VolatilitySurface() };

    private (Dictionary<string, FxMarketData> markets, Dictionary<string, double> vols) MakeTwoLegMarkets(
        double spot1, double rd1, double rf1, double vol1,
        double spot2, double rd2, double rf2, double vol2,
        string name1, string name2)
    {
        var markets = new Dictionary<string, FxMarketData>
        {
            [name1] = MakeMarket(name1, spot1, rd1, rf1),
            [name2] = MakeMarket(name2, spot2, rd2, rf2)
        };
        var vols = new Dictionary<string, double> { [name1] = vol1, [name2] = vol2 };
        return (markets, vols);
    }

    /// <summary>
    /// Two zero-vol legs ⇒ deterministic terminal spots. Returns the markets/vols
    /// dictionaries plus the (exact) terminal spot of each leg and the common
    /// domestic rate, for building exact expected prices.
    /// </summary>
    private (Dictionary<string, FxMarketData> markets, Dictionary<string, double> vols) BuildTwoZeroVolLegs(
        out double spot1Terminal, out double spot2Terminal, out double domesticRate)
    {
        const double spot1 = 1.10, rd1 = 0.05, rf1 = 0.02;
        const double spot2 = 1.20, rd2 = 0.05, rf2 = 0.01;

        spot1Terminal = spot1 * Math.Exp((rd1 - rf1) * T);
        spot2Terminal = spot2 * Math.Exp((rd2 - rf2) * T);
        domesticRate  = rd1;

        var markets = new Dictionary<string, FxMarketData>
        {
            ["EURUSD"] = MakeMarket("EURUSD", spot1, rd1, rf1),
            ["GBPUSD"] = MakeMarket("GBPUSD", spot2, rd2, rf2)
        };
        var vols = new Dictionary<string, double> { ["EURUSD"] = 0.0, ["GBPUSD"] = 0.0 };
        return (markets, vols);
    }

    private static void AssertWithinMcBounds(PricingResult mc, double analytical, double sigma)
    {
        var tol = sigma * mc.StandardError;
        Assert.True(Math.Abs(mc.Price - analytical) < tol,
            $"MC={mc.Price:F6}  Ref={analytical:F6}  Diff={Math.Abs(mc.Price - analytical):F6}  {sigma}σ tol={tol:F6}");
    }
}
