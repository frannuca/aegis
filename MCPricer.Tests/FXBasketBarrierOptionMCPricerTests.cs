using Aegis.Instruments;
using MCPricer.FX;
using RandomSimulator;

namespace MCPricer.Tests;

/// <summary>
/// Validates FXBasketBarrierOptionMCPricer — a single-barrier (knock-in/out)
/// option monitored on the *aggregated level* of a basket of correlated FX legs.
///
/// ── Benchmark strategy ───────────────────────────────────────────────────────
///
///   1. Single-leg basket degeneracy (the strongest check): with N=1, weight=1,
///      the basket level path is *identical* to the bare leg's spot path, and —
///      per the pricer's XML doc — the local-volatility proxy σ_eff used for
///      continuous-monitoring bridge sampling collapses *exactly* to the flat
///      leg vol for every aggregation method (WEIGHTED_SUM, BEST_OF, WORST_OF
///      alike). So on the same cube, FXBasketBarrierOptionMCPricer must match
///      FXBarrierOptionMCPricer to within float-summation noise — for both
///      observation modes. This validates the σ_eff proxy directly.
///
///   2. KI + KO = Vanilla(basket level) parity (path-by-path identity, exact
///      with the same cube and zero rebate) — validates the barrier-switching
///      logic and basket-level mechanics (multi-leg correlation, aggregation)
///      together, independent of the σ_eff proxy (both KI and KO derive
///      barrier-hit from the same per-path level trajectory and RNG draws, so
///      whatever that determination is, it cancels in the sum).
///
///   3. Zero-volatility deterministic edge cases — every leg path, and hence
///      the basket-level path, is deterministic, so the barrier-hit outcome
///      and the price are exact closed-form numbers.
///
///   4. Far-barrier limit — an unreachable barrier never triggers, so the price
///      converges to the basket vanilla price (FXBasketVanillaMCPricer).
/// </summary>
public sealed class FXBasketBarrierOptionMCPricerTests
{
    private const double T = 1.0;

    // Single-leg comparison constants — mirror FXBarrierOptionMCPricerTests so
    // the degeneracy check (benchmark 1) compares like-for-like.
    private const double S0    = 1.10;
    private const double K     = 1.10;
    private const double Sigma = 0.20;
    private const double Rd    = 0.05;
    private const double Rf    = 0.02;
    private const double H_Up  = 1.30;
    private const double H_Down = 0.90;

    private const int Paths = 100_000;
    private const int Steps = 50;
    private const int Seed  = 42;

    private static readonly DateOnly ValuationDate = new(2026, 1, 1);
    private static readonly DateOnly CurveMaturity = new(2036, 1, 1);

    // ── 1. Single-leg basket degenerates to FXBarrierOptionMCPricer ──────────

    [Theory]
    [InlineData(BasketAggregationMethod.WeightedSum)]
    [InlineData(BasketAggregationMethod.BestOf)]
    [InlineData(BasketAggregationMethod.WorstOf)]
    public void SingleLeg_Discrete_MatchesSingleAssetBarrier(BasketAggregationMethod method)
    {
        var (basketResult, singleResult) = RunSingleLegComparison(method, BarrierObservation.Discrete);

        Assert.True(Math.Abs(basketResult.Price - singleResult.Price) < 1e-9,
            $"Basket(N=1,{method},Discrete)={basketResult.Price:F10}  Single={singleResult.Price:F10}  " +
            $"Diff={Math.Abs(basketResult.Price - singleResult.Price):E3}");
    }

    [Theory]
    [InlineData(BasketAggregationMethod.WeightedSum)]
    [InlineData(BasketAggregationMethod.BestOf)]
    [InlineData(BasketAggregationMethod.WorstOf)]
    public void SingleLeg_Continuous_MatchesSingleAssetBarrier(BasketAggregationMethod method)
    {
        // The interesting case: validates that σ_eff (the local-vol bridge proxy)
        // collapses to the flat leg vol exactly, for all three aggregation branches.
        var (basketResult, singleResult) = RunSingleLegComparison(method, BarrierObservation.Continuous);

        Assert.True(Math.Abs(basketResult.Price - singleResult.Price) < 1e-9,
            $"Basket(N=1,{method},Continuous)={basketResult.Price:F10}  Single={singleResult.Price:F10}  " +
            $"Diff={Math.Abs(basketResult.Price - singleResult.Price):E3}");
    }

    private (PricingResult basket, PricingResult single) RunSingleLegComparison(
        BasketAggregationMethod method, BarrierObservation observation)
    {
        var cube   = SimulationCube.GenerateIndependent(Paths, Steps, assets: 1, Seed);
        var market = MakeMarket("EURUSD", S0, Rd, Rf);

        var basketOption = MakeBasketBarrierOption(
            legs: [("EURUSD", 1.0)], method, K, isCall: true, H_Up, BarrierType.UpAndOut, observation);
        var markets = new Dictionary<string, FxMarketData> { ["EURUSD"] = market };
        var vols    = new Dictionary<string, double> { ["EURUSD"] = Sigma };

        using var basketPricer = new FXBasketBarrierOptionMCPricer(basketOption, ValuationDate, markets, vols, cube);
        var basketResult = basketPricer.Price();

        var singleOption = new Option
        {
            Underlying    = "EURUSD",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Barrier = new BarrierOption
            {
                BarrierLevel = H_Up,
                BarrierType  = BarrierType.UpAndOut,
                Observation  = observation,
                Rebate       = 0.0
            }
        };
        using var singlePricer = new FXBarrierOptionMCPricer(singleOption, ValuationDate, market, Sigma, cube);
        var singleResult = singlePricer.Price();

        return (basketResult, singleResult);
    }

    // ── 2. KI + KO = Vanilla(basket level) parity ────────────────────────────

    [Theory]
    [InlineData(true,  BarrierType.UpAndIn,   BarrierType.UpAndOut,   1.30)]
    [InlineData(false, BarrierType.UpAndIn,   BarrierType.UpAndOut,   1.30)]
    [InlineData(true,  BarrierType.DownAndIn, BarrierType.DownAndOut, 1.05)]
    [InlineData(false, BarrierType.DownAndIn, BarrierType.DownAndOut, 1.05)]
    public void KnockInPlusKnockOut_EqualsBasketVanilla_Discrete(
        bool isCall, BarrierType kiType, BarrierType koType, double h)
    {
        AssertKnockInPlusKnockOutEqualsVanilla(isCall, kiType, koType, h, BarrierObservation.Discrete);
    }

    [Theory]
    [InlineData(true,  BarrierType.UpAndIn,   BarrierType.UpAndOut,   1.30)]
    [InlineData(false, BarrierType.DownAndIn, BarrierType.DownAndOut, 1.05)]
    public void KnockInPlusKnockOut_EqualsBasketVanilla_Continuous(
        bool isCall, BarrierType kiType, BarrierType koType, double h)
    {
        AssertKnockInPlusKnockOutEqualsVanilla(isCall, kiType, koType, h, BarrierObservation.Continuous);
    }

    private void AssertKnockInPlusKnockOutEqualsVanilla(
        bool isCall, BarrierType kiType, BarrierType koType, double h, BarrierObservation observation)
    {
        const double w1 = 0.5, w2 = 0.5, strike = 1.18;
        var legs = MakeTwoLegMarkets(
            spot1: 1.10, rd1: 0.05, rf1: 0.02, vol1: 0.20,
            spot2: 1.25, rd2: 0.05, rf2: 0.01, vol2: 0.25,
            name1: "EURUSD", name2: "GBPUSD");

        var cube = SimulationCube.Generate(Paths, Steps, assets: 2,
            correlation: new double[,] { { 1.0, 0.3 }, { 0.3, 1.0 } }, Seed);

        var legSpec = new (string, double)[] { ("EURUSD", w1), ("GBPUSD", w2) };

        var vanilla = new FXBasketVanillaMCPricer(
            MakeBasketOption(legSpec, BasketAggregationMethod.WeightedSum, strike, isCall),
            ValuationDate, legs.markets, legs.vols, cube).Price();

        var ki = new FXBasketBarrierOptionMCPricer(
            MakeBasketBarrierOption(legSpec, BasketAggregationMethod.WeightedSum, strike, isCall, h, kiType, observation, rebate: 0.0),
            ValuationDate, legs.markets, legs.vols, cube).Price();

        var ko = new FXBasketBarrierOptionMCPricer(
            MakeBasketBarrierOption(legSpec, BasketAggregationMethod.WeightedSum, strike, isCall, h, koType, observation, rebate: 0.0),
            ValuationDate, legs.markets, legs.vols, cube).Price();

        // Same cube ⇒ same per-path level trajectories ⇒ same per-path barrier-hit
        // determinations for KI and KO ⇒ KI + KO = Vanilla exactly, up to float rounding.
        Assert.True(
            Math.Abs(ki.Price + ko.Price - vanilla.Price) < 1e-9,
            $"KI={ki.Price:F8} + KO={ko.Price:F8} = {ki.Price + ko.Price:F8}, vanilla={vanilla.Price:F8}");
    }

    // ── 3. Zero-volatility edge cases (deterministic basket-level path) ──────

    [Fact]
    public void ZeroVol_UpAndOut_BarrierNeverReached_PricesAtDiscountedIntrinsic()
    {
        // Both legs have r_d > r_f,i ⇒ each S_i(t), and hence the weighted-sum
        // level(t) = Σ w_i·S_i(t), increases monotonically over [0,T] from
        // level(0)=1.15 to level(T)≈1.1912 (see BuildTwoZeroVolLegs). H=1.30 is
        // comfortably above this entire range ⇒ the up-barrier never triggers,
        // so UpAndOut pays the full undiscounted-then-discounted vanilla payoff.
        var legs = BuildTwoZeroVolLegs(out var s1T, out var s2T, out var rd);
        const double w1 = 0.5, w2 = 0.5, strike = 1.18, barrier = 1.30;

        var levelT   = w1 * s1T + w2 * s2T;
        var expected = Math.Exp(-rd * T) * Math.Max(levelT - strike, 0.0);

        var cube = SimulationCube.GenerateIndependent(Paths, Steps, assets: 2, Seed);
        var legSpec = new (string, double)[] { ("EURUSD", w1), ("GBPUSD", w2) };
        var mc = new FXBasketBarrierOptionMCPricer(
            MakeBasketBarrierOption(legSpec, BasketAggregationMethod.WeightedSum, strike, isCall: true,
                barrier, BarrierType.UpAndOut, BarrierObservation.Discrete),
            ValuationDate, legs.markets, legs.vols, cube).Price();

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_DownAndIn_BarrierAlwaysBreached_PricesAtDiscountedIntrinsic()
    {
        // Same deterministic level path, level(t) ∈ [1.15, 1.1912]. H=1.25 sits
        // comfortably *above* this entire range, so level(t) ≤ H holds at every
        // monitoring date ⇒ the down-barrier ("breached" ⇔ level ≤ H) triggers
        // immediately and stays activated ⇒ DownAndIn pays the vanilla payoff.
        var legs = BuildTwoZeroVolLegs(out var s1T, out var s2T, out var rd);
        const double w1 = 0.5, w2 = 0.5, strike = 1.18, barrier = 1.25;

        var levelT   = w1 * s1T + w2 * s2T;
        var expected = Math.Exp(-rd * T) * Math.Max(levelT - strike, 0.0);

        var cube = SimulationCube.GenerateIndependent(Paths, Steps, assets: 2, Seed);
        var legSpec = new (string, double)[] { ("EURUSD", w1), ("GBPUSD", w2) };
        var mc = new FXBasketBarrierOptionMCPricer(
            MakeBasketBarrierOption(legSpec, BasketAggregationMethod.WeightedSum, strike, isCall: true,
                barrier, BarrierType.DownAndIn, BarrierObservation.Discrete),
            ValuationDate, legs.markets, legs.vols, cube).Price();

        Assert.Equal(expected, mc.Price, precision: 10);
    }

    [Fact]
    public void ZeroVol_UpAndIn_BarrierNeverReached_PricesAtZero()
    {
        // Same setup as the UpAndOut never-reached test: the up-barrier is never
        // activated, so UpAndIn (rebate=0) pays nothing on every path.
        var legs = BuildTwoZeroVolLegs(out _, out _, out _);
        const double w1 = 0.5, w2 = 0.5, strike = 1.18, barrier = 1.30;

        var cube = SimulationCube.GenerateIndependent(Paths, Steps, assets: 2, Seed);
        var legSpec = new (string, double)[] { ("EURUSD", w1), ("GBPUSD", w2) };
        var mc = new FXBasketBarrierOptionMCPricer(
            MakeBasketBarrierOption(legSpec, BasketAggregationMethod.WeightedSum, strike, isCall: true,
                barrier, BarrierType.UpAndIn, BarrierObservation.Discrete),
            ValuationDate, legs.markets, legs.vols, cube).Price();

        Assert.Equal(0.0, mc.Price, precision: 10);
    }

    // ── 4. Far-barrier limit → matches basket vanilla ────────────────────────

    [Fact]
    public void UpAndOut_BarrierFarAway_EqualsBasketVanilla()
    {
        const double w1 = 0.5, w2 = 0.5, strike = 1.18;
        var legs = MakeTwoLegMarkets(
            spot1: 1.10, rd1: 0.05, rf1: 0.02, vol1: 0.20,
            spot2: 1.25, rd2: 0.05, rf2: 0.01, vol2: 0.25,
            name1: "EURUSD", name2: "GBPUSD");
        var legSpec = new (string, double)[] { ("EURUSD", w1), ("GBPUSD", w2) };

        var cube = SimulationCube.Generate(Paths, Steps, assets: 2,
            correlation: new double[,] { { 1.0, 0.3 }, { 0.3, 1.0 } }, Seed);

        var barrier = new FXBasketBarrierOptionMCPricer(
            MakeBasketBarrierOption(legSpec, BasketAggregationMethod.WeightedSum, strike, isCall: true,
                barrierLevel: 100.0, BarrierType.UpAndOut, BarrierObservation.Discrete),
            ValuationDate, legs.markets, legs.vols, cube).Price();

        var vanilla = new FXBasketVanillaMCPricer(
            MakeBasketOption(legSpec, BasketAggregationMethod.WeightedSum, strike, isCall: true),
            ValuationDate, legs.markets, legs.vols, cube).Price();

        AssertWithinMcBounds(barrier, vanilla.Price, sigma: 5.0);
    }

    // ── Input validation ──────────────────────────────────────────────────────

    [Fact]
    public void NonBasketUnderlying_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, Steps, assets: 1, Seed);
        var option = new Option
        {
            Underlying    = "EURUSD",   // single underlying, not a Basket
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Barrier = new BarrierOption { BarrierLevel = H_Up, BarrierType = BarrierType.UpAndOut, Observation = BarrierObservation.Discrete }
        };
        var markets = new Dictionary<string, FxMarketData> { ["EURUSD"] = MakeMarket("EURUSD", S0, Rd, Rf) };
        var vols    = new Dictionary<string, double> { ["EURUSD"] = Sigma };

        Assert.Throws<ArgumentException>(() =>
            new FXBasketBarrierOptionMCPricer(option, ValuationDate, markets, vols, cube));
    }

    [Fact]
    public void NonBarrierKind_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, Steps, assets: 1, Seed);
        var legSpec = new (string, double)[] { ("EURUSD", 1.0) };
        var option = MakeBasketOption(legSpec, BasketAggregationMethod.WeightedSum, K, isCall: true); // Vanilla, not Barrier
        var markets = new Dictionary<string, FxMarketData> { ["EURUSD"] = MakeMarket("EURUSD", S0, Rd, Rf) };
        var vols    = new Dictionary<string, double> { ["EURUSD"] = Sigma };

        Assert.Throws<ArgumentException>(() =>
            new FXBasketBarrierOptionMCPricer(option, ValuationDate, markets, vols, cube));
    }

    [Fact]
    public void UnspecifiedBarrierType_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, Steps, assets: 1, Seed);
        var legSpec = new (string, double)[] { ("EURUSD", 1.0) };
        var option = MakeBasketBarrierOption(legSpec, BasketAggregationMethod.WeightedSum, K, isCall: true,
            H_Up, BarrierType.Unspecified, BarrierObservation.Discrete);
        var markets = new Dictionary<string, FxMarketData> { ["EURUSD"] = MakeMarket("EURUSD", S0, Rd, Rf) };
        var vols    = new Dictionary<string, double> { ["EURUSD"] = Sigma };

        Assert.Throws<ArgumentException>(() =>
            new FXBasketBarrierOptionMCPricer(option, ValuationDate, markets, vols, cube));
    }

    [Fact]
    public void UnspecifiedObservation_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, Steps, assets: 1, Seed);
        var legSpec = new (string, double)[] { ("EURUSD", 1.0) };
        var option = MakeBasketBarrierOption(legSpec, BasketAggregationMethod.WeightedSum, K, isCall: true,
            H_Up, BarrierType.UpAndOut, BarrierObservation.Unspecified);
        var markets = new Dictionary<string, FxMarketData> { ["EURUSD"] = MakeMarket("EURUSD", S0, Rd, Rf) };
        var vols    = new Dictionary<string, double> { ["EURUSD"] = Sigma };

        Assert.Throws<ArgumentException>(() =>
            new FXBasketBarrierOptionMCPricer(option, ValuationDate, markets, vols, cube));
    }

    [Fact]
    public void ZeroBarrierLevel_Throws()
    {
        var cube = SimulationCube.GenerateIndependent(100, Steps, assets: 1, Seed);
        var legSpec = new (string, double)[] { ("EURUSD", 1.0) };
        var option = MakeBasketBarrierOption(legSpec, BasketAggregationMethod.WeightedSum, K, isCall: true,
            barrierLevel: 0.0, BarrierType.UpAndOut, BarrierObservation.Discrete);
        var markets = new Dictionary<string, FxMarketData> { ["EURUSD"] = MakeMarket("EURUSD", S0, Rd, Rf) };
        var vols    = new Dictionary<string, double> { ["EURUSD"] = Sigma };

        Assert.Throws<ArgumentException>(() =>
            new FXBasketBarrierOptionMCPricer(option, ValuationDate, markets, vols, cube));
    }

    [Fact]
    public void EmptyWeightMap_Throws()
    {
        var cube   = SimulationCube.GenerateIndependent(100, Steps, assets: 1, Seed);
        var basket = new Basket { AggregationMethod = BasketAggregationMethod.WeightedSum };
        var option = new Option
        {
            Basket = basket, Strike = K, ExpiryYears = T,
            OptionType = OptionType.Call, ExerciseStyle = ExerciseStyle.European,
            Barrier = new BarrierOption { BarrierLevel = H_Up, BarrierType = BarrierType.UpAndOut, Observation = BarrierObservation.Discrete }
        };
        var markets = new Dictionary<string, FxMarketData>();
        var vols    = new Dictionary<string, double>();

        Assert.Throws<ArgumentException>(() =>
            new FXBasketBarrierOptionMCPricer(option, ValuationDate, markets, vols, cube));
    }

    [Fact]
    public void CubeAssetCountMismatch_Throws()
    {
        // Two legs declared, but the cube carries only one correlated factor.
        var cube = SimulationCube.GenerateIndependent(100, Steps, assets: 1, Seed);
        var legSpec = new (string, double)[] { ("EURUSD", 0.5), ("GBPUSD", 0.5) };
        var option = MakeBasketBarrierOption(legSpec, BasketAggregationMethod.WeightedSum, K, isCall: true,
            H_Up, BarrierType.UpAndOut, BarrierObservation.Discrete);
        var markets = new Dictionary<string, FxMarketData>
        {
            ["EURUSD"] = MakeMarket("EURUSD", S0,   Rd, Rf),
            ["GBPUSD"] = MakeMarket("GBPUSD", 1.25, Rd, 0.01)
        };
        var vols = new Dictionary<string, double> { ["EURUSD"] = Sigma, ["GBPUSD"] = 0.25 };

        Assert.Throws<ArgumentException>(() =>
            new FXBasketBarrierOptionMCPricer(option, ValuationDate, markets, vols, cube));
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private static Option MakeBasketBarrierOption(
        (string name, double weight)[] legs,
        BasketAggregationMethod method,
        double strike, bool isCall,
        double barrierLevel, BarrierType barrierType, BarrierObservation observation,
        double rebate = 0.0)
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
            Barrier = new BarrierOption
            {
                BarrierLevel = barrierLevel,
                BarrierType  = barrierType,
                Observation  = observation,
                Rebate       = rebate
            }
        };
    }

    private static Option MakeBasketOption(
        (string name, double weight)[] legs,
        BasketAggregationMethod method,
        double strike, bool isCall)
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
        new()
        {
            CurrencyPair = pair,
            Spot         = spot,
            DomesticRate = ZeroCurve.Flat(rd, CurveMaturity),
            ForeignRate  = ZeroCurve.Flat(rf, CurveMaturity),
            VolSurface   = new VolatilitySurface()
        };

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
    /// Two zero-vol legs with r_d > r_f,i (both legs trend up) ⇒ both S_i(t),
    /// and hence the weighted-sum level(t) = 0.5·S_1(t) + 0.5·S_2(t), increase
    /// monotonically over [0,T]: level(0) = 1.15 → level(T) ≈ 1.1912.
    /// Returns markets/vols plus the (exact) terminal spot of each leg and the
    /// common domestic rate, for building exact expected prices.
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
