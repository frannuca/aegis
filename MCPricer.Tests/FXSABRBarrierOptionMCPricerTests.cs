using Aegis.Instruments;
using MCPricer.FX;
using RandomSimulator;
using NodaTime;

namespace MCPricer.Tests;

/// <summary>
/// Validates FXSABRBarrierOptionMCPricer via four complementary strategies.
///
/// ── 1. KI + KO = SABR vanilla parity (exact, same cube) ─────────────────────
/// For any path ω:
///   KnockIn_payoff(ω) + KnockOut_payoff(ω) = Vanilla_payoff(ω)   (rebate = 0)
///
/// Pricing all three on the same SimulationCube makes the identity exact at the
/// level of floating-point summation across paths.  No analytical formula required.
/// Tested for up/down × call/put × discrete/continuous combinations.
///
/// ── 2. ν=0 degeneracy: SABR → GBM ──────────────────────────────────────────
/// When ν=0 (zero vol-of-vol), σ is constant at α: SABR reduces to GBM with σ=α.
/// The SABR barrier price must agree with FXBarrierOptionMCPricer priced at σ=α,
/// within the 5σ Monte Carlo confidence interval (cubes are independent).
/// This is the strongest model-degeneracy correctness check.
///
/// ── 3. Far-barrier limit ──────────────────────────────────────────────────────
/// A barrier far from the initial forward (H=100×F for UpAndOut) is never reached;
/// the barrier option price must equal the SABR vanilla price (same cube, exact).
///
/// ── 4. Qualitative ordering ───────────────────────────────────────────────────
/// Knock-out can only remove value relative to the vanilla:
///   0 ≤ UpAndOut price ≤ SABR vanilla price  (5σ slack for MC noise)
///
/// Continuous monitoring detects more crossings than discrete (in expectation):
///   continuous UpAndOut price ≤ discrete UpAndOut price  (5σ slack)
///
/// ── Barrier convention ───────────────────────────────────────────────────────
/// Barrier is monitored on the forward F (not the spot S), consistent with the
/// T-forward measure under which the SABR SDE is simulated.
/// </summary>
public sealed class FXSABRBarrierOptionMCPricerTests
{
    // Reference market: EURUSD
    private const double S0 = 1.10;
    private const double Rd = 0.05;
    private const double Rf = 0.02;
    private const double T  = 1.0;

    private static readonly LocalDate ValuationDate = new(2026, 1, 1);
    private static readonly LocalDate CurveMaturity = new(2036, 1, 1);

    // Reference SABR parameters (lognormal backbone)
    private const double Alpha = 0.20;
    private const double Beta1 = 1.0;
    private const double Rho   = -0.25;
    private const double Nu    = 0.30;

    // Barrier levels relative to the forward F(0) ≈ 1.10·exp(0.03) ≈ 1.133
    private const double H_Up   = 1.25;   // above initial forward (~10% up from F0)
    private const double H_Down = 1.00;   // below initial forward (~11% down from F0)
    private const double K      = 1.10;   // strike = spot (ATM-spot)

    private const int Paths       = 200_000;
    private const int Steps       = 100;      // SABR needs more steps than GBM
    private const int DefaultSeed = 42;

    // ── 1. KI + KO = SABR vanilla parity (exact, same cube) ─────────────────

    [Theory]
    [InlineData(true,  BarrierType.UpAndIn,   BarrierType.UpAndOut,   H_Up,   BarrierObservation.Discrete,   "up-discrete call")]
    [InlineData(false, BarrierType.UpAndIn,   BarrierType.UpAndOut,   H_Up,   BarrierObservation.Discrete,   "up-discrete put")]
    [InlineData(true,  BarrierType.DownAndIn, BarrierType.DownAndOut, H_Down, BarrierObservation.Discrete,   "down-discrete call")]
    [InlineData(false, BarrierType.DownAndIn, BarrierType.DownAndOut, H_Down, BarrierObservation.Discrete,   "down-discrete put")]
    [InlineData(true,  BarrierType.UpAndIn,   BarrierType.UpAndOut,   H_Up,   BarrierObservation.Continuous, "up-continuous call")]
    [InlineData(false, BarrierType.UpAndIn,   BarrierType.UpAndOut,   H_Up,   BarrierObservation.Continuous, "up-continuous put")]
    [InlineData(true,  BarrierType.DownAndIn, BarrierType.DownAndOut, H_Down, BarrierObservation.Continuous, "down-continuous call")]
    [InlineData(false, BarrierType.DownAndIn, BarrierType.DownAndOut, H_Down, BarrierObservation.Continuous, "down-continuous put")]
    public void KnockInPlusKnockOut_EqualsSabrVanilla(
        bool isCall, BarrierType kiType, BarrierType koType, double h,
        BarrierObservation obs, string _)
    {
        var cube   = BuildCube();
        var market = MakeMarket();

        var vanilla = BuildSabrVanilla(K, Alpha, Beta1, Rho, Nu, isCall, cube).Price(market);
        var ki      = BuildSabrBarrier(K, Alpha, Beta1, Rho, Nu, h, kiType, obs, isCall, rebate: 0.0, cube).Price(market);
        var ko      = BuildSabrBarrier(K, Alpha, Beta1, Rho, Nu, h, koType, obs, isCall, rebate: 0.0, cube).Price(market);

        // Same cube → same path payoffs → KI + KO = Vanilla exactly up to float rounding.
        Assert.True(
            Math.Abs(ki.Price + ko.Price - vanilla.Price) < 1e-10,
            $"KI={ki.Price:F8} + KO={ko.Price:F8} = {ki.Price + ko.Price:F8}, " +
            $"vanilla={vanilla.Price:F8}  diff={Math.Abs(ki.Price + ko.Price - vanilla.Price):E3}");
    }

    // ── 2. ν=0 degeneracy: SABR → GBM (r_d = r_f so F(t,T) = S(t) for all t) ──

    // The SABR pricer monitors the forward F(t,T); the GBM barrier pricer monitors
    // the spot S(t). Under covered-interest parity, F(t,T) = S(t)·exp((r_d−r_f)·(T−t)).
    // When r_d = r_f, this equals S(t) for ALL monitoring dates, so the two barrier
    // specifications coincide and ν=0 SABR must match the GBM barrier pricer (5σ bound).
    private const double RdEq = 0.04;   // equal domestic/foreign rate for this test block
    private const double RfEq = 0.04;
    private const double H_UpEq   = 1.30;  // above S0=1.10 (= F(0) when rd=rf)
    private const double H_DownEq = 0.90;  // below S0=1.10

    [Theory]
    [InlineData(true,  BarrierType.UpAndOut,   H_UpEq,   BarrierObservation.Discrete,   "UpAndOut call discrete")]
    [InlineData(false, BarrierType.UpAndOut,   H_UpEq,   BarrierObservation.Discrete,   "UpAndOut put discrete")]
    [InlineData(true,  BarrierType.DownAndOut, H_DownEq, BarrierObservation.Discrete,   "DownAndOut call discrete")]
    [InlineData(true,  BarrierType.UpAndIn,    H_UpEq,   BarrierObservation.Discrete,   "UpAndIn call discrete")]
    [InlineData(true,  BarrierType.UpAndOut,   H_UpEq,   BarrierObservation.Continuous, "UpAndOut call continuous")]
    public void ZeroVolOfVol_EqualRates_SabrBarrier_MatchesGbmBarrier(
        bool isCall, BarrierType barrierType, double h, BarrierObservation obs, string _)
    {
        const int    paths = 500_000;
        const int    steps = 100;
        const int    seed  = 42;

        var sabrCube  = SimulationCube.GenerateIndependent(paths, steps, assets: 2, seed);
        var gbmCube   = SimulationCube.GenerateIndependent(paths, steps, assets: 1, seed + 1);
        var eqMarket  = MakeMarketWithRates(RdEq, RfEq);

        var sabrPrice = BuildSabrBarrier(K, Alpha, Beta1, rho: 0.0, nu: 0.0,
                            h, barrierType, obs, isCall, rebate: 0.0, sabrCube).Price(eqMarket);
        var gbmPrice  = BuildGbmBarrier(K, Alpha, h, barrierType, obs, isCall, rebate: 0.0, gbmCube).Price(eqMarket);

        var tolerance = 5.0 * (sabrPrice.StandardError + gbmPrice.StandardError);
        Assert.True(Math.Abs(sabrPrice.Price - gbmPrice.Price) < tolerance,
            $"SABR(ν=0,r_d=r_f)={sabrPrice.Price:F6} ± {sabrPrice.StandardError:F6}  " +
            $"GBM={gbmPrice.Price:F6} ± {gbmPrice.StandardError:F6}  " +
            $"5σ tol={tolerance:F6}");
    }

    // ── 3. Far barrier → SABR vanilla (same cube; exact within MC rounding) ────

    [Theory]
    [InlineData(true,  BarrierType.UpAndOut,   "call")]
    [InlineData(false, BarrierType.UpAndOut,   "put")]
    [InlineData(true,  BarrierType.DownAndOut, "call")]
    [InlineData(false, BarrierType.DownAndOut, "put")]
    public void FarBarrier_EqualsVanilla(bool isCall, BarrierType barrierType, string _)
    {
        // A barrier placed far from the forward is never reached;
        // the knock-out never activates → price = SABR vanilla.
        // Same cube → identical path realizations → exact equality.
        var farH = barrierType is BarrierType.UpAndOut ? 1e6 : 1e-6;
        var cube   = BuildCube();
        var market = MakeMarket();

        var barrier = BuildSabrBarrier(K, Alpha, Beta1, Rho, Nu, farH,
                          barrierType, BarrierObservation.Discrete, isCall, rebate: 0.0, cube).Price(market);
        var vanilla = BuildSabrVanilla(K, Alpha, Beta1, Rho, Nu, isCall, cube).Price(market);

        Assert.Equal(vanilla.Price, barrier.Price, precision: 10);
    }

    // ── 4. UpAndOut price ≤ vanilla price (5σ slack) ─────────────────────────

    [Fact]
    public void UpAndOut_PriceLessThanOrEqualToVanilla()
    {
        var cube   = BuildCube();
        var market = MakeMarket();

        var barrier = BuildSabrBarrier(K, Alpha, Beta1, Rho, Nu, H_Up,
                          BarrierType.UpAndOut, BarrierObservation.Discrete,
                          isCall: true, rebate: 0.0, cube).Price(market);
        var vanilla = BuildSabrVanilla(K, Alpha, Beta1, Rho, Nu, isCall: true, cube).Price(market);

        var slack = 5.0 * (barrier.StandardError + vanilla.StandardError);
        Assert.True(barrier.Price <= vanilla.Price + slack,
            $"UpAndOut={barrier.Price:F6}  Vanilla={vanilla.Price:F6}  5σ slack={slack:F6}");
    }

    // ── 5. Continuous monitoring ≤ discrete monitoring (UpAndOut) ────────────

    [Fact]
    public void Continuous_UpAndOut_NotGreaterThanDiscrete()
    {
        // Continuous monitoring detects more barrier crossings → more knockouts
        // → lower UpAndOut price.  Same cube for both.
        var cube   = BuildCube();
        var market = MakeMarket();

        var discrete   = BuildSabrBarrier(K, Alpha, Beta1, Rho, Nu, H_Up,
                             BarrierType.UpAndOut, BarrierObservation.Discrete,
                             isCall: true, rebate: 0.0, cube).Price(market);
        var continuous = BuildSabrBarrier(K, Alpha, Beta1, Rho, Nu, H_Up,
                             BarrierType.UpAndOut, BarrierObservation.Continuous,
                             isCall: true, rebate: 0.0, cube).Price(market);

        var tolerance = 5.0 * (discrete.StandardError + continuous.StandardError);
        Assert.True(continuous.Price <= discrete.Price + tolerance,
            $"Continuous={continuous.Price:F6}  Discrete={discrete.Price:F6}  5σ slack={tolerance:F6}");
    }

    // ── Rebate monotonicity ───────────────────────────────────────────────────

    [Fact]
    public void UpAndOut_RebateIncreasesPrice()
    {
        var cube   = BuildCube();
        var market = MakeMarket();

        var noRebate   = BuildSabrBarrier(K, Alpha, Beta1, Rho, Nu, H_Up,
                             BarrierType.UpAndOut, BarrierObservation.Discrete,
                             isCall: true, rebate: 0.00, cube).Price(market);
        var withRebate = BuildSabrBarrier(K, Alpha, Beta1, Rho, Nu, H_Up,
                             BarrierType.UpAndOut, BarrierObservation.Discrete,
                             isCall: true, rebate: 0.05, cube).Price(market);

        Assert.True(withRebate.Price > noRebate.Price,
            $"NoRebate={noRebate.Price:F6}  WithRebate={withRebate.Price:F6}");
    }

    // ── Input validation ─────────────────────────────────────────────────────

    [Fact]
    public void WrongOptionKind_Throws()
    {
        var cube      = BuildCube();
        var vanillaOpt = new Option
        {
            Underlying    = "EURUSD",
            Strike        = K,
            ExpiryYears   = T,
            OptionType    = OptionType.Call,
            ExerciseStyle = ExerciseStyle.European,
            Vanilla       = new VanillaOption()
        };
        Assert.Throws<ArgumentException>(() =>
            new FXSABRBarrierOptionMCPricer(vanillaOpt, ValuationDate, MakeSabr(Alpha, Beta1, Rho, Nu), cube));
    }

    [Fact]
    public void UnspecifiedBarrierType_Throws()
    {
        var cube = BuildCube();
        var opt  = MakeBarrierOption(K, T, isCall: true,
            barrierLevel: H_Up, barrierType: BarrierType.Unspecified,
            obs: BarrierObservation.Discrete, rebate: 0.0);
        Assert.Throws<ArgumentException>(() =>
            new FXSABRBarrierOptionMCPricer(opt, ValuationDate, MakeSabr(Alpha, Beta1, Rho, Nu), cube));
    }

    [Fact]
    public void UnspecifiedObservation_Throws()
    {
        var cube = BuildCube();
        var opt  = MakeBarrierOption(K, T, isCall: true,
            barrierLevel: H_Up, barrierType: BarrierType.UpAndOut,
            obs: BarrierObservation.Unspecified, rebate: 0.0);
        Assert.Throws<ArgumentException>(() =>
            new FXSABRBarrierOptionMCPricer(opt, ValuationDate, MakeSabr(Alpha, Beta1, Rho, Nu), cube));
    }

    [Fact]
    public void ZeroBarrierLevel_Throws()
    {
        var cube = BuildCube();
        var opt  = MakeBarrierOption(K, T, isCall: true,
            barrierLevel: 0.0, barrierType: BarrierType.UpAndOut,
            obs: BarrierObservation.Discrete, rebate: 0.0);
        Assert.Throws<ArgumentException>(() =>
            new FXSABRBarrierOptionMCPricer(opt, ValuationDate, MakeSabr(Alpha, Beta1, Rho, Nu), cube));
    }

    [Fact]
    public void SingleAssetCube_Throws()
    {
        var singleCube = SimulationCube.GenerateIndependent(100, Steps, assets: 1, DefaultSeed);
        var opt        = MakeBarrierOption(K, T, isCall: true,
            barrierLevel: H_Up, barrierType: BarrierType.UpAndOut,
            obs: BarrierObservation.Discrete, rebate: 0.0);
        Assert.Throws<ArgumentException>(() =>
            new FXSABRBarrierOptionMCPricer(opt, ValuationDate, MakeSabr(Alpha, Beta1, Rho, Nu), singleCube));
    }

    [Fact]
    public void ZeroAlpha_Throws()
    {
        var cube = BuildCube();
        var opt  = MakeBarrierOption(K, T, isCall: true,
            barrierLevel: H_Up, barrierType: BarrierType.UpAndOut,
            obs: BarrierObservation.Discrete, rebate: 0.0);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new FXSABRBarrierOptionMCPricer(opt, ValuationDate,
                MakeSabr(alpha: 0.0, Beta1, Rho, Nu), cube));
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private double Forward() => S0 * Math.Exp((Rd - Rf) * T);

    private FXSABRBarrierOptionMCPricer BuildSabrBarrier(
        double strike, double alpha, double beta, double rho, double nu,
        double barrierLevel, BarrierType barrierType, BarrierObservation obs,
        bool isCall, double rebate, SimulationCube cube)
    {
        var opt = MakeBarrierOption(strike, T, isCall, barrierLevel, barrierType, obs, rebate);
        return new FXSABRBarrierOptionMCPricer(opt, ValuationDate, MakeSabr(alpha, beta, rho, nu), cube);
    }

    private FXSABRVanillaOptionMCPricer BuildSabrVanilla(
        double strike, double alpha, double beta, double rho, double nu,
        bool isCall, SimulationCube cube)
    {
        var opt = new Option
        {
            Underlying    = "EURUSD",
            Strike        = strike,
            ExpiryYears   = T,
            OptionType    = isCall ? OptionType.Call : OptionType.Put,
            ExerciseStyle = ExerciseStyle.European,
            Vanilla       = new VanillaOption()
        };
        return new FXSABRVanillaOptionMCPricer(opt, ValuationDate, MakeSabr(alpha, beta, rho, nu), cube);
    }

    private static FXBarrierOptionMCPricer BuildGbmBarrier(
        double strike, double sigma, double barrierLevel,
        BarrierType barrierType, BarrierObservation obs,
        bool isCall, double rebate, SimulationCube cube)
    {
        var opt = MakeBarrierOption(strike, T, isCall, barrierLevel, barrierType, obs, rebate);
        return new FXBarrierOptionMCPricer(opt, ValuationDate, sigma, cube);
    }

    private static Option MakeBarrierOption(
        double strike, double expiry, bool isCall,
        double barrierLevel, BarrierType barrierType, BarrierObservation obs, double rebate)
    {
        return new Option
        {
            Underlying    = "EURUSD",
            Strike        = strike,
            ExpiryYears   = expiry,
            OptionType    = isCall ? OptionType.Call : OptionType.Put,
            ExerciseStyle = ExerciseStyle.European,
            Barrier       = new BarrierOption
            {
                BarrierLevel = barrierLevel,
                BarrierType  = barrierType,
                Observation  = obs,
                Rebate       = rebate
            }
        };
    }

    private FxMarketData MakeMarket() => MakeMarketWithRates(Rd, Rf);

    private FxMarketData MakeMarketWithRates(double rd, double rf) =>
        new()
        {
            CurrencyPair = "EURUSD",
            Spot         = S0,
            DomesticRate = ZeroCurve.Flat(rd, CurveMaturity),
            ForeignRate  = ZeroCurve.Flat(rf, CurveMaturity),
            VolSurface   = new VolatilitySurface()
        };

    private static SabrParameters MakeSabr(double alpha, double beta, double rho, double nu) =>
        new() { Alpha = alpha, Beta = beta, Rho = rho, Nu = nu };

    private static SimulationCube BuildCube() =>
        SimulationCube.GenerateIndependent(Paths, Steps, assets: 2, seed: DefaultSeed);
}
