using Aegis.Instruments;
using MCPricer.Bonds;
using NodaTime;
using RandomSimulator;

namespace MCPricer.Tests.Bonds;

/// <summary>
/// Validates CallableBondMCPricer (G2++ short-rate model + Longstaff-Schwartz
/// early-redemption regression) against:
///   - the model-independent bullet-bond benchmark (call schedule empty, or
///     call price set high enough the option is never exercised),
///   - the deterministic zero-vol limit (no simulation noise; the call
///     decision becomes a simple deterministic comparison),
///   - no-arbitrage bounds (a callable bond can never be worth more than the
///     equivalent bullet bond — the issuer's option only removes value from
///     the bondholder).
/// </summary>
public sealed class CallableBondMCPricerTests
{
    private static readonly LocalDate ValuationDate = new(2026, 1, 1);
    private static readonly LocalDate CurveMaturity  = new(2046, 1, 1);

    private static readonly G2ppParameters ModelParams =
        new(a1: 0.6, sigma1: 0.010, a2: 0.06, sigma2: 0.006, rho: -0.5);

    private const double Notional = 100.0;
    private const double CouponRate = 0.05;
    private const int CouponsPerYear = 2;
    private const int Paths = 40_000;
    private const int DefaultSeed = 42;

    private static readonly double[,] Correlation = { { 1.0, -0.5 }, { -0.5, 1.0 } };

    // ── Bullet-bond benchmark (no call optionality) ───────────────────────────

    [Fact]
    public void NoCallSchedule_MatchesAnalyticalBulletBondPrice_WithinMcBounds()
    {
        var maturity = ValuationDate.PlusYears(10);
        var bond = new CallableBondDefinition(Notional, CouponRate, CouponsPerYear, maturity);
        var curve = ZeroCurve.Flat(0.04, CurveMaturity);
        var cube = SimulationCube.Generate(Paths, steps: 26 * 10, assets: 2, Correlation, DefaultSeed);

        var mc = new CallableBondMCPricer(bond, ValuationDate, curve, ModelParams, cube).Price();
        var analytical = CallableBondMCPricer.AnalyticalBulletBondPrice(bond, ValuationDate, curve);

        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    [Fact]
    public void CallPriceNeverReached_MatchesAnalyticalBulletBondPrice_WithinMcBounds()
    {
        var maturity = ValuationDate.PlusYears(10);
        var callSchedule = new[]
        {
            new CallDate(ValuationDate.PlusYears(5), CallPrice: 5.0), // absurdly high: never optimal to call
            new CallDate(ValuationDate.PlusYears(7), CallPrice: 5.0),
        };
        var bond = new CallableBondDefinition(Notional, CouponRate, CouponsPerYear, maturity, callSchedule);
        var curve = ZeroCurve.Flat(0.04, CurveMaturity);
        var cube = SimulationCube.Generate(Paths, steps: 26 * 10, assets: 2, Correlation, DefaultSeed);

        var mc = new CallableBondMCPricer(bond, ValuationDate, curve, ModelParams, cube).Price();
        var analytical = CallableBondMCPricer.AnalyticalBulletBondPrice(bond, ValuationDate, curve);

        AssertWithinMcBounds(mc, analytical, sigma: 5.0);
    }

    // ── No-arbitrage bound ─────────────────────────────────────────────────────

    [Fact]
    public void CallableBond_NeverWorthMoreThanBulletBond()
    {
        var maturity = ValuationDate.PlusYears(10);
        var callSchedule = new[]
        {
            new CallDate(ValuationDate.PlusYears(3), CallPrice: 1.0),
            new CallDate(ValuationDate.PlusYears(5), CallPrice: 1.0),
            new CallDate(ValuationDate.PlusYears(7), CallPrice: 1.0),
        };
        var bond = new CallableBondDefinition(Notional, CouponRate, CouponsPerYear, maturity, callSchedule);
        var curve = ZeroCurve.Flat(0.04, CurveMaturity);
        var cube = SimulationCube.Generate(Paths, steps: 26 * 10, assets: 2, Correlation, DefaultSeed);

        var mc = new CallableBondMCPricer(bond, ValuationDate, curve, ModelParams, cube).Price();
        var analytical = CallableBondMCPricer.AnalyticalBulletBondPrice(bond, ValuationDate, curve);

        Assert.True(mc.Price < analytical + 5.0 * mc.StandardError,
            $"callable={mc.Price:F6} bullet={analytical:F6}");
    }

    // ── Zero-vol deterministic limit ──────────────────────────────────────────

    [Fact]
    public void ZeroVol_DeepItmCall_PricesAtDiscountedCallValue()
    {
        // With sigma1=sigma2=0 the short rate is deterministic (r(t)=phi(t),
        // matching the flat market curve exactly), so the "continuation value"
        // at the first call date is deterministic too. A call price of 0.5
        // (50% of par) is always below any plausible continuation value, so
        // the issuer calls at the very first opportunity on every path.
        var maturity = ValuationDate.PlusYears(10);
        var firstCallDate = ValuationDate.PlusYears(2);
        var callSchedule = new[] { new CallDate(firstCallDate, CallPrice: 0.5) };
        var bond = new CallableBondDefinition(Notional, CouponRate, CouponsPerYear, maturity, callSchedule);
        var curve = ZeroCurve.Flat(0.04, CurveMaturity);
        var zeroVolParams = new G2ppParameters(a1: 0.6, sigma1: 0.0, a2: 0.06, sigma2: 0.0, rho: -0.5);
        var cube = SimulationCube.Generate(Paths, steps: 26 * 10, assets: 2, Correlation, DefaultSeed);

        var mc = new CallableBondMCPricer(bond, ValuationDate, curve, zeroVolParams, cube).Price();

        // Expected value = coupons paid *before* the call date (received by the
        // bondholder unconditionally, regardless of the later call decision)
        // plus the call payoff (call price + that date's coupon) at the call date.
        var callYears = Period.Between(ValuationDate, firstCallDate, PeriodUnits.Days).Days / 365.0;
        var couponAtCall = Notional * CouponRate / CouponsPerYear;
        var expected = 0.0;
        foreach (var cf in bond.BuildCashflowSchedule(ValuationDate))
        {
            if (cf.YearsFromValuation < callYears - 1e-9)
                expected += cf.Amount * ZeroCurve.DiscountFactor(curve, ValuationDate, cf.YearsFromValuation);
        }
        expected += (0.5 * Notional + couponAtCall) * ZeroCurve.DiscountFactor(curve, ValuationDate, callYears);

        Assert.Equal(0.0, mc.StandardError, precision: 10); // deterministic: no path variance at all

        // Small residual tolerance: coupon dates are calendar-day-count year
        // fractions (Act/365) that generally do not land exactly on the
        // uniform simulation grid; each event snapshots the nearest grid node
        // (see BuildEventGrid), which is an intentional, documented source of
        // O(dt) discretization error, not statistical MC noise.
        Assert.True(Math.Abs(mc.Price - expected) < 0.01,
            $"MC={mc.Price:F6} expected={expected:F6} diff={Math.Abs(mc.Price - expected):F6}");
    }

    [Fact]
    public void ZeroVol_NoCallSchedule_MatchesAnalyticalBulletBondPrice()
    {
        var maturity = ValuationDate.PlusYears(6);
        var bond = new CallableBondDefinition(Notional, CouponRate, CouponsPerYear, maturity);
        var curve = ZeroCurve.Flat(0.035, CurveMaturity);
        var zeroVolParams = new G2ppParameters(a1: 0.6, sigma1: 0.0, a2: 0.06, sigma2: 0.0, rho: 0.0);
        // Daily-resolution grid: shrinks the calendar-day-count-vs-grid-node
        // rounding residual (see comment in ZeroVol_DeepItmCall test above)
        // to a fraction of a basis point.
        var cube = SimulationCube.Generate(1000, steps: 365 * 6, assets: 2, new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }, DefaultSeed);

        var mc = new CallableBondMCPricer(bond, ValuationDate, curve, zeroVolParams, cube).Price();
        var analytical = CallableBondMCPricer.AnalyticalBulletBondPrice(bond, ValuationDate, curve);

        Assert.True(Math.Abs(mc.Price - analytical) < 1e-3,
            $"MC={mc.Price:F6} analytical={analytical:F6} diff={Math.Abs(mc.Price - analytical):F6}");
    }

    // ── DV01 sign ──────────────────────────────────────────────────────────────

    [Fact]
    public void Dv01_IsNegative_ForFixedCouponBond()
    {
        // A fixed-coupon bond loses value when rates rise, regardless of the
        // embedded call option (which only ever reduces bondholder value).
        var maturity = ValuationDate.PlusYears(10);
        var bond = new CallableBondDefinition(Notional, CouponRate, CouponsPerYear, maturity);
        var curve = ZeroCurve.Flat(0.04, CurveMaturity);
        var cube = SimulationCube.Generate(Paths, steps: 26 * 10, assets: 2, Correlation, DefaultSeed);

        var pricer = new CallableBondMCPricer(bond, ValuationDate, curve, ModelParams, cube);
        var dv01 = pricer.ComputeDv01();

        Assert.True(dv01 < 0.0, $"DV01={dv01:F6} should be negative");
    }

    // ── Input validation ──────────────────────────────────────────────────────

    [Fact]
    public void Constructor_RejectsCubeWithWrongAssetCount()
    {
        var maturity = ValuationDate.PlusYears(5);
        var bond = new CallableBondDefinition(Notional, CouponRate, CouponsPerYear, maturity);
        var curve = ZeroCurve.Flat(0.04, CurveMaturity);
        var cube = SimulationCube.GenerateIndependent(1000, steps: 100, assets: 1, seed: DefaultSeed);

        Assert.Throws<ArgumentException>(() => new CallableBondMCPricer(bond, ValuationDate, curve, ModelParams, cube));
    }

    [Fact]
    public void Constructor_RejectsCorrelationMismatch()
    {
        var maturity = ValuationDate.PlusYears(5);
        var bond = new CallableBondDefinition(Notional, CouponRate, CouponsPerYear, maturity);
        var curve = ZeroCurve.Flat(0.04, CurveMaturity);
        var mismatchedCorrelation = new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }; // model Rho = -0.5
        var cube = SimulationCube.Generate(1000, steps: 100, assets: 2, mismatchedCorrelation, DefaultSeed);

        Assert.Throws<ArgumentException>(() => new CallableBondMCPricer(bond, ValuationDate, curve, ModelParams, cube));
    }

    [Fact]
    public void Constructor_RejectsMaturityBeforeValuationDate()
    {
        var bond = new CallableBondDefinition(Notional, CouponRate, CouponsPerYear, ValuationDate.PlusYears(-1));
        var curve = ZeroCurve.Flat(0.04, CurveMaturity);
        var cube = SimulationCube.Generate(1000, steps: 100, assets: 2, Correlation, DefaultSeed);

        Assert.Throws<ArgumentException>(() => new CallableBondMCPricer(bond, ValuationDate, curve, ModelParams, cube));
    }

    [Fact]
    public void CallableBondDefinition_RejectsCallDateOnOrAfterMaturity()
    {
        var maturity = ValuationDate.PlusYears(5);
        Assert.Throws<ArgumentException>(() =>
            new CallableBondDefinition(Notional, CouponRate, CouponsPerYear, maturity,
                new[] { new CallDate(maturity, CallPrice: 1.0) }));
    }

    [Fact]
    public void CallableBondDefinition_RejectsNonDivisorCouponFrequency()
    {
        Assert.Throws<ArgumentException>(() =>
            new CallableBondDefinition(Notional, CouponRate, couponsPerYear: 5, ValuationDate.PlusYears(5)));
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private static void AssertWithinMcBounds(PricingResult mc, double analytical, double sigma)
    {
        var tolerance = sigma * mc.StandardError;
        Assert.True(Math.Abs(mc.Price - analytical) < tolerance,
            $"MC={mc.Price:F6}  Analytical={analytical:F6}  Diff={Math.Abs(mc.Price - analytical):F6}  " +
            $"{sigma}σ tol={tolerance:F6}  SE={mc.StandardError:F6}");
    }
}
