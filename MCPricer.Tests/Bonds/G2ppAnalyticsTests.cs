using Aegis.Instruments;
using MCPricer.Bonds;
using NodaTime;

namespace MCPricer.Tests.Bonds;

/// <summary>
/// Validates the closed-form G2++ (two-factor Hull-White) formulas:
/// φ(t), the zero-coupon bond price P(t,T), and the zero-coupon bond option
/// (ZBO) price — against the model's defining property (exact curve
/// reproduction) and against deterministic/deep-ITM/deep-OTM limits.
/// </summary>
public sealed class G2ppAnalyticsTests
{
    private static readonly LocalDate ValuationDate = new(2026, 1, 1);
    private static readonly LocalDate CurveMaturity  = new(2056, 1, 1);

    private static readonly G2ppParameters DefaultParams =
        new(a1: 0.6, sigma1: 0.012, a2: 0.08, sigma2: 0.008, rho: -0.6);

    // ── P(0,T) must reproduce the input curve exactly ─────────────────────────

    [Theory]
    [InlineData(0.5)]
    [InlineData(1.0)]
    [InlineData(5.0)]
    [InlineData(10.0)]
    [InlineData(29.9)]
    public void ZeroCouponBondPrice_AtT0_MatchesMarketCurveExactly(double maturity)
    {
        var curve = ZeroCurve.Flat(0.04, CurveMaturity);

        var modelPrice  = G2ppAnalytics.ZeroCouponBondPrice(curve, ValuationDate, DefaultParams, t: 0.0, T: maturity, x: 0.0, y: 0.0);
        var marketPrice = ZeroCurve.DiscountFactor(curve, ValuationDate, maturity);

        Assert.Equal(marketPrice, modelPrice, precision: 12);
    }

    [Fact]
    public void ZeroCouponBondPrice_AtT0_MatchesCurve_RegardlessOfParameters()
    {
        var curve = ZeroCurve.Flat(0.03, CurveMaturity);
        var paramSets = new[]
        {
            new G2ppParameters(0.1, 0.001, 0.1, 0.001, 0.0),
            new G2ppParameters(2.0, 0.05,  0.02, 0.02, 0.9),
            new G2ppParameters(0.5, 0.02,  0.5,  0.02, -0.9),
        };

        foreach (var p in paramSets)
        {
            var modelPrice = G2ppAnalytics.ZeroCouponBondPrice(curve, ValuationDate, p, 0.0, 7.0, 0.0, 0.0);
            var marketPrice = ZeroCurve.DiscountFactor(curve, ValuationDate, 7.0);
            Assert.Equal(marketPrice, modelPrice, precision: 10);
        }
    }

    [Fact]
    public void ZeroCouponBondPrice_SameMaturityAsValuation_IsOne()
    {
        var curve = ZeroCurve.Flat(0.04, CurveMaturity);
        var price = G2ppAnalytics.ZeroCouponBondPrice(curve, ValuationDate, DefaultParams, t: 3.0, T: 3.0, x: 0.1, y: -0.05);
        Assert.Equal(1.0, price, precision: 12);
    }

    [Fact]
    public void ZeroCouponBondPrice_PositiveStateShiftsDownwardBondPrice()
    {
        // Higher realized short rate (positive x, y) must lower the bond price:
        // dP/dx = -B(A1,tau)*P < 0 since B(a,tau) > 0 for a,tau > 0.
        var curve = ZeroCurve.Flat(0.04, CurveMaturity);
        var baseline = G2ppAnalytics.ZeroCouponBondPrice(curve, ValuationDate, DefaultParams, 1.0, 10.0, 0.0, 0.0);
        var shifted  = G2ppAnalytics.ZeroCouponBondPrice(curve, ValuationDate, DefaultParams, 1.0, 10.0, 0.02, 0.02);

        Assert.True(shifted < baseline, $"shifted={shifted:F6} should be < baseline={baseline:F6}");
    }

    [Fact]
    public void ZeroCouponBondPrice_MonotonicallyDecreasingInMaturity()
    {
        var curve = ZeroCurve.Flat(0.04, CurveMaturity);
        var p5  = G2ppAnalytics.ZeroCouponBondPrice(curve, ValuationDate, DefaultParams, 0.0, 5.0,  0.0, 0.0);
        var p10 = G2ppAnalytics.ZeroCouponBondPrice(curve, ValuationDate, DefaultParams, 0.0, 10.0, 0.0, 0.0);
        var p20 = G2ppAnalytics.ZeroCouponBondPrice(curve, ValuationDate, DefaultParams, 0.0, 20.0, 0.0, 0.0);

        Assert.True(p5 > p10);
        Assert.True(p10 > p20);
    }

    // ── Zero-coupon bond option: zero-vol limit reduces to discounted intrinsic ─

    [Fact]
    public void ZboCall_NearZeroVol_MatchesDiscountedIntrinsic()
    {
        var curve = ZeroCurve.Flat(0.03, CurveMaturity);
        var tinyVolParams = new G2ppParameters(a1: 0.5, sigma1: 1e-9, a2: 0.05, sigma2: 1e-9, rho: 0.0);

        const double t = 2.0, T = 5.0, notional = 100.0;
        var forward = notional * ZeroCurve.DiscountFactor(curve, ValuationDate, T)
                                / ZeroCurve.DiscountFactor(curve, ValuationDate, t);

        foreach (var strike in new[] { forward * 0.8, forward, forward * 1.2 })
        {
            var callPrice = G2ppAnalytics.ZeroCouponBondOption(curve, ValuationDate, tinyVolParams, t, T, strike, notional, isCall: true);
            var expected  = ZeroCurve.DiscountFactor(curve, ValuationDate, t) * Math.Max(forward - strike, 0.0);
            Assert.Equal(expected, callPrice, precision: 6);
        }
    }

    [Fact]
    public void Zbo_DeepItmCall_ApproachesDiscountedForwardMinusStrike()
    {
        var curve = ZeroCurve.Flat(0.03, CurveMaturity);
        const double t = 1.0, T = 3.0, notional = 100.0;
        var forward = notional * ZeroCurve.DiscountFactor(curve, ValuationDate, T)
                                / ZeroCurve.DiscountFactor(curve, ValuationDate, t);
        var deepItmStrike = forward * 0.3;

        var callPrice = G2ppAnalytics.ZeroCouponBondOption(curve, ValuationDate, DefaultParams, t, T, deepItmStrike, notional, isCall: true);
        var intrinsic = ZeroCurve.DiscountFactor(curve, ValuationDate, t) * (forward - deepItmStrike);

        Assert.True(callPrice > intrinsic * 0.99, $"call={callPrice:F6} intrinsic={intrinsic:F6}");
    }

    [Fact]
    public void Zbo_DeepOtmCall_IsNearZero()
    {
        var curve = ZeroCurve.Flat(0.03, CurveMaturity);
        const double t = 1.0, T = 3.0, notional = 100.0;
        var forward = notional * ZeroCurve.DiscountFactor(curve, ValuationDate, T)
                                / ZeroCurve.DiscountFactor(curve, ValuationDate, t);
        var deepOtmStrike = forward * 3.0;

        var callPrice = G2ppAnalytics.ZeroCouponBondOption(curve, ValuationDate, DefaultParams, t, T, deepOtmStrike, notional, isCall: true);
        Assert.True(callPrice < 0.01 * notional, $"call={callPrice:F6}");
    }

    [Fact]
    public void Zbo_PutCallParity_HoldsExactly()
    {
        // Model-free identity: Call - Put = N*P(0,T) - X*P(0,t).
        var curve = ZeroCurve.Flat(0.035, CurveMaturity);
        const double t = 1.5, T = 6.0, strike = 90.0, notional = 100.0;

        var call = G2ppAnalytics.ZeroCouponBondOption(curve, ValuationDate, DefaultParams, t, T, strike, notional, isCall: true);
        var put  = G2ppAnalytics.ZeroCouponBondOption(curve, ValuationDate, DefaultParams, t, T, strike, notional, isCall: false);

        var parity = notional * ZeroCurve.DiscountFactor(curve, ValuationDate, T)
                   - strike   * ZeroCurve.DiscountFactor(curve, ValuationDate, t);

        Assert.Equal(parity, call - put, precision: 9);
    }

    // ── Input validation ──────────────────────────────────────────────────────

    [Fact]
    public void Constructor_RejectsNonPositiveMeanReversion()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new G2ppParameters(0.0, 0.01, 0.1, 0.01, 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new G2ppParameters(0.1, 0.01, -0.1, 0.01, 0.0));
    }

    [Fact]
    public void Constructor_RejectsOutOfRangeCorrelation()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new G2ppParameters(0.1, 0.01, 0.1, 0.01, 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new G2ppParameters(0.1, 0.01, 0.1, 0.01, -1.0));
    }

    [Fact]
    public void ZeroCouponBondOption_RejectsNonPositiveExpiry()
    {
        var curve = ZeroCurve.Flat(0.03, CurveMaturity);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            G2ppAnalytics.ZeroCouponBondOption(curve, ValuationDate, DefaultParams, 0.0, 5.0, 90.0, 100.0, true));
    }
}
