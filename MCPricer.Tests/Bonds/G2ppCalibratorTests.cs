using Aegis.Instruments;
using MCPricer.Bonds;
using NodaTime;

namespace MCPricer.Tests.Bonds;

/// <summary>
/// Validates G2++ calibration by round-tripping: generate synthetic
/// zero-coupon-bond-option "market" prices from a known parameter set, then
/// verify the calibrator recovers parameters that reprice those same
/// instruments to within tight tolerance.
///
/// This is the correct sanity check for a nonlinear least-squares calibration
/// — checking that the *fitted model reprices the calibration instruments*,
/// not that the raw parameter vector matches bit-for-bit (least-squares fits
/// to a handful of instruments are not guaranteed to be unique).
/// </summary>
public sealed class G2ppCalibratorTests
{
    private static readonly LocalDate ValuationDate = new(2026, 1, 1);
    private static readonly LocalDate CurveMaturity  = new(2056, 1, 1);

    private static readonly G2ppParameters TrueParams =
        new(a1: 0.6, sigma1: 0.010, a2: 0.06, sigma2: 0.006, rho: -0.5);

    [Fact]
    public void Calibrate_RecoversInstrumentPrices_WithinTightTolerance()
    {
        var curve = ZeroCurve.Flat(0.035, CurveMaturity);
        var instruments = BuildSyntheticInstruments(curve, TrueParams);

        var calibrated = G2ppCalibrator.Calibrate(curve, ValuationDate, instruments);

        foreach (var inst in instruments)
        {
            var repriced = G2ppAnalytics.ZeroCouponBondOption(
                curve, ValuationDate, calibrated, inst.OptionExpiry, inst.BondMaturity, inst.Strike, inst.Notional, inst.IsCall);
            var relError = Math.Abs(repriced - inst.MarketPrice) / inst.MarketPrice;
            Assert.True(relError < 1e-3,
                $"expiry={inst.OptionExpiry} maturity={inst.BondMaturity}: repriced={repriced:F6} " +
                $"market={inst.MarketPrice:F6} relError={relError:E3}");
        }
    }

    [Fact]
    public void Calibrate_ProducesValidParameters()
    {
        var curve = ZeroCurve.Flat(0.03, CurveMaturity);
        var instruments = BuildSyntheticInstruments(curve, TrueParams);

        var calibrated = G2ppCalibrator.Calibrate(curve, ValuationDate, instruments);

        Assert.True(calibrated.A1 > 0.0);
        Assert.True(calibrated.A2 > 0.0);
        Assert.True(calibrated.Sigma1 >= 0.0);
        Assert.True(calibrated.Sigma2 >= 0.0);
        Assert.InRange(calibrated.Rho, -1.0, 1.0);
    }

    [Fact]
    public void Calibrate_TooFewInstruments_Throws()
    {
        var curve = ZeroCurve.Flat(0.03, CurveMaturity);
        var instruments = BuildSyntheticInstruments(curve, TrueParams).Take(3).ToList();

        Assert.Throws<ArgumentException>(() => G2ppCalibrator.Calibrate(curve, ValuationDate, instruments));
    }

    [Fact]
    public void Calibrate_NullCurve_Throws()
    {
        var instruments = BuildSyntheticInstruments(ZeroCurve.Flat(0.03, CurveMaturity), TrueParams);
        Assert.Throws<ArgumentNullException>(() => G2ppCalibrator.Calibrate(null!, ValuationDate, instruments));
    }

    [Fact]
    public void CalibrationInstrument_RejectsInvertedExpiryMaturity()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new G2ppCalibrationInstrument(OptionExpiry: 5.0, BondMaturity: 3.0, Strike: 90.0, Notional: 100.0, IsCall: true, MarketPrice: 1.0));
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private static List<G2ppCalibrationInstrument> BuildSyntheticInstruments(Pillars curve, G2ppParameters trueParams)
    {
        // A small "cap-like" grid of options on zero-coupon bonds spanning
        // short and long expiries/tenors, priced under the true parameters.
        var grid = new (double Expiry, double Maturity, double MoneynessOfForward)[]
        {
            (1.0, 2.0,  1.00),
            (1.0, 5.0,  1.00),
            (2.0, 5.0,  0.95),
            (3.0, 8.0,  1.05),
            (5.0, 10.0, 1.00),
            (5.0, 15.0, 0.90),
            (7.0, 20.0, 1.10),
        };

        var instruments = new List<G2ppCalibrationInstrument>();
        foreach (var (expiry, maturity, moneyness) in grid)
        {
            const double notional = 100.0;
            var forward = notional * ZeroCurve.DiscountFactor(curve, ValuationDate, maturity)
                                    / ZeroCurve.DiscountFactor(curve, ValuationDate, expiry);
            var strike = forward * moneyness;

            var price = G2ppAnalytics.ZeroCouponBondOption(curve, ValuationDate, trueParams, expiry, maturity, strike, notional, isCall: true);
            instruments.Add(new G2ppCalibrationInstrument(expiry, maturity, strike, notional, IsCall: true, MarketPrice: price));
        }
        return instruments;
    }
}
