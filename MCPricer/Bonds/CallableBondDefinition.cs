using NodaTime;

namespace MCPricer.Bonds;

/// <summary>
/// One date on which the issuer may redeem the bond early.
/// CallPrice is expressed as a fraction of notional (1.0 = par; 1.02 = 102% of
/// par, a typical make-whole-lite premium).
/// </summary>
public sealed record CallDate(LocalDate Date, double CallPrice)
{
    public double CallPrice { get; } = CallPrice > 0.0
        ? CallPrice
        : throw new ArgumentOutOfRangeException(nameof(CallPrice), "CallPrice must be > 0.");
}

/// <summary>
/// Fixed-rate callable bond: a bullet bond (periodic coupons + principal at
/// maturity) plus a schedule of dates on which the issuer holds the right to
/// redeem early at a specified price. An empty <see cref="CallSchedule"/>
/// reduces this to a plain bullet bond.
///
/// Product definition only — no market data, no pricing model. Coupon-date
/// generation (<see cref="BuildCashflowSchedule"/>) is pure calendar
/// arithmetic; it does not depend on the discount curve or the short-rate model.
/// </summary>
public sealed class CallableBondDefinition
{
    public double Notional { get; }
    public double CouponRate { get; }
    public int CouponsPerYear { get; }
    public LocalDate MaturityDate { get; }
    public IReadOnlyList<CallDate> CallSchedule { get; }

    public CallableBondDefinition(
        double notional,
        double couponRate,
        int couponsPerYear,
        LocalDate maturityDate,
        IReadOnlyList<CallDate>? callSchedule = null)
    {
        if (notional <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(notional), "Notional must be > 0.");
        if (couponRate < 0.0)
            throw new ArgumentOutOfRangeException(nameof(couponRate), "CouponRate must be >= 0.");
        if (couponsPerYear <= 0 || 12 % couponsPerYear != 0)
            throw new ArgumentException(
                "CouponsPerYear must divide 12 evenly (1, 2, 3, 4, 6, or 12).", nameof(couponsPerYear));

        Notional       = notional;
        CouponRate     = couponRate;
        CouponsPerYear = couponsPerYear;
        MaturityDate   = maturityDate;

        var schedule = (callSchedule ?? Array.Empty<CallDate>()).OrderBy(c => c.Date).ToArray();
        for (var i = 0; i < schedule.Length; i++)
        {
            if (schedule[i].Date >= maturityDate)
                throw new ArgumentException(
                    $"Call date {schedule[i].Date} must be strictly before the maturity date {maturityDate}.",
                    nameof(callSchedule));
            if (i > 0 && schedule[i].Date == schedule[i - 1].Date)
                throw new ArgumentException("CallSchedule must not contain duplicate dates.", nameof(callSchedule));
        }
        CallSchedule = schedule;
    }

    /// <summary>
    /// Generates the coupon cashflow schedule as seen from <paramref name="valuationDate"/>:
    /// dates stepped back from <see cref="MaturityDate"/> in
    /// 12/<see cref="CouponsPerYear"/>-month increments, keeping only payment
    /// dates strictly after the valuation date. The final entry carries both
    /// the last coupon and the principal redemption.
    ///
    /// Year fractions use Act/365, matching <see cref="ZeroCurve"/>'s convention.
    /// </summary>
    public IReadOnlyList<BondCashflow> BuildCashflowSchedule(LocalDate valuationDate)
    {
        if (MaturityDate <= valuationDate)
            throw new ArgumentException("MaturityDate must be after valuationDate.", nameof(valuationDate));

        var step = Period.FromMonths(12 / CouponsPerYear);
        var couponAmount = Notional * CouponRate / CouponsPerYear;

        var dates = new List<LocalDate>();
        var d = MaturityDate;
        while (d > valuationDate)
        {
            dates.Add(d);
            d -= step;
        }
        dates.Reverse();

        var result = new List<BondCashflow>(dates.Count);
        foreach (var date in dates)
        {
            var years = Period.Between(valuationDate, date, PeriodUnits.Days).Days / 365.0;
            var cash  = couponAmount + (date == MaturityDate ? Notional : 0.0);
            result.Add(new BondCashflow(date, years, cash));
        }
        return result;
    }
}

/// <summary>One scheduled cashflow: coupon, or coupon + principal at maturity.</summary>
public sealed record BondCashflow(LocalDate Date, double YearsFromValuation, double Amount);
