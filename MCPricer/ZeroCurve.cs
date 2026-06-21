using Aegis.Instruments;
using NodaTime;
using NodaTime.Text;

namespace MCPricer;

/// <summary>
/// Interpolates continuously-compounded zero rates from a <see cref="Pillars"/>
/// curve, and applies parallel shifts (the standard bump used for rho-style
/// bump-and-revalue Greeks).
///
/// ── Convention ────────────────────────────────────────────────────────────────
/// Each <see cref="Pillar"/> carries an absolute ISO-8601 maturity date. Its
/// year-fraction from the valuation date is computed Act/365 fixed:
///   τ_i = (PillarDate_i − valuationDate).Days / 365.0
///
/// The zero rate at an arbitrary maturity τ ≥ 0 is obtained by linear
/// interpolation in (τ, rate) space — the standard "linear in the zero rate"
/// convention — with flat extrapolation beyond the first/last pillar:
///   τ ≤ τ_0      → r(τ) = r_0
///   τ ≥ τ_{n-1}  → r(τ) = r_{n-1}
///   otherwise    → linear interpolation between the bracketing pillars
///
/// Pillars must be supplied in strictly increasing date order — this is
/// validated aggressively (no silent reordering / no silent fallback).
/// </summary>
public static class ZeroCurve
{
    private static readonly LocalDatePattern IsoPattern = LocalDatePattern.Iso;

    /// <summary>
    /// Interpolates the continuously-compounded zero rate r(T) implied by
    /// <paramref name="curve"/> at maturity <paramref name="maturityYears"/>,
    /// measured in year-fractions (Act/365) from <paramref name="valuationDate"/>.
    /// </summary>
    public static double InterpolateRate(Pillars curve, LocalDate valuationDate, double maturityYears)
    {
        ArgumentNullException.ThrowIfNull(curve);
        if (curve.Pillar.Count == 0)
            throw new ArgumentException("Curve must contain at least one pillar.", nameof(curve));
        if (maturityYears < 0)
            throw new ArgumentOutOfRangeException(nameof(maturityYears), "Maturity must be ≥ 0.");

        var n = curve.Pillar.Count;
        var tenors = new double[n];
        var rates  = new double[n];

        for (var i = 0; i < n; i++)
        {
            var pillar      = curve.Pillar[i];
            var parseResult = IsoPattern.Parse(pillar.Date);
            if (!parseResult.Success)
                throw new ArgumentException(
                    $"Pillar date '{pillar.Date}' is not a valid ISO-8601 date.", nameof(curve));

            var date   = parseResult.Value;
            tenors[i]  = Period.Between(valuationDate, date, PeriodUnits.Days).Days / 365.0;
            rates[i]   = pillar.Value;

            if (i > 0 && tenors[i] <= tenors[i - 1])
                throw new ArgumentException(
                    "Curve pillars must be strictly increasing in maturity date " +
                    $"(pillar {i - 1} → {i} is not increasing: {curve.Pillar[i - 1].Date} → {pillar.Date}).",
                    nameof(curve));
        }

        if (maturityYears <= tenors[0])
            return rates[0];
        if (maturityYears >= tenors[^1])
            return rates[^1];

        for (var i = 1; i < n; i++)
        {
            if (maturityYears > tenors[i]) continue;

            var t0 = tenors[i - 1];
            var t1 = tenors[i];
            var w  = (maturityYears - t0) / (t1 - t0);
            return rates[i - 1] + w * (rates[i] - rates[i - 1]);
        }

        // Unreachable: maturityYears < tenors[^1] guarantees the loop returns.
        throw new InvalidOperationException("Failed to bracket maturity within the curve.");
    }

    /// <summary>
    /// Returns a new curve with every pillar's zero rate shifted by
    /// <paramref name="amount"/> (a parallel shift) — the standard bump used
    /// to compute rate-sensitivity Greeks (rho) by bump-and-revalue.
    /// </summary>
    public static Pillars Shift(Pillars curve, double amount)
    {
        ArgumentNullException.ThrowIfNull(curve);

        var shifted = new Pillars();
        foreach (var pillar in curve.Pillar)
            shifted.Pillar.Add(new Pillar { Date = pillar.Date, Value = pillar.Value + amount });

        return shifted;
    }

    /// <summary>
    /// Returns the continuously-compounded instantaneous forward zero rate
    /// f(t0, t1) implied by the curve between year-fractions t0 and t1:
    ///   f(t0, t1) = (r(t1)·t1 − r(t0)·t0) / (t1 − t0)
    ///
    /// Exact shortcuts preserve bit-for-bit results with flat curves:
    ///   t0 == 0     → f = r(t1)           (no t0 interpolation needed)
    ///   r(t0)==r(t1) → f = r(t0)          (flat segment; avoids ULP drift)
    ///   t0 == t1    → f = r(t1)           (degenerate interval)
    /// </summary>
    public static double ForwardRate(Pillars curve, LocalDate valuationDate, double t0, double t1)
    {
        if (t0 >= t1)
            return InterpolateRate(curve, valuationDate, t1);
        var r1 = InterpolateRate(curve, valuationDate, t1);
        if (t0 == 0.0)
            return r1;
        var r0 = InterpolateRate(curve, valuationDate, t0);
        if (r0 == r1)
            return r0;
        return (r1 * t1 - r0 * t0) / (t1 - t0);
    }

    /// <summary>
    /// Discount factor P(0, T) = exp(−r(T)·T), where r(T) is the interpolated
    /// zero rate from the curve at maturity T.
    /// </summary>
    public static double DiscountFactor(Pillars curve, LocalDate valuationDate, double maturityYears)
        => Math.Exp(-InterpolateRate(curve, valuationDate, maturityYears) * maturityYears);

    /// <summary>
    /// Builds a flat single-pillar curve r(τ) ≡ <paramref name="rate"/> for all
    /// maturities — convenient for constructing market data from a single quoted
    /// rate (the common case when no term structure is available).
    /// </summary>
    public static Pillars Flat(double rate, LocalDate maturity)
    {
        var curve = new Pillars();
        curve.Pillar.Add(new Pillar { Date = maturity.ToString("yyyy-MM-dd", null), Value = rate });
        return curve;
    }
}
