using Aegis.Instruments;
using AnalyticalPricers;
using NodaTime;

namespace MCPricer.Bonds;

/// <summary>
/// Closed-form G2++ (two-factor Hull-White) formulas: the deterministic drift
/// φ(t) that fits the initial market curve, the zero-coupon bond price P(t,T),
/// and the European option on a zero-coupon bond (Brigo &amp; Mercurio, 2006,
/// §4.2, eqs. 4.10 and 4.14) — used both as the calibration pricing formula and
/// as the analytical benchmark for the Monte Carlo callable-bond pricer.
///
/// All time arguments are year-fractions (Act/365) from the valuation date
/// (consistent with <see cref="ZeroCurve"/>).
/// </summary>
public static class G2ppAnalytics
{
    /// <summary>
    /// Finite-difference estimate of the market instantaneous forward rate
    /// f^M(0,t) = -d/dt ln P^M(0,t), obtained from <see cref="ZeroCurve.ForwardRate"/>
    /// over a narrow window [t-h, t+h] (one-sided [0, h] at t=0).
    ///
    /// ZeroCurve only exposes rates at discrete/interpolated points, so the
    /// instantaneous forward is approximated rather than differentiated in
    /// closed form. h = 1e-4 years (~53 minutes) is far smaller than any
    /// realistic curve pillar spacing, so the approximation error is negligible
    /// relative to typical curve interpolation error.
    /// </summary>
    public static double InstantaneousForward(
        Pillars curve, LocalDate valuationDate, double t, double h = 1e-4)
    {
        if (t < 0.0)
            throw new ArgumentOutOfRangeException(nameof(t), "t must be >= 0.");

        return t <= h
            ? ZeroCurve.ForwardRate(curve, valuationDate, 0.0, h)
            : ZeroCurve.ForwardRate(curve, valuationDate, t - h, t + h);
    }

    /// <summary>
    /// φ(t) = f^M(0,t) + Σ1²/(2A1²)·(1-e^(-A1 t))² + Σ2²/(2A2²)·(1-e^(-A2 t))²
    ///        + Rho·Σ1·Σ2/(A1·A2)·(1-e^(-A1 t))·(1-e^(-A2 t))
    ///
    /// The unique deterministic shift that makes E[r(t)] under the model equal
    /// the market instantaneous forward rate, so that P(0,T) reproduces the
    /// input curve exactly for every T (Brigo &amp; Mercurio, eq. 4.10).
    /// </summary>
    public static double Phi(Pillars curve, LocalDate valuationDate, G2ppParameters p, double t)
    {
        var f = InstantaneousForward(curve, valuationDate, t);

        var oneMinusEa = 1.0 - Math.Exp(-p.A1 * t);
        var oneMinusEb = 1.0 - Math.Exp(-p.A2 * t);

        var termA = p.Sigma1 * p.Sigma1 / (2.0 * p.A1 * p.A1) * oneMinusEa * oneMinusEa;
        var termB = p.Sigma2 * p.Sigma2 / (2.0 * p.A2 * p.A2) * oneMinusEb * oneMinusEb;
        var termC = p.Rho * p.Sigma1 * p.Sigma2 / (p.A1 * p.A2) * oneMinusEa * oneMinusEb;

        return f + termA + termB + termC;
    }

    /// <summary>
    /// V(τ) — the time-homogeneous variance kernel used by both P(t,T) and the
    /// zero-coupon bond option formula (Brigo &amp; Mercurio, eq. 4.11). Depends
    /// only on the horizon τ = T-t because x(0) = y(0) = 0 and the OU
    /// coefficients are constant.
    /// </summary>
    private static double VFunction(double tau, G2ppParameters p)
    {
        var a = p.A1; var sigma = p.Sigma1;
        var b = p.A2; var eta   = p.Sigma2;
        var rho = p.Rho;

        var termA = sigma * sigma / (a * a) *
                    (tau + 2.0 / a * Math.Exp(-a * tau) - 1.0 / (2.0 * a) * Math.Exp(-2.0 * a * tau) - 3.0 / (2.0 * a));
        var termB = eta * eta / (b * b) *
                    (tau + 2.0 / b * Math.Exp(-b * tau) - 1.0 / (2.0 * b) * Math.Exp(-2.0 * b * tau) - 3.0 / (2.0 * b));
        var termC = 2.0 * rho * sigma * eta / (a * b) *
                    (tau + (Math.Exp(-a * tau) - 1.0) / a + (Math.Exp(-b * tau) - 1.0) / b
                         - (Math.Exp(-(a + b) * tau) - 1.0) / (a + b));

        return termA + termB + termC;
    }

    /// <summary>
    /// Closed-form zero-coupon bond price under G2++, conditional on the state
    /// (x(t), y(t)):
    ///
    ///   P(t,T) = [P^M(0,T)/P^M(0,t)] · exp{ ½·(V(T-t) - V(T) + V(t)) - B(A1,T-t)·x - B(A2,T-t)·y }
    ///
    /// where B(a,τ) = (1-e^(-aτ))/a (Brigo &amp; Mercurio, eq. 4.10).
    ///
    /// Sanity check: at t=0, x=y=0, this collapses to P(0,T) = P^M(0,T) — the
    /// model reproduces the input curve exactly regardless of the five model
    /// parameters, which is the defining property of φ(t).
    /// </summary>
    public static double ZeroCouponBondPrice(
        Pillars curve, LocalDate valuationDate, G2ppParameters p,
        double t, double T, double x, double y)
    {
        if (t < 0.0)
            throw new ArgumentOutOfRangeException(nameof(t), "t must be >= 0.");
        if (T < t)
            throw new ArgumentOutOfRangeException(nameof(T), "T must be >= t.");
        if (T == t)
            return 1.0;

        var pmT = ZeroCurve.DiscountFactor(curve, valuationDate, T);
        var pmT0 = t == 0.0 ? 1.0 : ZeroCurve.DiscountFactor(curve, valuationDate, t);

        var tau = T - t;
        var vTau = VFunction(tau, p);
        var vT   = VFunction(T, p);
        var vT0  = t == 0.0 ? 0.0 : VFunction(t, p);

        var bA = (1.0 - Math.Exp(-p.A1 * tau)) / p.A1;
        var bB = (1.0 - Math.Exp(-p.A2 * tau)) / p.A2;

        var exponent = 0.5 * (vTau - vT + vT0) - bA * x - bB * y;
        return pmT / pmT0 * Math.Exp(exponent);
    }

    /// <summary>
    /// Closed-form price of a European call/put with expiry t on a zero-coupon
    /// bond maturing at T, strike X, notional N (Brigo &amp; Mercurio, eq. 4.14).
    ///
    /// Structurally identical to Black-76 on the forward bond price
    /// F = N·P^M(0,T)/P^M(0,t) with total (non-annualised) volatility Σ(t,T):
    ///
    ///   Σ² = Σ1²/(2A1³)·(1-e^(-A1(T-t)))²·(1-e^(-2A1 t))
    ///      + Σ2²/(2A2³)·(1-e^(-A2(T-t)))²·(1-e^(-2A2 t))
    ///      + 2·Rho·Σ1·Σ2/(A1·A2·(A1+A2))·(1-e^(-A1(T-t)))·(1-e^(-A2(T-t)))·(1-e^(-(A1+A2)t))
    ///
    ///   Call = N·P^M(0,T)·Φ(h1) - X·P^M(0,t)·Φ(h2)
    ///   Put  = X·P^M(0,t)·Φ(-h2) - N·P^M(0,T)·Φ(-h1)
    ///   h1 = ln(N·P^M(0,T)/(X·P^M(0,t)))/Σ + Σ/2,   h2 = h1 - Σ
    ///
    /// Used as the natural calibration instrument for the G2++ model — caps,
    /// floors and swaptions all reduce to portfolios of options of this type.
    ///
    /// Edge case: Σ → 0 (zero vol, or t → 0) collapses to the discounted
    /// intrinsic value of the deterministic forward bond price.
    /// </summary>
    public static double ZeroCouponBondOption(
        Pillars curve, LocalDate valuationDate, G2ppParameters p,
        double optionExpiry, double bondMaturity, double strike, double notional, bool isCall)
    {
        if (optionExpiry <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(optionExpiry), "Option expiry must be > 0.");
        if (bondMaturity <= optionExpiry)
            throw new ArgumentOutOfRangeException(nameof(bondMaturity), "Bond maturity must be > option expiry.");
        if (strike <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(strike), "Strike must be > 0.");
        if (notional <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(notional), "Notional must be > 0.");

        var t = optionExpiry;
        var T = bondMaturity;
        var a = p.A1; var sigma = p.Sigma1;
        var b = p.A2; var eta   = p.Sigma2;
        var rho = p.Rho;
        var tau = T - t;

        var oneMinusEaTau = 1.0 - Math.Exp(-a * tau);
        var oneMinusEbTau = 1.0 - Math.Exp(-b * tau);

        var sigmaSq = sigma * sigma / (2.0 * a * a * a) * oneMinusEaTau * oneMinusEaTau * (1.0 - Math.Exp(-2.0 * a * t))
                    + eta * eta / (2.0 * b * b * b) * oneMinusEbTau * oneMinusEbTau * (1.0 - Math.Exp(-2.0 * b * t))
                    + 2.0 * rho * sigma * eta / (a * b * (a + b)) * oneMinusEaTau * oneMinusEbTau * (1.0 - Math.Exp(-(a + b) * t));

        var pmT = ZeroCurve.DiscountFactor(curve, valuationDate, T);
        var pmt = ZeroCurve.DiscountFactor(curve, valuationDate, t);
        var forwardValue = notional * pmT / pmt;

        var sigmaTotal = Math.Sqrt(Math.Max(sigmaSq, 0.0));
        if (sigmaTotal < 1e-12)
        {
            var intrinsic = isCall ? Math.Max(forwardValue - strike, 0.0) : Math.Max(strike - forwardValue, 0.0);
            return pmt * intrinsic;
        }

        var h1 = Math.Log(notional * pmT / (strike * pmt)) / sigmaTotal + 0.5 * sigmaTotal;
        var h2 = h1 - sigmaTotal;

        return isCall
            ? notional * pmT * NormalDistribution.Cdf(h1) - strike * pmt * NormalDistribution.Cdf(h2)
            : strike * pmt * NormalDistribution.Cdf(-h2) - notional * pmT * NormalDistribution.Cdf(-h1);
    }
}
