namespace AnalyticalPricers;

/// <summary>
/// Bachelier (normal/arithmetic Brownian motion) closed-form price for a
/// European vanilla option on a forward — the standard benchmark for
/// normal-SABR (β = 0) and other normal-vol models.
///
/// ── Model ─────────────────────────────────────────────────────────────────────
/// Stochastic process (T-forward measure Q^T):
///   dF = σ_N dW                         (σ_N = normal/basis-point volatility)
///
///   Call = P(0,T)·[ (F−K)·Φ(d) + σ_N√T·φ(d) ]
///   Put  = P(0,T)·[ (K−F)·Φ(−d) + σ_N√T·φ(d) ]
///   d = (F−K) / (σ_N√T)
///
/// where Φ, φ are the standard normal CDF/PDF (φ is even, so φ(d) = φ(−d)) and
/// P(0,T) is the discount factor supplied by the caller (Bachelier prices a
/// forward payoff; this formula folds in discounting explicitly so it can be
/// compared directly to discounted MC prices).
///
/// ── Edge cases ────────────────────────────────────────────────────────────────
///   σ_N = 0: deterministic forward; price = P(0,T)·max(φ·(F−K), 0).
///   T = 0:   d is singular (0/0 at F=K); both branches collapse to the same
///            zero-vol intrinsic value, so the σ_N·√T = 0 short-circuit covers it.
/// </summary>
public static class Bachelier
{
    /// <summary>
    /// Prices a European vanilla option on a forward under the Bachelier (normal) model.
    /// </summary>
    /// <param name="forward">Forward level F = F(0,T).</param>
    /// <param name="strike">Strike K.</param>
    /// <param name="expiry">Time to expiry T, in years.</param>
    /// <param name="normalVolatility">Normal (basis-point) volatility σ_N, in forward units per √year.</param>
    /// <param name="discountFactor">Discount factor P(0,T) applied to the forward-measure price.</param>
    /// <param name="isCall">True for a call, false for a put.</param>
    public static double Price(
        double forward, double strike, double expiry, double normalVolatility,
        double discountFactor, bool isCall)
    {
        var phi = isCall ? 1.0 : -1.0;
        var totalVol = normalVolatility * Math.Sqrt(expiry);

        if (totalVol <= 0.0)
            return discountFactor * Math.Max(phi * (forward - strike), 0.0);

        var d = (forward - strike) / totalVol;
        return discountFactor * (phi * (forward - strike) * NormalDistribution.Cdf(phi * d)
                                 + totalVol * NormalDistribution.Pdf(d));
    }
}
