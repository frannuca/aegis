namespace AnalyticalPricers;

/// <summary>
/// Garman-Kohlhagen closed-form price for a European vanilla FX option — the
/// Black-Scholes formula adapted to a currency pair, where the foreign rate
/// plays the role of a continuous dividend yield.
///
/// ── Model ─────────────────────────────────────────────────────────────────────
/// Stochastic process (domestic risk-neutral measure):
///   dS/S = (r_d − r_f) dt + σ dW
///
/// Numeraire/measure: domestic money-market account, domestic risk-neutral measure Q^d.
///
/// Price = e^{−r_f·T}·S·Φ(φ·d₁) − e^{−r_d·T}·K·Φ(φ·d₂)   ... for φ = +1 (call)
///       = e^{−r_d·T}·K·Φ(−φ·d₂) − e^{−r_f·T}·S·Φ(−φ·d₁)  ... for φ = −1 (put)
///
///   d₁ = [ln(S/K) + (r_d − r_f + ½σ²)T] / (σ√T)
///   d₂ = d₁ − σ√T
///
/// ── Edge cases ────────────────────────────────────────────────────────────────
///   σ = 0: deterministic forward F = S·e^{(r_d−r_f)T}; price = e^{−r_d·T}·max(φ(F−K), 0).
/// </summary>
public static class GarmanKohlhagen
{
    /// <summary>
    /// Prices a European vanilla FX option under Garman-Kohlhagen.
    /// </summary>
    /// <param name="spot">Current FX spot S (domestic per unit foreign).</param>
    /// <param name="strike">Strike K.</param>
    /// <param name="expiry">Time to expiry T, in years.</param>
    /// <param name="volatility">Lognormal volatility σ (flat, Black convention).</param>
    /// <param name="domesticRate">Continuously compounded domestic zero rate r_d.</param>
    /// <param name="foreignRate">Continuously compounded foreign zero rate r_f.</param>
    /// <param name="isCall">True for a call, false for a put.</param>
    public static double Price(
        double spot, double strike, double expiry, double volatility,
        double domesticRate, double foreignRate, bool isCall)
    {
        if (volatility <= 0.0)
        {
            var forward   = spot * Math.Exp((domesticRate - foreignRate) * expiry);
            var intrinsic = isCall
                ? Math.Max(forward - strike, 0.0)
                : Math.Max(strike - forward, 0.0);
            return Math.Exp(-domesticRate * expiry) * intrinsic;
        }

        var sqrtT = Math.Sqrt(expiry);
        var d1 = (Math.Log(spot / strike) + (domesticRate - foreignRate + 0.5 * volatility * volatility) * expiry)
                 / (volatility * sqrtT);
        var d2 = d1 - volatility * sqrtT;

        return isCall
            ? spot   * Math.Exp(-foreignRate * expiry) * NormalDistribution.Cdf( d1)
              - strike * Math.Exp(-domesticRate * expiry) * NormalDistribution.Cdf( d2)
            : strike * Math.Exp(-domesticRate * expiry) * NormalDistribution.Cdf(-d2)
              - spot   * Math.Exp(-foreignRate * expiry) * NormalDistribution.Cdf(-d1);
    }
}
