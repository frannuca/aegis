namespace AnalyticalPricers;

/// <summary>
/// Black-Scholes (Merton) closed-form price for a European vanilla equity option
/// with a continuous proportional dividend yield.
///
/// ── Model ─────────────────────────────────────────────────────────────────────
/// Stochastic process (risk-neutral measure Q):
///   dS/S = (r − q) dt + σ dW
///
/// Numeraire/measure: domestic money-market account, risk-neutral measure Q.
///
///   Call = S·e^{−q·T}·Φ(d₁) − K·e^{−r·T}·Φ(d₂)
///   Put  = K·e^{−r·T}·Φ(−d₂) − S·e^{−q·T}·Φ(−d₁)
///   d₁ = [ln(S/K) + (r − q + ½σ²)T] / (σ√T)
///   d₂ = d₁ − σ√T
///
/// ── Edge cases ────────────────────────────────────────────────────────────────
///   σ = 0: deterministic forward F = S·e^{(r−q)T}; price = e^{−r·T}·max(φ(F−K), 0).
/// </summary>
public static class BlackScholes
{
    /// <summary>
    /// Prices a European vanilla equity option under Black-Scholes-Merton.
    /// </summary>
    /// <param name="spot">Current spot S.</param>
    /// <param name="strike">Strike K.</param>
    /// <param name="expiry">Time to expiry T, in years.</param>
    /// <param name="volatility">Lognormal volatility σ (flat, Black convention).</param>
    /// <param name="riskFreeRate">Continuously compounded risk-free rate r.</param>
    /// <param name="dividendYield">Continuously compounded dividend yield q.</param>
    /// <param name="isCall">True for a call, false for a put.</param>
    public static double Price(
        double spot, double strike, double expiry, double volatility,
        double riskFreeRate, double dividendYield, bool isCall)
    {
        if (volatility <= 0.0)
        {
            var forward = spot * Math.Exp((riskFreeRate - dividendYield) * expiry);
            var intrinsic = isCall
                ? Math.Max(forward - strike, 0.0)
                : Math.Max(strike - forward, 0.0);
            return Math.Exp(-riskFreeRate * expiry) * intrinsic;
        }

        var sqrtT = Math.Sqrt(expiry);
        var d1 = (Math.Log(spot / strike) + (riskFreeRate - dividendYield + 0.5 * volatility * volatility) * expiry)
                 / (volatility * sqrtT);
        var d2 = d1 - volatility * sqrtT;

        return isCall
            ? spot   * Math.Exp(-dividendYield * expiry) * NormalDistribution.Cdf( d1)
              - strike * Math.Exp(-riskFreeRate * expiry) * NormalDistribution.Cdf( d2)
            : strike * Math.Exp(-riskFreeRate * expiry) * NormalDistribution.Cdf(-d2)
              - spot   * Math.Exp(-dividendYield * expiry) * NormalDistribution.Cdf(-d1);
    }
}
