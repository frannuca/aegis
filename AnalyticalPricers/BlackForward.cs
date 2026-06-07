namespace AnalyticalPricers;

/// <summary>
/// Undiscounted Black formula on a forward — the T-forward-measure price of a
/// European vanilla payoff on a lognormal forward F(T), with no separate
/// discounting (the numeraire already absorbs it; multiply by P(0,T) to recover
/// a spot-measure price).
///
/// ── Model ─────────────────────────────────────────────────────────────────────
/// Stochastic process (T-forward measure Q^T):
///   dF/F = σ dW
///
///   Call = F·Φ(d₁) − K·Φ(d₂)
///   Put  = K·Φ(−d₂) − F·Φ(−d₁)
///   d₁ = [ln(F/K) + ½σ²T] / (σ√T)
///   d₂ = d₁ − σ√T
///
/// Used as the lognormal-SABR (β = 1) benchmark: combine with
/// <see cref="HaganSabr.LognormalVol"/> for σ and discount by P(0,T) externally.
///
/// ── Edge cases ────────────────────────────────────────────────────────────────
///   σ = 0: deterministic forward; price = max(φ(F−K), 0).
/// </summary>
public static class BlackForward
{
    /// <summary>European call on a lognormal forward, undiscounted.</summary>
    public static double Call(double forward, double strike, double expiry, double volatility)
    {
        if (volatility <= 0.0) return Math.Max(forward - strike, 0.0);

        var sqrtT = Math.Sqrt(expiry);
        var d1    = (Math.Log(forward / strike) + 0.5 * volatility * volatility * expiry) / (volatility * sqrtT);
        var d2    = d1 - volatility * sqrtT;
        return forward * NormalDistribution.Cdf(d1) - strike * NormalDistribution.Cdf(d2);
    }

    /// <summary>European put on a lognormal forward, undiscounted.</summary>
    public static double Put(double forward, double strike, double expiry, double volatility)
    {
        if (volatility <= 0.0) return Math.Max(strike - forward, 0.0);

        var sqrtT = Math.Sqrt(expiry);
        var d1    = (Math.Log(forward / strike) + 0.5 * volatility * volatility * expiry) / (volatility * sqrtT);
        var d2    = d1 - volatility * sqrtT;
        return strike * NormalDistribution.Cdf(-d2) - forward * NormalDistribution.Cdf(-d1);
    }
}
