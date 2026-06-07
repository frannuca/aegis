namespace AnalyticalPricers;

/// <summary>
/// Closed-form prices for European cash-or-nothing and asset-or-nothing
/// digital ("binary") options under a Black-Scholes-style lognormal model
/// with a continuous carry (dividend yield for equities, foreign rate for FX
/// — Garman-Kohlhagen and Black-Scholes are structurally identical here, so
/// one pair of formulas serves both: pass (rate, carry) = (r, q) for equities
/// or (r_d, r_f) for FX).
///
/// ── Model / measure ──────────────────────────────────────────────────────────
/// Stochastic process (risk-neutral measure Q, money-market numeraire):
///   dS/S = (rate − carry) dt + σ dW
///
/// ── Formulas (Hull, "Options, Futures and Other Derivatives") ────────────────
///   Cash-or-nothing:  pays Q·1{S(T) ITM}
///     Call = e^{-rate·T}·Q·Φ(d₂)
///     Put  = e^{-rate·T}·Q·Φ(−d₂)
///
///   Asset-or-nothing: pays S(T)·1{S(T) ITM}
///     Call = S·e^{-carry·T}·Φ(d₁)
///     Put  = S·e^{-carry·T}·Φ(−d₁)
///
///   d₁ = [ln(S/K) + (rate − carry + ½σ²)·T] / (σ√T)
///   d₂ = d₁ − σ√T
///
/// "ITM" is defined consistently with the vanilla payoff: call ⇒ S(T) > K,
/// put ⇒ S(T) < K (the boundary S(T) = K has probability zero).
///
/// ── Edge cases ────────────────────────────────────────────────────────────────
///   σ = 0 or T = 0: the path is deterministic, S(T) = forward F = S·e^{(rate−carry)T}.
///   Cash-or-nothing  → e^{-rate·T}·Q·1{F ITM}
///   Asset-or-nothing → e^{-rate·T}·F·1{F ITM} = S·e^{-carry·T}·1{F ITM}
///     (using e^{-rate·T}·F = S·e^{-carry·T})
/// </summary>
public static class DigitalBlackScholes
{
    /// <summary>Cash-or-nothing digital price: pays a fixed amount <paramref name="payout"/> if in-the-money at expiry.</summary>
    public static double CashOrNothing(
        double spot, double strike, double expiry, double volatility,
        double rate, double carry, double payout, bool isCall)
    {
        var phi      = isCall ? 1.0 : -1.0;
        var discount = Math.Exp(-rate * expiry);
        var totalVol = volatility * Math.Sqrt(expiry);

        if (totalVol <= 0.0)
        {
            var forward = spot * Math.Exp((rate - carry) * expiry);
            var isItm   = phi * (forward - strike) > 0.0;
            return isItm ? discount * payout : 0.0;
        }

        var d2 = (Math.Log(spot / strike) + (rate - carry - 0.5 * volatility * volatility) * expiry) / totalVol;
        return discount * payout * NormalDistribution.Cdf(phi * d2);
    }

    /// <summary>Asset-or-nothing digital price: pays the underlying spot S(T) if in-the-money at expiry.</summary>
    public static double AssetOrNothing(
        double spot, double strike, double expiry, double volatility,
        double rate, double carry, bool isCall)
    {
        var phi      = isCall ? 1.0 : -1.0;
        var totalVol = volatility * Math.Sqrt(expiry);

        if (totalVol <= 0.0)
        {
            var forward = spot * Math.Exp((rate - carry) * expiry);
            var isItm   = phi * (forward - strike) > 0.0;
            return isItm ? spot * Math.Exp(-carry * expiry) : 0.0;
        }

        var d2 = (Math.Log(spot / strike) + (rate - carry - 0.5 * volatility * volatility) * expiry) / totalVol;
        var d1 = d2 + totalVol;
        return spot * Math.Exp(-carry * expiry) * NormalDistribution.Cdf(phi * d1);
    }
}
