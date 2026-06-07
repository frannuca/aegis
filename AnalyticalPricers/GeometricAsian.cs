namespace AnalyticalPricers;

/// <summary>
/// Kemna-Vorst (1990) closed-form price for a European average-price Asian
/// option on the discrete *geometric* average, under Black-Scholes-style GBM
/// with a continuous carry (dividend yield for equities, foreign rate for FX).
///
/// ── Model / measure ──────────────────────────────────────────────────────────
/// Stochastic process (risk-neutral measure Q):
///   dS/S = (rate − carry) dt + σ dW
///   ln S(t) = ln S(0) + (rate − carry − ½σ²)·t + σ·W(t)
///
/// Monitoring at n equally spaced dates t_i = i·T/n, i = 1,…,n (t_n = T —
/// matching the MC pricer, which averages every simulated step). The discrete
/// geometric average
///   G = exp( (1/n)·Σᵢ ln S(t_i) )
/// is exactly lognormal (a sum of jointly Gaussian log-increments), with
///
///   E[ln G]   = ln S(0) + (rate − carry − ½σ²)·T·(n+1)/(2n)
///   Var[ln G] = σ²·T·(n+1)(2n+1)/(6n²)
///
/// (the variance uses Cov(W(t_i), W(t_j)) = min(t_i, t_j) and the identity
/// Σᵢ Σⱼ min(i,j) = n(n+1)(2n+1)/6).
///
/// ── Reduction to the Black formula on a forward ───────────────────────────────
/// Writing v² = Var[ln G] and F_G = E[G] = exp(E[ln G] + ½v²) (the "forward" of
/// the lognormal G under Q), the price of a European payoff on G reduces to the
/// undiscounted Black formula on lognormal forward F_G with total volatility v:
///
///   Price = e^{-rate·T} · Black(F_G, K, T, σ_eff),   σ_eff = v / √T
///
/// (BlackForward.Call/Put expects an *annualized* vol; σ_eff·√T = v reproduces
/// the exact total log-stdev of G — see <see cref="BlackForward"/>.)
///
/// ── Continuous-monitoring limit ──────────────────────────────────────────────
/// As n → ∞: (n+1)(2n+1)/(6n²) → 1/3 and (n+1)/(2n) → 1/2, recovering the
/// textbook continuous geometric-Asian results Var[ln G] → σ²T/3 and a halved
/// drift term — the standard sanity check on this formula.
///
/// ── Edge cases ────────────────────────────────────────────────────────────────
///   σ = 0 or T = 0: G is deterministic; price = e^{-rate·T}·max(φ·(G−K), 0).
///
/// Note: there is no analogous closed form for the *arithmetic* average (the
/// sum of correlated lognormals is not lognormal) — arithmetic Asian options
/// must be priced by Monte Carlo or moment-matching approximations. This
/// formula is exact only for geometric averaging, and serves as the MC
/// pricer's analytical benchmark for ASIAN_AVERAGING_METHOD_GEOMETRIC.
/// </summary>
public static class GeometricAsian
{
    /// <summary>
    /// Price of a European geometric-average-price Asian option,
    /// max(φ·(Ḡ − K), 0) discounted under Q, with <paramref name="observations"/>
    /// equally spaced monitoring dates over [0, T].
    /// </summary>
    public static double Price(
        double spot, double strike, double expiry, double volatility,
        double rate, double carry, int observations, bool isCall)
    {
        if (observations < 1)
            throw new ArgumentOutOfRangeException(nameof(observations), "Must have at least one monitoring observation.");

        var n = (double)observations;

        var meanLog = Math.Log(spot)
                    + (rate - carry - 0.5 * volatility * volatility) * expiry * (n + 1.0) / (2.0 * n);
        var variance = volatility * volatility * expiry * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n);

        var discount = Math.Exp(-rate * expiry);
        var phi      = isCall ? 1.0 : -1.0;

        if (variance <= 0.0)
        {
            var deterministicAverage = Math.Exp(meanLog);
            return discount * Math.Max(phi * (deterministicAverage - strike), 0.0);
        }

        var forward  = Math.Exp(meanLog + 0.5 * variance);
        var sigmaEff = Math.Sqrt(variance / expiry);

        return discount * (isCall
            ? BlackForward.Call(forward, strike, expiry, sigmaEff)
            : BlackForward.Put(forward, strike, expiry, sigmaEff));
    }
}
