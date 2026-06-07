namespace AnalyticalPricers;

/// <summary>
/// Hagan et al. (2002) SABR implied-vol approximations — the standard
/// analytical benchmark for SABR-based Monte Carlo pricers.
///
/// ── Model ─────────────────────────────────────────────────────────────────────
/// SABR dynamics under the T-forward measure (Hagan-style):
///   dF = α·F^β dW₁
///   dα = ν·α dW₂,    corr(dW₁, dW₂) = ρ
///
/// where F = F(0,T) is the forward, α the initial vol level, β the CEV exponent,
/// ρ the spot/vol correlation and ν the vol-of-vol.
/// </summary>
public static class Hagan
{
    /// <summary>
    /// Hagan's lognormal-SABR implied Black volatility approximation σ_B(F, K).
    ///
    /// Combine with <see cref="BlackForward"/> (β = 1 ⇒ pure lognormal limit
    /// when ν = 0) or <see cref="GarmanKohlhagen"/>/<see cref="BlackScholes"/>
    /// (discounted spot-measure price) as the analytical benchmark for a
    /// lognormal-SABR (β close to 1) Monte Carlo price.
    ///
    /// σ_B(F,K) = [ α / ((FK)^((1-β)/2)·D(ζ)) ] · (z/χ(z)) · (1 + correction·T)
    ///
    ///   x          = ln(F/K)
    ///   D(ζ)       = 1 + (1-β)²/24·x² + (1-β)⁴/1920·x⁴             (series in x)
    ///   z          = (ν/α)·(FK)^((1-β)/2)·x
    ///   χ(z)       = ln{ [√(1-2ρz+z²) + z - ρ] / (1-ρ) }
    ///   correction = (1-β)²α² / (24·(FK)^(1-β))
    ///              + ρβνα     / (4·(FK)^((1-β)/2))
    ///              + (2-3ρ²)ν²/ 24
    ///
    /// z/χ(z) → 1 as z → 0 (handles ν → 0 and at-the-money F → K limits).
    /// </summary>
    /// <param name="forward">Forward F.</param>
    /// <param name="strike">Strike K.</param>
    /// <param name="expiry">Time to expiry T, in years.</param>
    /// <param name="alpha">SABR initial volatility level α.</param>
    /// <param name="beta">SABR CEV exponent β ∈ [0,1].</param>
    /// <param name="rho">Spot/vol correlation ρ ∈ (-1,1).</param>
    /// <param name="nu">SABR vol-of-vol ν ≥ 0.</param>
    public static double BlackVol(
        double forward, double strike, double expiry,
        double alpha, double beta, double rho, double nu)
    {
        var f = forward;
        var k = strike;
        var t = expiry;

        var x      = Math.Log(f / k);
        var fk     = Math.Sqrt(f * k);
        var beta1  = 1.0 - beta;
        var fkb1   = Math.Pow(fk, beta1);   // (FK)^((1-β)/2)
        var fkb1sq = fkb1 * fkb1;           // (FK)^(1-β)

        // Correction term (applied at fk, the geometric average):
        //   (1-β)²α²/(24·(FK)^(1-β)) + ρβνα/(4·(FK)^((1-β)/2)) + (2-3ρ²)ν²/24
        var correction = (beta1 * beta1 * alpha * alpha / (24.0 * fkb1sq)
                       + rho * beta * nu * alpha / (4.0 * fkb1)
                       + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0) * t;

        // ATM or near-ATM: avoid 0/0 in z/χ and the log series
        if (Math.Abs(x) < 1e-7)
            return alpha / Math.Pow(f, beta1) * (1.0 + correction);

        // x-series denominator: 1 + (1-β)²/24·x² + (1-β)⁴/1920·x⁴
        var b1sq    = beta1 * beta1;
        var xSeries = 1.0 + b1sq / 24.0 * x * x + b1sq * b1sq / 1920.0 * x * x * x * x;

        if (nu < 1e-10)
        {
            // ν = 0: σ is constant; z = 0 → z/χ = 1; use geometric-mean formula
            return alpha / (fkb1 * xSeries) * (1.0 + correction);
        }

        // z = (ν/α) · (FK)^((1-β)/2) · log(F/K)
        var z = nu / alpha * fkb1 * x;

        // χ(z) = log{[√(1−2ρz+z²) + z − ρ] / (1−ρ)}
        var sqrtTerm = Math.Sqrt(1.0 - 2.0 * rho * z + z * z);
        var chiZ     = Math.Log((sqrtTerm + z - rho) / (1.0 - rho));

        // z/χ(z): limit → 1 when z → 0
        var zOverChi = Math.Abs(chiZ) < 1e-10 ? 1.0 : z / chiZ;

        return alpha / (fkb1 * xSeries) * zOverChi * (1.0 + correction);
    }

    /// <summary>
    /// Hagan's normal-SABR (β = 0) at-the-money implied normal-vol approximation.
    ///
    ///   σ_N(F,F) = α · (1 + (2-3ρ²)·ν²/24 · T)
    ///
    /// Combine with <see cref="Bachelier"/> (forward = strike) as the analytical
    /// benchmark for an at-the-money normal-SABR Monte Carlo price. This is the
    /// ATM specialisation of Hagan's normal-vol expansion (the general F ≠ K
    /// formula additionally involves a z/χ(z) term, mirroring <see cref="BlackVol"/>).
    /// </summary>
    /// <param name="alpha">SABR initial volatility level α (≡ σ_N at the base point when ν, ρ → 0).</param>
    /// <param name="rho">Spot/vol correlation ρ ∈ (-1,1).</param>
    /// <param name="nu">SABR vol-of-vol ν ≥ 0.</param>
    /// <param name="expiry">Time to expiry T, in years.</param>
    public static double NormalVolAtm(double alpha, double rho, double nu, double expiry) =>
        alpha * (1.0 + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0 * expiry);
}
