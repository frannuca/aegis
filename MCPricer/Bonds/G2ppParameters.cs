namespace MCPricer.Bonds;

/// <summary>
/// Immutable parameter set for the two-factor Hull-White / G2++ short-rate model
/// (Brigo &amp; Mercurio, "Interest Rate Models — Theory and Practice", 2nd ed., §4.2).
///
/// ── Stochastic process (risk-neutral measure Q) ───────────────────────────────
///   dx(t) = -A1·x(t) dt + Sigma1 dW1(t),  x(0) = 0
///   dy(t) = -A2·y(t) dt + Sigma2 dW2(t),  y(0) = 0
///   dW1(t)·dW2(t) = Rho dt
///   r(t) = x(t) + y(t) + φ(t)
///
/// φ(t) is not stored here — it is a deterministic function of the market
/// discount curve and these five parameters (see <see cref="G2ppAnalytics.Phi"/>),
/// chosen so the model reprices the initial term structure exactly.
///
/// A1, A2 are mean-reversion speeds; Sigma1, Sigma2 are factor volatilities;
/// Rho is the instantaneous correlation between the two Brownian drivers.
/// Two factors (rather than one) let the model fit a decorrelated short-end /
/// long-end volatility term structure, which a single-factor Hull-White cannot.
/// </summary>
public sealed record G2ppParameters
{
    public double A1     { get; }
    public double Sigma1 { get; }
    public double A2     { get; }
    public double Sigma2 { get; }
    public double Rho    { get; }

    public G2ppParameters(double a1, double sigma1, double a2, double sigma2, double rho)
    {
        if (a1 <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(a1), "Mean-reversion speed A1 must be > 0.");
        if (a2 <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(a2), "Mean-reversion speed A2 must be > 0.");
        if (sigma1 < 0.0)
            throw new ArgumentOutOfRangeException(nameof(sigma1), "Sigma1 must be >= 0.");
        if (sigma2 < 0.0)
            throw new ArgumentOutOfRangeException(nameof(sigma2), "Sigma2 must be >= 0.");
        if (rho <= -1.0 || rho >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(rho), "Rho must lie strictly within (-1, 1).");

        A1 = a1; Sigma1 = sigma1; A2 = a2; Sigma2 = sigma2; Rho = rho;
    }
}
