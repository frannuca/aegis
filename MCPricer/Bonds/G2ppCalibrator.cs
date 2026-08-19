using Aegis.Instruments;
using NodaTime;

namespace MCPricer.Bonds;

/// <summary>
/// A European option on a zero-coupon bond used as a calibration target.
/// Caps, floors and (approximately) swaptions all decompose into instruments
/// of this shape, which is why it is the model's native calibration instrument
/// — <see cref="G2ppAnalytics.ZeroCouponBondOption"/> prices it in closed form.
/// </summary>
public sealed record G2ppCalibrationInstrument(
    double OptionExpiry,
    double BondMaturity,
    double Strike,
    double Notional,
    bool   IsCall,
    double MarketPrice)
{
    public double OptionExpiry { get; } = OptionExpiry > 0.0
        ? OptionExpiry
        : throw new ArgumentOutOfRangeException(nameof(OptionExpiry));

    public double BondMaturity { get; } = BondMaturity > OptionExpiry
        ? BondMaturity
        : throw new ArgumentOutOfRangeException(nameof(BondMaturity), "Bond maturity must be > option expiry.");

    public double Strike { get; } = Strike > 0.0
        ? Strike
        : throw new ArgumentOutOfRangeException(nameof(Strike));

    public double Notional { get; } = Notional > 0.0
        ? Notional
        : throw new ArgumentOutOfRangeException(nameof(Notional));

    public double MarketPrice { get; } = MarketPrice > 0.0
        ? MarketPrice
        : throw new ArgumentOutOfRangeException(nameof(MarketPrice), "Market price must be > 0.");
}

/// <summary>
/// Calibrates G2++ (two-factor Hull-White) parameters (A1, Sigma1, A2, Sigma2,
/// Rho) to a set of market zero-coupon-bond-option prices by nonlinear
/// least squares, minimizing the sum of squared *relative* pricing errors:
///
///   RMSE² = (1/n) Σ [ (ModelPrice_i - MarketPrice_i) / MarketPrice_i ]²
///
/// Relative errors are used so that instruments of very different price
/// scales (short- vs long-dated options) contribute comparably to the
/// objective — an absolute-error objective would be dominated by the
/// largest-premium instrument.
///
/// ── Overfitting guard ─────────────────────────────────────────────────────────
/// The model has 5 free parameters; calibrating to fewer than 5 instruments
/// is rejected outright (an underdetermined fit is not a calibration, it is
/// curve-fitting noise).
///
/// ── No silent fallback ────────────────────────────────────────────────────────
/// If the optimizer fails to converge, or converges to a fit whose RMSE
/// exceeds <paramref name="maxAcceptableRmse"/>, <see cref="Calibrate"/> throws
/// rather than returning a poorly-fitted parameter set.
///
/// ── Unconstrained reparameterisation ──────────────────────────────────────────
/// Nelder-Mead searches over an unconstrained R⁵ and each candidate is mapped
/// back to the model's constrained domain via:
///   A1, A2         = exp(u)           (> 0)
///   Sigma1, Sigma2 = exp(u)           (>= 0, exp is always > 0)
///   Rho            = tanh(u)          (strictly within (-1, 1))
/// This lets the simplex explore freely without ever evaluating
/// <see cref="G2ppParameters"/> at an invalid point.
/// </summary>
public static class G2ppCalibrator
{
    private const int ParameterCount = 5;

    public static G2ppParameters Calibrate(
        Pillars curve,
        LocalDate valuationDate,
        IReadOnlyList<G2ppCalibrationInstrument> instruments,
        G2ppParameters? initialGuess = null,
        double maxAcceptableRmse = 1e-3)
    {
        ArgumentNullException.ThrowIfNull(curve);
        ArgumentNullException.ThrowIfNull(instruments);
        if (instruments.Count < ParameterCount)
            throw new ArgumentException(
                $"Calibration requires at least {ParameterCount} instruments (one per free " +
                $"parameter: A1, Sigma1, A2, Sigma2, Rho) to avoid overfitting; got {instruments.Count}.",
                nameof(instruments));

        var guess = initialGuess ?? new G2ppParameters(a1: 0.5, sigma1: 0.01, a2: 0.05, sigma2: 0.01, rho: -0.5);
        var u0 = new[]
        {
            Math.Log(guess.A1),
            Math.Log(guess.Sigma1),
            Math.Log(guess.A2),
            Math.Log(guess.Sigma2),
            Math.Atanh(Math.Clamp(guess.Rho, -0.999, 0.999)),
        };

        double Objective(double[] u)
        {
            var p = FromUnconstrained(u);
            var sumSq = 0.0;
            foreach (var inst in instruments)
            {
                var model = G2ppAnalytics.ZeroCouponBondOption(
                    curve, valuationDate, p, inst.OptionExpiry, inst.BondMaturity, inst.Strike, inst.Notional, inst.IsCall);
                var relError = (model - inst.MarketPrice) / inst.MarketPrice;
                sumSq += relError * relError;
            }
            return sumSq / instruments.Count;
        }

        var result = NelderMead.Minimize(Objective, u0, initialStep: 0.25, tolerance: 1e-12, maxIterations: 5000);
        var rmse = Math.Sqrt(Math.Max(result.Value, 0.0));

        if (rmse > maxAcceptableRmse)
            throw new InvalidOperationException(
                $"G2++ calibration failed to converge to an acceptable fit: RMSE={rmse:E3} " +
                $"exceeds tolerance {maxAcceptableRmse:E3} after {result.Iterations} iterations " +
                $"(optimizer converged={result.Converged}). No fallback parameter set is returned.");

        return FromUnconstrained(result.ArgMin);
    }

    private static G2ppParameters FromUnconstrained(double[] u) =>
        new(
            a1:     Math.Exp(u[0]),
            sigma1: Math.Exp(u[1]),
            a2:     Math.Exp(u[2]),
            sigma2: Math.Exp(u[3]),
            rho:    Math.Tanh(u[4]));
}
