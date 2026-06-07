namespace MCPricer;

/// <summary>
/// First- and second-order sensitivities produced by bump-and-reprice Greek calculation.
///
/// Delta   = dV/dS           (central finite difference in spot)
/// Gamma   = d²V/dS²         (second-order central finite difference)
/// Vega    = dV/dσ            (central finite difference in flat vol)
/// Theta   = -dV/dT           (NaN — time-bump requires rebuilding the SimulationCube;
///                             use an analytical formula or override ComputeGreeks)
/// Rho     = dV/dr            (forward difference in the primary rate;
///                             domestic rate for FX, risk-free rate for equity)
///
/// Units are consistent with the pricer's price units (e.g., domestic currency per unit
/// of notional). Users are responsible for choosing bump sizes (spotEps, volEps, rateEps)
/// appropriate to their parameter scale.
/// </summary>
public sealed record GreekResult(
    double Delta,
    double Gamma,
    double Vega,
    double Theta,
    double Rho);
