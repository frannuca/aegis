namespace MCPricer;

/// <summary>
/// Identifies which model parameter is bumped in a bump-and-reprice Greek calculation.
///
/// Spot / vol / rate bumps apply to all GBM-based and SABR pricers.
/// Carry-curve sensitivity (FX: foreign rate; equity: dividend yield) is computed
/// separately via GbmOptionMCPricer.ComputeCarryRho(market, eps).
///
/// SABR-specific bumps (Nu, Rho) are used only by FXSABRMCPricer.
/// </summary>
public enum BumpType
{
    SpotUp,
    SpotDown,
    VolUp,
    VolDown,
    RateUp,
    RateDown,

    // ── SABR model-specific bumps ─────────────────────────────────────────────
    SabrNuUp,    // vol-of-vol ν bump; dV/dν = Volvol greek
    SabrNuDown,
    SabrRhoUp,   // Brownian correlation ρ bump; dV/dρ = correlation sensitivity
    SabrRhoDown
}
