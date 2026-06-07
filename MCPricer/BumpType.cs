namespace MCPricer;

/// <summary>
/// Identifies which market parameter is bumped in a bump-and-reprice Greek calculation.
///
/// Convention for FX instruments:
///   RateUp / RateDown          → domestic rate (r_d)
///   ForeignRateUp / Down       → foreign rate  (r_f)
///
/// Convention for equity instruments:
///   RateUp / RateDown          → risk-free rate
///   DividendYieldUp / Down     → continuous dividend yield
///   ForeignRateUp / Down       → not applicable (throw NotSupportedException)
///
/// VolUp / VolDown bumps the flat volatility parameter.
/// For SABR pricers, derived classes should override CreateBumped so that
/// VolUp/Down bumps the SABR initial vol (alpha) rather than a non-existent
/// flat sigma. Vol-of-vol (nu) and correlation (rho) sensitivity require
/// additional BumpType values or a dedicated SABR greek override.
/// </summary>
public enum BumpType
{
    SpotUp,
    SpotDown,
    VolUp,
    VolDown,
    RateUp,
    RateDown,
    ForeignRateUp,
    ForeignRateDown,
    DividendYieldUp,
    DividendYieldDown,

    // ── SABR model–specific bumps ─────────────────────────────────────────────
    // Used by FXSABRMCPricer and any other SABR-based pricer.
    // VolUp/VolDown already maps to alpha (σ₀) in SABR pricers.
    SabrNuUp,    // vol-of-vol ν bump; dV/dν = Volvol greek
    SabrNuDown,
    SabrRhoUp,   // Brownian correlation ρ bump; dV/dρ = correlation sensitivity
    SabrRhoDown
}
