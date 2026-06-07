using Aegis.Instruments;
using RandomSimulator;

namespace MCPricer.FX;

/// <summary>
/// European vanilla FX option pricer under SABR stochastic volatility,
/// simulated under the T-forward measure.
///
/// Payoff at T: max(φ·(F(T) − K), 0)  where φ = +1 (call) or −1 (put).
///
/// Price = P(0,T) · E^T[max(φ·(F(T) − K), 0)]
///       = exp(−r_d·T) · E^T[payoff]
///
/// Analytical benchmark: Hagan et al. (2002) SABR implied vol approximation.
///   For β = 1 (lognormal SABR): compare MC price to GK/BS price with Hagan σ_B.
///   For β = 0 (normal SABR):   compare to Bachelier price with Hagan σ_N.
///   For ν = 0:                  SABR reduces to GBM → compare directly to GK.
///
/// ── Greeks ────────────────────────────────────────────────────────────────────
/// Inherited ComputeGreeks() computes: Delta, Gamma, Vega (= dV/dα), Rho.
/// Additional SABR Greeks available via:
///   ComputeVolOfVolSensitivity() → dV/dν
///   ComputeCorrelSensitivity()   → dV/dρ (Brownian correlation)
///
/// BumpType.VolUp/VolDown maps to bumping α (the SABR initial vol level),
/// consistent with market convention that "vega" = sensitivity to the
/// at-the-money vol level.
/// </summary>
public sealed class FXSABRVanillaOptionMCPricer : FXSABRMCPricer
{
    private readonly double _strike;
    private readonly double _phi;   // +1 call, −1 put

    public FXSABRVanillaOptionMCPricer(
        Option          option,
        DateOnly        valuationDate,
        FxMarketData    market,
        SabrParameters  sabr,
        SimulationCube  cube)
        : base(option, valuationDate, market, sabr, cube)
    {
        if (option.Strike <= 0)
            throw new ArgumentException("Strike must be positive.", nameof(option));
        if (option.KindCase != Option.KindOneofCase.Vanilla)
            throw new ArgumentException("Option.Kind must be Vanilla.", nameof(option));

        _strike = option.Strike;
        _phi    = option.OptionType == OptionType.Call ? 1.0 : -1.0;
    }

    protected override double EvaluatePayoff(
        ReadOnlySpan<double> fwdPath,
        ReadOnlySpan<double> _)          // vol path unused for vanilla payoff
    {
        return Math.Max(_phi * (fwdPath[^1] - _strike), 0.0);
    }

    /// <summary>
    /// Creates a new pricer with one SABR or market parameter bumped by epsilon,
    /// reusing the same SimulationCube for common-random-number variance reduction.
    ///
    /// BumpType mapping:
    ///   SpotUp/Down         → Market.Spot  (shifts InitialForward proportionally)
    ///   VolUp/Down          → Sabr.Alpha   (SABR vega = dV/dα, the ATM vol sensitivity)
    ///   RateUp/Down         → Market.DomesticRate
    ///   ForeignRateUp/Down  → Market.ForeignRate
    ///   SabrNuUp/Down       → Sabr.Nu      (vol-of-vol sensitivity)
    ///   SabrRhoUp/Down      → Sabr.Rho     (correlation sensitivity; clamped to (−1,1))
    /// </summary>
    protected override MCBasePricer CreateBumped(BumpType bump, double epsilon)
    {
        var m    = Market.Clone();
        var sabr = Sabr.Clone();

        switch (bump)
        {
            case BumpType.SpotUp:           m.Spot         += epsilon;                                           break;
            case BumpType.SpotDown:         m.Spot         -= epsilon;                                           break;
            case BumpType.VolUp:            sabr.Alpha     += epsilon;                                           break;
            case BumpType.VolDown:          sabr.Alpha      = Math.Max(1e-8, sabr.Alpha - epsilon);              break;
            case BumpType.RateUp:           m.DomesticRate  = ZeroCurve.Shift(m.DomesticRate,  epsilon);          break;
            case BumpType.RateDown:         m.DomesticRate  = ZeroCurve.Shift(m.DomesticRate, -epsilon);          break;
            case BumpType.ForeignRateUp:    m.ForeignRate   = ZeroCurve.Shift(m.ForeignRate,   epsilon);          break;
            case BumpType.ForeignRateDown:  m.ForeignRate   = ZeroCurve.Shift(m.ForeignRate,  -epsilon);          break;
            case BumpType.SabrNuUp:         sabr.Nu        += epsilon;                                           break;
            case BumpType.SabrNuDown:       sabr.Nu         = Math.Max(0.0, sabr.Nu - epsilon);                  break;
            case BumpType.SabrRhoUp:        sabr.Rho        = Math.Min(1.0 - 1e-6, sabr.Rho + epsilon);         break;
            case BumpType.SabrRhoDown:      sabr.Rho        = Math.Max(-1.0 + 1e-6, sabr.Rho - epsilon);        break;
            default: throw new NotSupportedException($"Bump {bump} not supported by {GetType().Name}.");
        }

        return new FXSABRVanillaOptionMCPricer(OptionDef, ValuationDate, m, sabr, Cube);
    }
}
