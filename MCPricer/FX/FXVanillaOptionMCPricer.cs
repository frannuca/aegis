using Aegis.Instruments;
using RandomSimulator;

namespace MCPricer.FX;

/// <summary>
/// European vanilla FX option pricer (call or put).
///
/// Payoff at T: max(φ·(S(T) − K), 0)  where φ = +1 (call) or −1 (put).
///
/// Only the terminal spot S(T) = spotPath[^1] is used; intermediate values
/// are simulated but irrelevant to the payoff.
///
/// Analytical benchmark: Garman-Kohlhagen formula.
///   Call = S·e^{-r_f·T}·N(d₁) − K·e^{-r_d·T}·N(d₂)
///   d₁ = [log(S/K) + (r_d−r_f+½σ²)·T] / (σ√T)
///   d₂ = d₁ − σ√T
///
/// Edge cases handled:
///   σ = 0: deep-ITM options price = discounted intrinsic; OTM = 0.
///   T = 0: price = max(φ·(S₀−K), 0) (immediate payoff, no discounting).
/// </summary>
public sealed class FXVanillaOptionMCPricer : FXMCPricer
{
    private readonly double _strike;
    private readonly double _phi;   // +1 for call, −1 for put

    public FXVanillaOptionMCPricer(
        Option         option,
        DateOnly       valuationDate,
        FxMarketData   market,
        double         volatility,
        SimulationCube cube)
        : base(option, valuationDate, market, volatility, cube)
    {
        if (option.Strike <= 0)
            throw new ArgumentException("Strike must be positive.", nameof(option));
        if (option.KindCase != Option.KindOneofCase.Vanilla)
            throw new ArgumentException("Option.Kind must be Vanilla.", nameof(option));

        _strike = option.Strike;
        _phi    = option.OptionType == OptionType.Call ? 1.0 : -1.0;
    }

    protected override double EvaluatePayoff(ReadOnlySpan<double> spotPath)
    {
        var terminal = spotPath[^1];
        return Math.Max(_phi * (terminal - _strike), 0.0);
    }

    protected override MCBasePricer CreateBumped(BumpType bump, double epsilon)
    {
        var m   = Market.Clone();
        var vol = Volatility;
        switch (bump)
        {
            case BumpType.SpotUp:           m.Spot         += epsilon;                         break;
            case BumpType.SpotDown:         m.Spot         -= epsilon;                         break;
            case BumpType.VolUp:            vol            += epsilon;                         break;
            case BumpType.VolDown:          vol             = Math.Max(0.0, vol - epsilon);    break;
            case BumpType.RateUp:           m.DomesticRate = ZeroCurve.Shift(m.DomesticRate,  epsilon);  break;
            case BumpType.RateDown:         m.DomesticRate = ZeroCurve.Shift(m.DomesticRate, -epsilon);  break;
            case BumpType.ForeignRateUp:    m.ForeignRate  = ZeroCurve.Shift(m.ForeignRate,   epsilon);  break;
            case BumpType.ForeignRateDown:  m.ForeignRate  = ZeroCurve.Shift(m.ForeignRate,  -epsilon);  break;
            default: throw new NotSupportedException($"Bump {bump} not supported by {GetType().Name}.");
        }
        return new FXVanillaOptionMCPricer(OptionDef, ValuationDate, m, vol, Cube);
    }
}
