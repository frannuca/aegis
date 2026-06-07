using Aegis.Instruments;
using RandomSimulator;

namespace MCPricer.Equity;

/// <summary>
/// European vanilla equity option pricer (call or put).
///
/// Payoff at T: max(φ·(S(T) − K), 0)  where φ = +1 (call) or −1 (put).
///
/// Only the terminal spot S(T) = spotPath[^1] is used.
///
/// Analytical benchmark: Black-Scholes formula.
///   Call = S·e^{-q·T}·N(d₁) − K·e^{-r·T}·N(d₂)
///   d₁ = [log(S/K) + (r − q + ½σ²)·T] / (σ√T)
///   d₂ = d₁ − σ√T
///
/// Edge cases handled:
///   σ = 0: price = e^{-r·T}·max(φ·(S·e^{(r-q)T} − K), 0) (forward intrinsic value).
/// </summary>
public sealed class EQVanillaOptionMCPricer : EQMCPricer
{
    private readonly double _strike;
    private readonly double _phi;   // +1 for call, −1 for put

    public EQVanillaOptionMCPricer(
        Option         option,
        DateOnly       valuationDate,
        EqMarketData   market,
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
            case BumpType.SpotUp:             m.Spot          += epsilon;                         break;
            case BumpType.SpotDown:           m.Spot          -= epsilon;                         break;
            case BumpType.VolUp:              vol             += epsilon;                         break;
            case BumpType.VolDown:            vol              = Math.Max(0.0, vol - epsilon);    break;
            case BumpType.RateUp:             m.RiskFreeRate  = ZeroCurve.Shift(m.RiskFreeRate,   epsilon);  break;
            case BumpType.RateDown:           m.RiskFreeRate  = ZeroCurve.Shift(m.RiskFreeRate,  -epsilon);  break;
            case BumpType.DividendYieldUp:    m.DividendYield = ZeroCurve.Shift(m.DividendYield,  epsilon);  break;
            case BumpType.DividendYieldDown:  m.DividendYield = ZeroCurve.Shift(m.DividendYield, -epsilon);  break;
            default: throw new NotSupportedException($"Bump {bump} not supported by {GetType().Name}.");
        }
        return new EQVanillaOptionMCPricer(OptionDef, ValuationDate, m, vol, Cube);
    }
}
