using Aegis.Instruments;
using RandomSimulator;
using NodaTime;

namespace MCPricer.FX;

/// <summary>
/// FX Asian (average-rate / average-strike) option pricer — arithmetic or
/// geometric averaging, fixed-strike (average-price) or floating-strike
/// (average-strike), call or put.
///
/// The average Ā is taken over every simulated monitoring point
/// S(t_1), …, S(t_n) = spotPath (n = the cube's step count):
///
///   arithmetic Ā = (1/n)·Σᵢ S(t_i)
///   geometric  Ā = exp((1/n)·Σᵢ ln S(t_i))
///
/// Payoff at T:
///   AVERAGE_PRICE   max(φ·(Ā      − K), 0)   — fixed strike K (= Option.strike)
///   AVERAGE_STRIKE  max(φ·(S(T)   − Ā), 0)   — floating strike = Ā; Option.strike unused
///
/// Analytical benchmark:
///   Geometric averaging has an exact closed form — Kemna-Vorst (1990), see
///   <see cref="AnalyticalPricers.GeometricAsian"/> — because a product of
///   correlated lognormals is itself lognormal.
///   Arithmetic averaging has *no* closed form (a sum of correlated lognormals
///   is not lognormal); it can only be benchmarked via Monte Carlo, moment
///   matching, or bounded against the geometric price (Jensen's inequality:
///   the arithmetic mean ≥ the geometric mean pathwise, so an arithmetic
///   average-price call is worth at least as much as the geometric one).
///
/// Edge cases handled:
///   σ = 0: the path is deterministic, so Ā collapses to the (single) realized
///   deterministic average and the payoff is exact intrinsic value.
/// </summary>
public sealed class FXAsianOptionMCPricer : GbmOptionMCPricer
{
    private readonly double               _strike;
    private readonly double               _phi;   // +1 for call, −1 for put
    private readonly AsianAveragingMethod _averagingMethod;
    private readonly AsianStrikeStyle     _strikeStyle;

    public FXAsianOptionMCPricer(
        Option         option,
        LocalDate       valuationDate,
        double         volatility,
        SimulationCube cube)
        : base(option, valuationDate, volatility, cube)
    {
        if (option.KindCase != Option.KindOneofCase.Asian)
            throw new ArgumentException("Option.Kind must be Asian.", nameof(option));

        var asian = option.Asian;
        if (asian.AveragingMethod == AsianAveragingMethod.Unspecified)
            throw new ArgumentException("AsianOption.AveragingMethod must be specified.", nameof(option));
        if (asian.StrikeStyle == AsianStrikeStyle.Unspecified)
            throw new ArgumentException("AsianOption.StrikeStyle must be specified.", nameof(option));
        if (asian.StrikeStyle == AsianStrikeStyle.AveragePrice && option.Strike <= 0)
            throw new ArgumentException("Strike must be positive for average-price Asian options.", nameof(option));

        _strike          = option.Strike;
        _phi             = option.OptionType == OptionType.Call ? 1.0 : -1.0;
        _averagingMethod = asian.AveragingMethod;
        _strikeStyle     = asian.StrikeStyle;
    }

    protected override double EvaluatePayoff(ReadOnlySpan<double> spotPath)
    {
        var average = _averagingMethod == AsianAveragingMethod.Geometric
            ? GeometricMean(spotPath)
            : ArithmeticMean(spotPath);

        return _strikeStyle == AsianStrikeStyle.AverageStrike
            ? Math.Max(_phi * (spotPath[^1] - average), 0.0)
            : Math.Max(_phi * (average - _strike), 0.0);
    }

    private static double ArithmeticMean(ReadOnlySpan<double> path)
    {
        var sum = 0.0;
        foreach (var s in path) sum += s;
        return sum / path.Length;
    }

    private static double GeometricMean(ReadOnlySpan<double> path)
    {
        var sumLog = 0.0;
        foreach (var s in path) sumLog += Math.Log(s);
        return Math.Exp(sumLog / path.Length);
    }

    protected override MCBasePricer CreateBumped(BumpType bump, double epsilon)
    {
        var vol = Volatility;
        switch (bump)
        {
            case BumpType.VolUp:   vol += epsilon;                          break;
            case BumpType.VolDown: vol  = Math.Max(0.0, vol - epsilon);    break;
            default: throw new NotSupportedException($"Bump {bump} not supported by {GetType().Name}. Use ComputeGreeks(market, ...) for spot and rate bumps.");
        }
        return new FXAsianOptionMCPricer(OptionDef, ValuationDate, vol, Cube);
    }
}
