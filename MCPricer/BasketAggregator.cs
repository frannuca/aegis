using Aegis.Instruments;

namespace MCPricer;

/// <summary>
/// Combines per-leg weighted terminal values into a single basket level,
/// according to <see cref="BasketAggregationMethod"/>.
///
/// Input convention: <c>weightedValues[i] = w_i · S_i(T)</c> — the portfolio
/// weight already applied. The aggregator only performs the reduction; it is
/// agnostic to asset class (FX, equity, ...).
///
///   WEIGHTED_SUM → Σ w_i · S_i(T)               (basket / portfolio level)
///   BEST_OF      → max_i (w_i · S_i(T))         (best-performing weighted leg)
///   WORST_OF     → min_i (w_i · S_i(T))         (worst-performing weighted leg)
///
/// The basket level feeds into the standard payoff max(φ·(level − K), 0).
/// </summary>
public static class BasketAggregator
{
    public static double Aggregate(BasketAggregationMethod method, ReadOnlySpan<double> weightedValues)
    {
        if (weightedValues.IsEmpty)
            throw new ArgumentException("Basket must contain at least one leg.", nameof(weightedValues));

        return method switch
        {
            BasketAggregationMethod.WeightedSum => WeightedSum(weightedValues),
            BasketAggregationMethod.BestOf      => BestOf(weightedValues),
            BasketAggregationMethod.WorstOf     => WorstOf(weightedValues),
            _ => throw new ArgumentOutOfRangeException(
                    nameof(method), method, "Aggregation method must be specified (not Unspecified).")
        };
    }

    private static double WeightedSum(ReadOnlySpan<double> values)
    {
        var sum = 0.0;
        foreach (var v in values) sum += v;
        return sum;
    }

    private static double BestOf(ReadOnlySpan<double> values)
    {
        var max = values[0];
        for (var i = 1; i < values.Length; i++)
            if (values[i] > max) max = values[i];
        return max;
    }

    private static double WorstOf(ReadOnlySpan<double> values)
    {
        var min = values[0];
        for (var i = 1; i < values.Length; i++)
            if (values[i] < min) min = values[i];
        return min;
    }
}
