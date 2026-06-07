namespace MCPricer;

/// <summary>
/// Result of a Monte Carlo pricing run.
///
/// Price is the empirical mean of the discounted payoff over all paths.
/// StandardError is the standard error of that mean: s / √N where s is the
/// sample standard deviation. The 95% confidence interval uses z = 1.96.
/// </summary>
public sealed record PricingResult(
    double Price,
    double StandardError,
    double ConfidenceIntervalLower,
    double ConfidenceIntervalUpper,
    int Paths)
{
    public double ConfidenceIntervalWidth => ConfidenceIntervalUpper - ConfidenceIntervalLower;

    public override string ToString() =>
        $"Price={Price:F6}  SE={StandardError:F6}  " +
        $"95% CI=[{ConfidenceIntervalLower:F6}, {ConfidenceIntervalUpper:F6}]  " +
        $"Paths={Paths}";
}
