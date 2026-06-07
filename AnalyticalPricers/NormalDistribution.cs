using MathNet.Numerics;

namespace AnalyticalPricers;

/// <summary>
/// Standard normal distribution helpers shared by every closed-form pricer in
/// this library (Garman-Kohlhagen, Black-Scholes, Bachelier, Hagan SABR, ...).
///
/// Cdf is evaluated via the complementary error function for full double
/// precision (no series-approximation error), matching the convention used
/// throughout this library and its analytical benchmark tests.
/// </summary>
public static class NormalDistribution
{
    private static readonly double InvSqrtTwoPi = 1.0 / Math.Sqrt(2.0 * Math.PI);

    /// <summary>Standard normal cumulative distribution function Φ(x) = ½·erfc(−x/√2).</summary>
    public static double Cdf(double x) => 0.5 * SpecialFunctions.Erfc(-x / Math.Sqrt(2.0));

    /// <summary>Standard normal probability density function φ(x) = exp(−x²/2)/√(2π).</summary>
    public static double Pdf(double x) => InvSqrtTwoPi * Math.Exp(-0.5 * x * x);
}
