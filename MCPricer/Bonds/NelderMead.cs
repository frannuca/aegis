namespace MCPricer.Bonds;

/// <summary>
/// Nelder-Mead downhill simplex minimizer (Nelder &amp; Mead, 1965) — a
/// derivative-free optimizer used to calibrate G2++ parameters to market
/// prices without adding an external optimization dependency.
///
/// Standard textbook coefficients: reflection α=1, expansion γ=2,
/// contraction ρ=0.5, shrink σ=0.5. These are the classical Nelder-Mead
/// defaults, not tunable magic constants — changing them changes the
/// algorithm's convergence behaviour, not the problem being solved.
///
/// Convergence: stops when the spread of objective values across the
/// simplex vertices falls below <paramref name="tolerance"/> (relative to
/// the best value), or after <paramref name="maxIterations"/> iterations —
/// whichever comes first. Callers must check <see cref="NelderMeadResult.Converged"/>
/// rather than assume success (no silent fallback).
/// </summary>
public static class NelderMead
{
    public static NelderMeadResult Minimize(
        Func<double[], double> objective,
        double[] initialGuess,
        double initialStep = 0.1,
        double tolerance = 1e-10,
        int maxIterations = 2000)
    {
        ArgumentNullException.ThrowIfNull(objective);
        ArgumentNullException.ThrowIfNull(initialGuess);
        var n = initialGuess.Length;
        if (n == 0)
            throw new ArgumentException("initialGuess must be non-empty.", nameof(initialGuess));

        // Build the initial simplex: the guess plus one vertex per dimension,
        // each perturbed by initialStep along that axis.
        var simplex = new double[n + 1][];
        var values  = new double[n + 1];
        simplex[0] = (double[])initialGuess.Clone();
        for (var i = 0; i < n; i++)
        {
            var vertex = (double[])initialGuess.Clone();
            vertex[i] += initialStep;
            simplex[i + 1] = vertex;
        }
        for (var i = 0; i <= n; i++)
            values[i] = objective(simplex[i]);

        const double alpha = 1.0;  // reflection
        const double gamma = 2.0;  // expansion
        const double rhoC  = 0.5;  // contraction
        const double sigma = 0.5;  // shrink

        var iterations = 0;
        for (; iterations < maxIterations; iterations++)
        {
            SortByValue(simplex, values);

            var best  = values[0];
            var worst = values[n];
            if (Math.Abs(worst - best) < tolerance * (1.0 + Math.Abs(best)))
                break;

            // Centroid of all vertices except the worst.
            var centroid = new double[n];
            for (var i = 0; i < n; i++)
            {
                for (var d = 0; d < n; d++)
                    centroid[d] += simplex[i][d];
            }
            for (var d = 0; d < n; d++)
                centroid[d] /= n;

            var reflected = Combine(centroid, simplex[n], 1.0 + alpha, -alpha);
            var fReflected = objective(reflected);

            if (fReflected < values[0])
            {
                var expanded  = Combine(centroid, simplex[n], 1.0 + gamma, -gamma);
                var fExpanded = objective(expanded);
                simplex[n] = fExpanded < fReflected ? expanded : reflected;
                values[n]  = Math.Min(fExpanded, fReflected);
            }
            else if (fReflected < values[n - 1])
            {
                simplex[n] = reflected;
                values[n]  = fReflected;
            }
            else
            {
                var contracted  = Combine(centroid, simplex[n], 1.0 - rhoC, rhoC);
                var fContracted = objective(contracted);
                if (fContracted < values[n])
                {
                    simplex[n] = contracted;
                    values[n]  = fContracted;
                }
                else
                {
                    for (var i = 1; i <= n; i++)
                    {
                        for (var d = 0; d < n; d++)
                            simplex[i][d] = simplex[0][d] + sigma * (simplex[i][d] - simplex[0][d]);
                        values[i] = objective(simplex[i]);
                    }
                }
            }
        }

        SortByValue(simplex, values);
        var spread = Math.Abs(values[n] - values[0]) / (1.0 + Math.Abs(values[0]));
        return new NelderMeadResult(simplex[0], values[0], iterations, spread < tolerance);
    }

    // x = w1*a + w2*b, componentwise.
    private static double[] Combine(double[] a, double[] b, double w1, double w2)
    {
        var result = new double[a.Length];
        for (var i = 0; i < a.Length; i++)
            result[i] = w1 * a[i] + w2 * b[i];
        return result;
    }

    private static void SortByValue(double[][] simplex, double[] values)
    {
        var order = Enumerable.Range(0, values.Length).ToArray();
        Array.Sort(order, (i, j) => values[i].CompareTo(values[j]));

        var sortedSimplex = order.Select(i => simplex[i]).ToArray();
        var sortedValues  = order.Select(i => values[i]).ToArray();
        Array.Copy(sortedSimplex, simplex, simplex.Length);
        Array.Copy(sortedValues, values, values.Length);
    }
}

/// <summary>Result of a Nelder-Mead minimization run.</summary>
public sealed record NelderMeadResult(
    double[] ArgMin,
    double   Value,
    int      Iterations,
    bool     Converged);
