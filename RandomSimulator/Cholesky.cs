namespace RandomSimulator;

/// <summary>
/// Cholesky-Banachiewicz decomposition: given a symmetric positive-definite matrix C,
/// computes lower triangular L such that C = L·Lᵀ.
///
/// Used to transform independent standard normals into correlated normals:
///   dW = L · dZ,  where dZ ~ N(0, I)  →  Cov(dW) = L·Lᵀ = C
/// </summary>
internal static class Cholesky
{
    /// <summary>
    /// Decomposes <paramref name="matrix"/> into its lower Cholesky factor.
    /// </summary>
    /// <returns>Lower-triangular L (n×n) such that L·Lᵀ = matrix.</returns>
    /// <exception cref="ArgumentException">
    /// When matrix is not square, not symmetric (within tolerance), or not positive definite.
    /// </exception>
    public static double[,] Decompose(double[,] matrix)
    {
        var n = matrix.GetLength(0);
        if (matrix.GetLength(1) != n)
            throw new ArgumentException("Correlation matrix must be square.");

        // Symmetry check
        for (var i = 0; i < n; i++)
        for (var j = i + 1; j < n; j++)
            if (Math.Abs(matrix[i, j] - matrix[j, i]) > 1e-10)
                throw new ArgumentException(
                    $"Correlation matrix is not symmetric at [{i},{j}]: " +
                    $"{matrix[i, j]} vs {matrix[j, i]}.");

        var L = new double[n, n];

        for (var j = 0; j < n; j++)
        {
            // Diagonal element
            var sum = matrix[j, j];
            for (var k = 0; k < j; k++)
                sum -= L[j, k] * L[j, k];

            if (sum <= 0.0)
                throw new ArgumentException(
                    $"Correlation matrix is not positive definite (failed at column {j}). " +
                    "Check that all correlations are in (-1,1) and the matrix has no " +
                    "linearly dependent rows.");

            L[j, j] = Math.Sqrt(sum);

            // Below-diagonal elements in column j
            var invDiag = 1.0 / L[j, j];
            for (var i = j + 1; i < n; i++)
            {
                var s = matrix[i, j];
                for (var k = 0; k < j; k++)
                    s -= L[i, k] * L[j, k];
                L[i, j] = s * invDiag;
            }
        }

        return L;
    }
}