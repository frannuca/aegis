using MathNet.Numerics.Distributions;

namespace RandomSimulator;

/// <summary>
/// In-memory multi-asset correlated Brownian motion increments.
///
/// ── Stochastic process ────────────────────────────────────────────────────────
/// For each path n and time step t the cube stores a vector of increments:
///
///   dW[n, t, :] = L · dZ[n, t, :]
///
/// where:
///   dZ[n, t, i]  ~ N(0, 1)  i.i.d. across paths, steps, and assets i
///   L             = lower Cholesky factor of the instantaneous correlation ρ
///   ρ[i, j]       = corr(dW_i, dW_j)   (must be symmetric positive-definite)
///
/// ── Usage ─────────────────────────────────────────────────────────────────────
/// The cube stores unit-time increments. For a time step dt, the actual
/// Brownian increment for asset i is:
///
///   ΔW_i(t) = cube[path, step, i] · √dt
///
/// For GBM (Black-Scholes):
///   S_i(t+dt) = S_i(t) · exp((μ_i - ½σ_i²)·dt + σ_i·√dt·cube[path, step, i])
///
/// ── Sampling methods ─────────────────────────────────────────────────────────
/// PseudoRandom (default):
///   Box-Muller transform applied to System.Random (seeded for reproducibility).
///   Convergence O(1/√N). Unlimited paths and steps.
///
/// QuasiRandom (Sobol):
///   Joe-Kuo (2010) Sobol sequence → inverse normal CDF (MathNet.Numerics).
///   Convergence O((log N)^d / N) — dramatically faster for smooth payoffs.
///   Requires steps × assets ≤ SobolSequence.MaxDimensions (= 21 built-in).
///   The seed parameter is ignored; Sobol sequences are deterministic.
///   For more dimensions, call SobolSequence.SetAdditionalDirections().
///
/// ── Antithetic variates ───────────────────────────────────────────────────────
/// When useAntithetics = true (works with both methods):
///   - PseudoRandom: second half of paths = negatives of first half (standard).
///   - QuasiRandom:  for Sobol uniform u, the antithetic point uses 1-u,
///                   which maps to -Φ⁻¹(u) after the inverse CDF.
///   Paths must be even. Variance reduction is automatic; price by averaging all N paths.
///
/// ── Memory layout ────────────────────────────────────────────────────────────
/// Flat array, row-major by (path, step, asset):
///   index = path * Steps * Assets + step * Assets + asset
/// This is cache-optimal for step-by-step simulation (one path, step inner loop).
/// </summary>
public sealed class SimulationCube
{
    private readonly double[] _data;

    public int Paths  { get; }
    public int Steps  { get; }
    public int Assets { get; }

    /// <summary>The correlation matrix used to generate this cube (defensive copy).</summary>
    public double[,] Correlation { get; }

    /// <summary>Whether antithetic variates were used.</summary>
    public bool UsesAntithetics { get; }

    /// <summary>Which sampling method produced this cube.</summary>
    public SamplingMethod Method { get; }

    private SimulationCube(
        int            paths,
        int            steps,
        int            assets,
        double[,]      correlation,
        double[]       data,
        bool           antithetics,
        SamplingMethod method)
    {
        Paths           = paths;
        Steps           = steps;
        Assets          = assets;
        Correlation     = correlation;
        _data           = data;
        UsesAntithetics = antithetics;
        Method          = method;
    }

    // ── Accessors ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns the correlated N(0,1) increment for a given (path, step, asset).
    /// Multiply by √dt to obtain the actual Brownian increment.
    /// </summary>
    public double this[int path, int step, int asset]
        => _data[path * Steps * Assets + step * Assets + asset];

    /// <summary>
    /// Returns all asset increments for a given (path, step) as a contiguous span.
    /// Length equals <see cref="Assets"/>. Cache-friendly for hot pricing loops.
    /// </summary>
    public ReadOnlySpan<double> StepIncrements(int path, int step)
        => _data.AsSpan(path * Steps * Assets + step * Assets, Assets);

    // ── Factory ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Generates a new simulation cube.
    /// </summary>
    /// <param name="paths">Number of Monte Carlo paths. Must be even when useAntithetics=true.</param>
    /// <param name="steps">Number of time steps per path.</param>
    /// <param name="assets">Number of correlated assets.</param>
    /// <param name="correlation">
    ///   Instantaneous correlation matrix (assets × assets), symmetric positive-definite
    ///   with unit diagonal.
    /// </param>
    /// <param name="seed">
    ///   RNG seed for PseudoRandom reproducibility. Ignored for QuasiRandom (Sobol
    ///   sequences are deterministic from their direction numbers).
    /// </param>
    /// <param name="useAntithetics">
    ///   When true, the second half of paths are the negatives of the first half.
    ///   Requires paths to be even.
    /// </param>
    /// <param name="method">
    ///   <see cref="SamplingMethod.PseudoRandom"/> (Box-Muller, unlimited dimensions) or
    ///   <see cref="SamplingMethod.QuasiRandom"/> (Sobol, requires steps×assets ≤ 21).
    /// </param>
    public static SimulationCube Generate(
        int            paths,
        int            steps,
        int            assets,
        double[,]      correlation,
        int            seed           = 42,
        bool           useAntithetics = false,
        SamplingMethod method         = SamplingMethod.PseudoRandom)
    {
        ValidateDimensions(paths, steps, assets, useAntithetics);
        ValidateCorrelation(correlation, assets);

        var L        = Cholesky.Decompose(correlation);
        var corrCopy = (double[,])correlation.Clone();

        var data = method == SamplingMethod.QuasiRandom
            ? FillDataSobol(paths, steps, assets, L, useAntithetics)
            : FillDataPseudo(paths, steps, assets, L, seed, useAntithetics);

        return new SimulationCube(paths, steps, assets, corrCopy, data, useAntithetics, method);
    }

    /// <summary>
    /// Generates a cube with independent assets (identity correlation).
    /// </summary>
    public static SimulationCube GenerateIndependent(
        int            paths,
        int            steps,
        int            assets,
        int            seed           = 42,
        bool           useAntithetics = false,
        SamplingMethod method         = SamplingMethod.QuasiRandom)
    {
        var identity = Identity(assets);
        return Generate(paths, steps, assets, identity, seed, useAntithetics, method);
    }

    // ── Pseudo-random path (Box-Muller + System.Random) ───────────────────────

    private static double[] FillDataPseudo(
        int       paths,
        int       steps,
        int       assets,
        double[,] L,
        int       seed,
        bool      antithetics)
    {
        var data         = new double[paths * steps * assets];
        var rng          = new Random(seed);
        var rawZ         = new double[assets];
        var correlated   = new double[assets];
        var primaryPaths = antithetics ? paths / 2 : paths;

        for (var n = 0; n < primaryPaths; n++)
        {
            var baseOff   = n * steps * assets;
            var mirrorOff = antithetics ? (n + primaryPaths) * steps * assets : -1;

            for (var t = 0; t < steps; t++)
            {
                SampleBoxMuller(rng, rawZ, assets);
                ApplyCholesky(L, rawZ, correlated, assets);

                var stepOff = t * assets;
                for (var a = 0; a < assets; a++)
                {
                    var v = correlated[a];
                    data[baseOff + stepOff + a] = v;
                    if (antithetics)
                        data[mirrorOff + stepOff + a] = -v;
                }
            }
        }

        return data;
    }

    // ── Quasi-random path (Sobol + inverse normal CDF) ────────────────────────

    /// <summary>
    /// Fills the cube using a Sobol sequence.
    ///
    /// Dimensionality mapping: coordinate k = step × assets + asset.
    /// Each of the `paths` Monte Carlo paths uses one distinct Sobol point
    /// in [0,1)^{steps × assets}. The uniform coordinates are transformed to
    /// N(0,1) via the inverse CDF (MathNet.Numerics), then correlated via
    /// the Cholesky factor L.
    ///
    /// Why inverse CDF, not Box-Muller? Box-Muller pairs coordinates in polar
    /// form; applying it to Sobol numbers would destroy the low-discrepancy
    /// structure. The inverse CDF is a monotone bijection [0,1) → ℝ that
    /// preserves the equidistribution of the Sobol sequence.
    ///
    /// Antithetics with Sobol: for a Sobol point u ∈ (0,1)^d, the antithetic
    /// is Φ⁻¹(1-u) = -Φ⁻¹(u), i.e., flip u → 1-u before applying the CDF.
    /// </summary>
    private static double[] FillDataSobol(
        int       paths,
        int       steps,
        int       assets,
        double[,] L,
        bool      antithetics)
    {
        var dim = steps * assets;

        if (dim > SobolSequence.MaxDimensions)
            throw new ArgumentException(
                $"Sobol requires steps × assets ≤ {SobolSequence.MaxDimensions} dimensions, " +
                $"but got {steps} × {assets} = {dim}. " +
                $"Reduce steps/assets, switch to PseudoRandom, or call " +
                $"SobolSequence.SetAdditionalDirections() with Joe-Kuo data from " +
                $"https://web.maths.unsw.edu.au/~fkuo/sobol/");

        var data         = new double[paths * steps * assets];
        var primaryPaths = antithetics ? paths / 2 : paths;
        var sobol        = new SobolSequence(dim);
        var point        = new double[dim];
        var rawZ         = new double[assets];
        var correlated   = new double[assets];

        for (var n = 0; n < primaryPaths; n++)
        {
            sobol.NextPoint(point);

            var baseOff   = n * steps * assets;
            var mirrorOff = antithetics ? (n + primaryPaths) * steps * assets : -1;

            for (var t = 0; t < steps; t++)
            {
                // Convert each Sobol uniform to N(0,1) via inverse CDF.
                // Sobol uniform u ∈ (0,1) — clamp away from boundaries for
                // numerical stability of InvCDF (which diverges at 0 and 1).
                for (var a = 0; a < assets; a++)
                {
                    var u = Math.Clamp(point[t * assets + a], 1e-15, 1.0 - 1e-15);
                    rawZ[a] = Normal.InvCDF(0.0, 1.0, u);
                }

                ApplyCholesky(L, rawZ, correlated, assets);

                var stepOff = t * assets;
                for (var a = 0; a < assets; a++)
                {
                    var v = correlated[a];
                    data[baseOff + stepOff + a] = v;
                    if (antithetics)
                        data[mirrorOff + stepOff + a] = -v;
                }
            }
        }

        return data;
    }

    // ── Shared helpers ─────────────────────────────────────────────────────────

    // Box-Muller: fills buf[0..n-1] with i.i.d. N(0,1) samples.
    private static void SampleBoxMuller(Random rng, double[] buf, int n)
    {
        for (var i = 0; i < n - 1; i += 2)
        {
            double u1;
            do { u1 = rng.NextDouble(); } while (u1 <= double.Epsilon);
            var u2    = rng.NextDouble();
            var r     = Math.Sqrt(-2.0 * Math.Log(u1));
            var theta = 2.0 * Math.PI * u2;
            buf[i]     = r * Math.Cos(theta);
            buf[i + 1] = r * Math.Sin(theta);
        }
        if (n % 2 == 1)
        {
            double u1;
            do { u1 = rng.NextDouble(); } while (u1 <= double.Epsilon);
            var u2    = rng.NextDouble();
            buf[n - 1] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }

    // correlated[i] = Σ_j L[i,j] · z[j]  (L lower triangular)
    private static void ApplyCholesky(double[,] L, double[] z, double[] out_, int n)
    {
        for (var i = 0; i < n; i++)
        {
            var s = 0.0;
            for (var j = 0; j <= i; j++)
                s += L[i, j] * z[j];
            out_[i] = s;
        }
    }

    // ── Validation ─────────────────────────────────────────────────────────────

    private static void ValidateDimensions(int paths, int steps, int assets, bool antithetics)
    {
        if (paths  <= 0) throw new ArgumentOutOfRangeException(nameof(paths),  "Must be > 0.");
        if (steps  <= 0) throw new ArgumentOutOfRangeException(nameof(steps),  "Must be > 0.");
        if (assets <= 0) throw new ArgumentOutOfRangeException(nameof(assets), "Must be > 0.");
        if (antithetics && paths % 2 != 0)
            throw new ArgumentException("Paths must be even when useAntithetics=true.", nameof(paths));
    }

    private static void ValidateCorrelation(double[,] rho, int assets)
    {
        if (rho.GetLength(0) != assets || rho.GetLength(1) != assets)
            throw new ArgumentException(
                $"Correlation matrix must be {assets}×{assets} to match assets count.");

        for (var i = 0; i < assets; i++)
        {
            if (Math.Abs(rho[i, i] - 1.0) > 1e-10)
                throw new ArgumentException(
                    $"Correlation matrix diagonal [{i},{i}] must be 1.0 (got {rho[i, i]}).");

            for (var j = 0; j < assets; j++)
                if (rho[i, j] < -1.0 || rho[i, j] > 1.0)
                    throw new ArgumentException(
                        $"Correlation [{i},{j}] = {rho[i, j]} is outside [-1, 1].");
        }
    }

    private static double[,] Identity(int n)
    {
        var m = new double[n, n];
        for (var i = 0; i < n; i++) m[i, i] = 1.0;
        return m;
    }
}

/// <summary>Selects the random number generation strategy for <see cref="SimulationCube"/>.</summary>
public enum SamplingMethod
{
    /// <summary>
    /// Box-Muller transform on System.Random. Unlimited dimensions.
    /// Convergence O(1/√N).
    /// </summary>
    PseudoRandom,

    /// <summary>
    /// Joe-Kuo (2010) Sobol sequence with inverse normal CDF (MathNet.Numerics).
    /// Convergence O((log N)^d / N) — fewer paths needed for the same accuracy.
    /// Requires steps × assets ≤ SobolSequence.MaxDimensions (default: 21).
    /// The seed parameter is ignored.
    /// </summary>
    QuasiRandom
}
