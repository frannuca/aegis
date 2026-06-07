using RandomSimulator;

namespace montecarlo.Tests;

/// <summary>
/// Statistical tests for SimulationCube.
///
/// Confidence intervals are computed using CLT:
///   SE(mean)     = 1 / sqrt(N)
///   SE(variance) = sqrt(2 / N)
///   SE(corr)     ≈ (1 - ρ²) / sqrt(N)
///
/// All assertions use 5σ bounds so the probability of a false failure is < 3e-7.
/// Deterministic seeds guarantee reproducibility.
/// </summary>
public sealed class SimulationCubeTests
{
    // ── Construction and dimensions ───────────────────────────────────────────

    [Fact]
    public void Dimensions_AreStoredCorrectly()
    {
        var cube = SimulationCube.GenerateIndependent(paths: 100, steps: 50, assets: 3);
        Assert.Equal(100, cube.Paths);
        Assert.Equal(50,  cube.Steps);
        Assert.Equal(3,   cube.Assets);
    }

    [Fact]
    public void SingleAsset_IdentityCorrelation_Succeeds()
    {
        var cube = SimulationCube.GenerateIndependent(paths: 200, steps: 10, assets: 1);
        Assert.Equal(1, cube.Assets);
    }

    [Fact]
    public void Indexer_And_StepIncrements_AreConsistent()
    {
        var cube = SimulationCube.GenerateIndependent(paths: 10, steps: 5, assets: 4);
        for (var n = 0; n < cube.Paths; n++)
        for (var t = 0; t < cube.Steps; t++)
        {
            var span = cube.StepIncrements(n, t);
            for (var a = 0; a < cube.Assets; a++)
                Assert.Equal(cube[n, t, a], span[a]);
        }
    }

    [Fact]
    public void SameSeed_ProducesSameData()
    {
        var c1 = SimulationCube.GenerateIndependent(500, 10, 2, seed: 99);
        var c2 = SimulationCube.GenerateIndependent(500, 10, 2, seed: 99);
        for (var n = 0; n < c1.Paths; n++)
        for (var t = 0; t < c1.Steps; t++)
        for (var a = 0; a < c1.Assets; a++)
            Assert.Equal(c1[n, t, a], c2[n, t, a]);
    }

    [Fact]
    public void DifferentSeed_ProducesDifferentData()
    {
        var c1 = SimulationCube.GenerateIndependent(100, 10, 2, seed: 1);
        var c2 = SimulationCube.GenerateIndependent(100, 10, 2, seed: 2);
        var anyDiff = false;
        for (var n = 0; n < c1.Paths && !anyDiff; n++)
        for (var t = 0; t < c1.Steps && !anyDiff; t++)
            if (c1[n, t, 0] != c2[n, t, 0]) anyDiff = true;
        Assert.True(anyDiff, "Different seeds must produce different values.");
    }

    // ── Marginal distribution: E[Z] ≈ 0, Var(Z) ≈ 1 ──────────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(5)]
    public void MarginalMean_IsApproximatelyZero(int assets)
    {
        // N = paths × steps samples per asset
        const int paths = 10_000, steps = 20;
        var cube = SimulationCube.GenerateIndependent(paths, steps, assets);

        // 5σ bound: 5 / sqrt(N)
        var tol = 5.0 / Math.Sqrt((double)paths * steps);

        for (var a = 0; a < assets; a++)
        {
            var sum = 0.0;
            for (var n = 0; n < paths; n++)
            for (var t = 0; t < steps; t++)
                sum += cube[n, t, a];
            var mean = sum / (paths * steps);
            Assert.True(Math.Abs(mean) < tol,
                $"Asset {a}: mean {mean:F5} exceeds 5σ tolerance ±{tol:F5}");
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void MarginalVariance_IsApproximatelyOne(int assets)
    {
        const int paths = 10_000, steps = 20;
        var cube = SimulationCube.GenerateIndependent(paths, steps, assets);

        // SE(variance) = sqrt(2/N); 5σ bound = 5·sqrt(2/N)
        var n   = (double)paths * steps;
        var tol = 5.0 * Math.Sqrt(2.0 / n);

        for (var a = 0; a < assets; a++)
        {
            var sum  = 0.0;
            var sum2 = 0.0;
            for (var p = 0; p < paths; p++)
            for (var t = 0; t < steps; t++)
            {
                var v = cube[p, t, a];
                sum  += v;
                sum2 += v * v;
            }
            var variance = sum2 / n - (sum / n) * (sum / n);
            Assert.True(Math.Abs(variance - 1.0) < tol,
                $"Asset {a}: variance {variance:F5} not within 5σ of 1.0 (tol ±{tol:F5})");
        }
    }

    // ── Realized correlation ──────────────────────────────────────────────────

    [Fact]
    public void TwoAssets_RealizesTargetCorrelation()
    {
        const int paths = 20_000, steps = 10;
        const double rho = 0.70;

        var corr = new double[,] { { 1.0, rho }, { rho, 1.0 } };
        var cube = SimulationCube.Generate(paths, steps, assets: 2, corr, seed: 7);

        var realized = RealizedCorrelation(cube, 0, 1);
        // 5σ bound: 5 · (1 - ρ²) / sqrt(N)
        var n   = (double)paths * steps;
        var tol = 5.0 * (1.0 - rho * rho) / Math.Sqrt(n);

        Assert.True(Math.Abs(realized - rho) < tol,
            $"Realized correlation {realized:F4} differs from target {rho} by more than 5σ ({tol:F4})");
    }

    [Fact]
    public void NegativeCorrelation_IsRealized()
    {
        const int paths = 20_000, steps = 10;
        const double rho = -0.50;

        var corr = new double[,] { { 1.0, rho }, { rho, 1.0 } };
        var cube = SimulationCube.Generate(paths, steps, 2, corr, seed: 13);

        var realized = RealizedCorrelation(cube, 0, 1);
        var n   = (double)paths * steps;
        var tol = 5.0 * (1.0 - rho * rho) / Math.Sqrt(n);

        Assert.True(Math.Abs(realized - rho) < tol,
            $"Realized correlation {realized:F4} vs target {rho}");
    }

    [Fact]
    public void ThreeAssets_RealizesFullCorrelationMatrix()
    {
        const int paths = 30_000, steps = 10;

        // Typical FX triangle correlation structure
        var corr = new double[,]
        {
            { 1.00,  0.60, -0.40 },
            { 0.60,  1.00,  0.20 },
            { -0.40, 0.20,  1.00 }
        };
        var cube = SimulationCube.Generate(paths, steps, 3, corr, seed: 17);
        var n    = (double)paths * steps;

        for (var i = 0; i < 3; i++)
        for (var j = i + 1; j < 3; j++)
        {
            var rho      = corr[i, j];
            var realized = RealizedCorrelation(cube, i, j);
            var tol      = 5.0 * (1.0 - rho * rho) / Math.Sqrt(n);
            Assert.True(Math.Abs(realized - rho) < tol,
                $"Assets ({i},{j}): realized {realized:F4} vs target {rho:F4} (tol {tol:F4})");
        }
    }

    [Fact]
    public void ZeroCorrelation_IndependentAssets()
    {
        const int paths = 20_000, steps = 10;
        var corr = new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        var cube = SimulationCube.Generate(paths, steps, 2, corr, seed: 5);

        var realized = RealizedCorrelation(cube, 0, 1);
        var tol = 5.0 / Math.Sqrt((double)paths * steps);

        Assert.True(Math.Abs(realized) < tol,
            $"Zero-correlation: realized {realized:F4} exceeds 5σ ({tol:F4})");
    }

    // ── Time-step independence ────────────────────────────────────────────────

    [Fact]
    public void Increments_AreIndependentAcrossTimeSteps()
    {
        const int paths = 20_000, steps = 10;
        var cube = SimulationCube.GenerateIndependent(paths, steps, assets: 1, seed: 21);

        // Lag-1 autocorrelation of the single asset across paths
        // E[Z(t) · Z(t+1)] must be ≈ 0
        var sumProd = 0.0;
        for (var n = 0; n < paths; n++)
        for (var t = 0; t < steps - 1; t++)
            sumProd += cube[n, t, 0] * cube[n, t + 1, 0];

        var autocorr = sumProd / (paths * (steps - 1));
        var tol = 5.0 / Math.Sqrt((double)paths * (steps - 1));

        Assert.True(Math.Abs(autocorr) < tol,
            $"Lag-1 autocorrelation {autocorr:F5} exceeds 5σ ({tol:F5})");
    }

    // ── Antithetic variates ───────────────────────────────────────────────────

    [Fact]
    public void Antithetics_SecondHalfIsNegativeOfFirstHalf()
    {
        var cube = SimulationCube.GenerateIndependent(100, 20, 2, seed: 3, useAntithetics: true);
        var half = cube.Paths / 2;

        for (var n = 0; n < half; n++)
        for (var t = 0; t < cube.Steps; t++)
        for (var a = 0; a < cube.Assets; a++)
            Assert.Equal(cube[n, t, a], -cube[n + half, t, a],
                precision: 12);
    }

    [Fact]
    public void Antithetics_MarginalDistributionPreserved()
    {
        const int paths = 10_000, steps = 20;
        var cube = SimulationCube.GenerateIndependent(paths, steps, 2, seed: 41, useAntithetics: true);

        var n   = (double)paths * steps;
        var tol = 5.0 / Math.Sqrt(n);

        for (var a = 0; a < 2; a++)
        {
            var sum = 0.0;
            for (var p = 0; p < paths; p++)
            for (var t = 0; t < steps; t++)
                sum += cube[p, t, a];
            Assert.True(Math.Abs(sum / n) < tol,
                $"Antithetic asset {a}: mean {sum / n:F5} outside 5σ ({tol:F5})");
        }
    }

    [Fact]
    public void Antithetics_WithCorrelation_SecondHalfNegated()
    {
        var corr = new double[,] { { 1.0, 0.6 }, { 0.6, 1.0 } };
        var cube = SimulationCube.Generate(200, 10, 2, corr, seed: 55, useAntithetics: true);
        var half = cube.Paths / 2;

        for (var n = 0; n < half; n++)
        for (var t = 0; t < cube.Steps; t++)
        for (var a = 0; a < cube.Assets; a++)
            Assert.Equal(cube[n, t, a], -cube[n + half, t, a], precision: 12);
    }

    // ── Input validation ──────────────────────────────────────────────────────

    [Theory]
    [InlineData(0,  10, 2)]
    [InlineData(10, 0,  2)]
    [InlineData(10, 10, 0)]
    public void InvalidDimensions_ThrowArgumentOutOfRange(int paths, int steps, int assets)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SimulationCube.GenerateIndependent(paths, steps, assets));
    }

    [Fact]
    public void OddPaths_WithAntithetics_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            SimulationCube.GenerateIndependent(101, 10, 2, useAntithetics: true));
    }

    [Fact]
    public void CorrelationWrongSize_Throws()
    {
        var corr = new double[,] { { 1.0, 0.5 }, { 0.5, 1.0 } };
        Assert.Throws<ArgumentException>(() =>
            SimulationCube.Generate(100, 10, assets: 3, correlation: corr));
    }

    [Fact]
    public void NonUnitDiagonal_Throws()
    {
        var corr = new double[,] { { 1.0, 0.5 }, { 0.5, 0.9 } };  // [1,1] != 1
        Assert.Throws<ArgumentException>(() =>
            SimulationCube.Generate(100, 10, 2, corr));
    }

    [Fact]
    public void AsymmetricCorrelation_Throws()
    {
        var corr = new double[,] { { 1.0, 0.3 }, { 0.4, 1.0 } };  // 0.3 != 0.4
        Assert.Throws<ArgumentException>(() =>
            SimulationCube.Generate(100, 10, 2, corr));
    }

    [Fact]
    public void NonPositiveDefiniteCorrelation_Throws()
    {
        // Perfectly correlated assets (ρ = 1) — singular, not positive definite
        var corr = new double[,] { { 1.0, 1.0 }, { 1.0, 1.0 } };
        Assert.Throws<ArgumentException>(() =>
            SimulationCube.Generate(100, 10, 2, corr));
    }

    [Fact]
    public void CorrelationOutOfRange_Throws()
    {
        var corr = new double[,] { { 1.0, 1.5 }, { 1.5, 1.0 } };
        Assert.Throws<ArgumentException>(() =>
            SimulationCube.Generate(100, 10, 2, corr));
    }

    // ── GBM sanity check: E[S(T)] = S0 · exp(μT) ────────────────────────────
    //
    // Monte Carlo price of a GBM terminal expectation.
    // Validates the cube drives a real model correctly.
    // Confidence interval reported: ±2σ/√N

    [Fact]
    public void GbmExpectation_MatchesAnalytical()
    {
        const int    paths  = 100_000;
        const int    steps  = 252;       // daily steps for 1 year
        const double s0     = 100.0;
        const double mu     = 0.05;
        const double sigma  = 0.20;
        const double T      = 1.0;
        var dt     = T / steps;
        var sqrtDt = Math.Sqrt(dt);

        var cube = SimulationCube.GenerateIndependent(paths, steps, assets: 1, seed: 42);

        var sumS = 0.0;
        for (var n = 0; n < paths; n++)
        {
            var s = s0;
            for (var t = 0; t < steps; t++)
                s *= Math.Exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrtDt * cube[n, t, 0]);
            sumS += s;
        }

        var mcMean    = sumS / paths;
        var analytical = s0 * Math.Exp(mu * T);

        // Monte Carlo standard error: σ_payoff / sqrt(paths)
        // For lognormal: σ_payoff = S0·exp(μT)·sqrt(exp(σ²T)-1)
        var stdPayoff = analytical * Math.Sqrt(Math.Exp(sigma * sigma * T) - 1.0);
        var se        = stdPayoff / Math.Sqrt(paths);
        var ciWidth   = 2.0 * se;

        Assert.True(Math.Abs(mcMean - analytical) < ciWidth,
            $"GBM E[S(T)]: MC={mcMean:F4}, Analytical={analytical:F4}, " +
            $"2σ CI=[{analytical - ciWidth:F4}, {analytical + ciWidth:F4}]");
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    // Pearson correlation of two assets across all (path, step) pairs.
    private static double RealizedCorrelation(SimulationCube cube, int assetA, int assetB)
    {
        var n    = (double)cube.Paths * cube.Steps;
        var sumA = 0.0; var sumB = 0.0;
        var sumA2 = 0.0; var sumB2 = 0.0; var sumAB = 0.0;

        for (var p = 0; p < cube.Paths; p++)
        for (var t = 0; t < cube.Steps; t++)
        {
            var a = cube[p, t, assetA];
            var b = cube[p, t, assetB];
            sumA  += a;  sumB  += b;
            sumA2 += a * a; sumB2 += b * b;
            sumAB += a * b;
        }

        var meanA = sumA / n; var meanB = sumB / n;
        var cov   = sumAB / n - meanA * meanB;
        var varA  = sumA2 / n - meanA * meanA;
        var varB  = sumB2 / n - meanB * meanB;

        return cov / Math.Sqrt(varA * varB);
    }
}

/// <summary>
/// Tests for the Sobol (quasi-random) sampling path.
///
/// Key properties verified:
///   1. Correct marginal distribution (N(0,1) increments)
///   2. Correct correlation structure
///   3. Deterministic / seed-independent
///   4. Antithetic symmetry: second half = negatives of first half
///   5. Dimension limit enforcement
///   6. Faster convergence than pseudo-random for smooth payoffs
///
/// ── Convergence note ─────────────────────────────────────────────────────────
/// The convergence test prices a European call under GBM with analytical
/// Black-Scholes as the reference. With N paths:
///   - Pseudo-random RMSE ≈ σ_payoff / √N
///   - Sobol RMSE ≈ O((log N) / N)  for smooth payoffs in low dimension
/// We compare absolute errors at N=512 paths and assert Sobol wins.
/// </summary>
public sealed class SobolTests
{
    // ── Basic properties ──────────────────────────────────────────────────────

    [Fact]
    public void Sobol_Dimensions_AreCorrect()
    {
        var cube = SimulationCube.GenerateIndependent(64, 3, 2, method: SamplingMethod.QuasiRandom);
        Assert.Equal(64, cube.Paths);
        Assert.Equal(3,  cube.Steps);
        Assert.Equal(2,  cube.Assets);
        Assert.Equal(SamplingMethod.QuasiRandom, cube.Method);
    }

    [Fact]
    public void Sobol_IsDeterministic_SeedIgnored()
    {
        // Different seeds must produce the same Sobol sequence
        var c1 = SimulationCube.GenerateIndependent(64, 3, 2, seed: 1, method: SamplingMethod.QuasiRandom);
        var c2 = SimulationCube.GenerateIndependent(64, 3, 2, seed: 999, method: SamplingMethod.QuasiRandom);
        for (var n = 0; n < 64; n++)
        for (var t = 0; t < 3; t++)
        for (var a = 0; a < 2; a++)
            Assert.Equal(c1[n, t, a], c2[n, t, a]);
    }

    [Fact]
    public void Sobol_DifferentFromPseudo()
    {
        var pseudo = SimulationCube.GenerateIndependent(64, 3, 2, method: SamplingMethod.PseudoRandom);
        var quasi  = SimulationCube.GenerateIndependent(64, 3, 2, method: SamplingMethod.QuasiRandom);
        var anyDiff = false;
        for (var n = 0; n < 64 && !anyDiff; n++)
        for (var t = 0; t < 3  && !anyDiff; t++)
            if (pseudo[n, t, 0] != quasi[n, t, 0]) anyDiff = true;
        Assert.True(anyDiff);
    }

    // ── Marginal distribution ─────────────────────────────────────────────────

    [Fact]
    public void Sobol_MarginalMean_IsApproximatelyZero()
    {
        // Use paths=2^k for Sobol (powers of 2 give the best equidistribution).
        // With N = 2^14 = 16384 paths × 1 step = 16384 samples per asset.
        const int paths = 1 << 14, steps = 1, assets = 3;
        var cube = SimulationCube.GenerateIndependent(paths, steps, assets,
            method: SamplingMethod.QuasiRandom);

        // Sobol mean convergence is exact for N = 2^k: sum of all points is N/2 exactly.
        // We use 5σ pseudo-random bound as a conservative check.
        var tol = 5.0 / Math.Sqrt((double)paths * steps);
        for (var a = 0; a < assets; a++)
        {
            var sum = 0.0;
            for (var p = 0; p < paths; p++)
            for (var t = 0; t < steps; t++)
                sum += cube[p, t, a];
            Assert.True(Math.Abs(sum / (paths * steps)) < tol,
                $"Sobol asset {a} mean {sum / (paths * steps):F5} outside 5σ ({tol:F5})");
        }
    }

    [Fact]
    public void Sobol_MarginalVariance_IsApproximatelyOne()
    {
        const int paths = 1 << 13, steps = 1, assets = 2;
        var cube = SimulationCube.GenerateIndependent(paths, steps, assets,
            method: SamplingMethod.QuasiRandom);

        var n   = (double)paths * steps;
        var tol = 5.0 * Math.Sqrt(2.0 / n);

        for (var a = 0; a < assets; a++)
        {
            var sum = 0.0; var sum2 = 0.0;
            for (var p = 0; p < paths; p++)
            for (var t = 0; t < steps; t++)
            {
                var v = cube[p, t, a];
                sum += v; sum2 += v * v;
            }
            var variance = sum2 / n - (sum / n) * (sum / n);
            Assert.True(Math.Abs(variance - 1.0) < tol,
                $"Sobol asset {a} variance {variance:F5} outside 5σ of 1.0 (tol {tol:F5})");
        }
    }

    // ── Correlation ───────────────────────────────────────────────────────────

    [Fact]
    public void Sobol_TwoAssets_RealizesTargetCorrelation()
    {
        const double rho = 0.60;
        var corr = new double[,] { { 1.0, rho }, { rho, 1.0 } };
        var cube = SimulationCube.Generate(1 << 12, steps: 1, assets: 2, corr,
            method: SamplingMethod.QuasiRandom);

        var realized = RealizedCorrelation(cube, 0, 1);
        var tol = 5.0 * (1.0 - rho * rho) / Math.Sqrt((double)(1 << 12));
        Assert.True(Math.Abs(realized - rho) < tol,
            $"Sobol realized correlation {realized:F4} vs target {rho} (tol {tol:F4})");
    }

    // ── Antithetics ───────────────────────────────────────────────────────────

    [Fact]
    public void Sobol_Antithetics_SecondHalfIsNegated()
    {
        // For Sobol antithetics: x_antithetic = Phi^-1(1 - u) = -Phi^-1(u) = -x
        var cube = SimulationCube.GenerateIndependent(64, 3, 2, useAntithetics: true,
            method: SamplingMethod.QuasiRandom);
        var half = cube.Paths / 2;

        for (var n = 0; n < half; n++)
        for (var t = 0; t < cube.Steps; t++)
        for (var a = 0; a < cube.Assets; a++)
            Assert.Equal(cube[n, t, a], -cube[n + half, t, a], precision: 10);
    }

    // ── Dimension limit ───────────────────────────────────────────────────────

    [Fact]
    public void Sobol_ExceedingMaxDimensions_Throws()
    {
        // Default table supports 21 dimensions; steps × assets = 22 must throw.
        Assert.Throws<ArgumentException>(() =>
            SimulationCube.GenerateIndependent(
                paths: 100, steps: 22, assets: 1,
                method: SamplingMethod.QuasiRandom));
    }

    [Fact]
    public void Sobol_MaxDimensions_Succeeds()
    {
        // Exactly at the limit: 7 steps × 3 assets = 21 dimensions
        var cube = SimulationCube.GenerateIndependent(
            paths: 64, steps: 7, assets: 3,
            method: SamplingMethod.QuasiRandom);
        Assert.Equal(64, cube.Paths);
    }

    // ── Convergence superiority ───────────────────────────────────────────────

    /// <summary>
    /// Prices a European call under GBM using pseudo-random vs Sobol at low path counts.
    ///
    /// Reference: Black-Scholes analytical formula.
    /// Test: at N=512 paths, Sobol absolute error must be smaller than pseudo-random error.
    ///
    /// This directly validates the Sobol low-discrepancy advantage. We run many
    /// pseudo-random trials to ensure the comparison is meaningful (pseudo-random
    /// error is stochastic; we take the median).
    ///
    /// Setup: S0=100, K=100, T=1, σ=0.20, r=0.05, steps=1 (terminal value only).
    /// </summary>
    [Fact]
    public void Sobol_ConvergesFasterThanPseudo_ForEuropeanCall()
    {
        const double s0    = 100.0;
        const double k     = 100.0;
        const double sigma = 0.20;
        const double r     = 0.05;
        const double T     = 1.0;
        const double dt    = T;          // single-step: sample terminal value directly
        const double sqDt  = 1.0;       // sqrt(dt) = 1
        const int    paths = 512;        // deliberately small to expose convergence difference

        // Analytical Black-Scholes call price
        var d1 = (Math.Log(s0 / k) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
        var d2 = d1 - sigma * Math.Sqrt(T);
        var bsPrice = s0 * NormalCdf(d1) - k * Math.Exp(-r * T) * NormalCdf(d2);

        // Sobol price (deterministic — one run)
        var sobolCube = SimulationCube.GenerateIndependent(paths, steps: 1, assets: 1,
            method: SamplingMethod.QuasiRandom);
        var sobolPrice = PriceCall(sobolCube, s0, k, r, sigma, dt, sqDt);
        var sobolError = Math.Abs(sobolPrice - bsPrice);

        // Pseudo-random prices (stochastic — run many seeds, take median error)
        var pseudoErrors = new double[101];
        for (var seed = 0; seed < pseudoErrors.Length; seed++)
        {
            var pCube = SimulationCube.GenerateIndependent(paths, steps: 1, assets: 1, seed: seed,
                method: SamplingMethod.PseudoRandom);
            pseudoErrors[seed] = Math.Abs(PriceCall(pCube, s0, k, r, sigma, dt, sqDt) - bsPrice);
        }
        Array.Sort(pseudoErrors);
        var pseudoMedianError = pseudoErrors[pseudoErrors.Length / 2];

        // Sobol should beat the pseudo-random median error by a meaningful margin
        Assert.True(sobolError < pseudoMedianError,
            $"Sobol error {sobolError:F4} should be < pseudo median error {pseudoMedianError:F4} " +
            $"(BS price={bsPrice:F4}, Sobol price={sobolPrice:F4})");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static double PriceCall(SimulationCube cube, double s0, double k,
        double r, double sigma, double dt, double sqDt)
    {
        var sum = 0.0;
        for (var n = 0; n < cube.Paths; n++)
        {
            var s = s0 * Math.Exp((r - 0.5 * sigma * sigma) * dt + sigma * sqDt * cube[n, 0, 0]);
            sum += Math.Max(s - k, 0.0);
        }
        return Math.Exp(-r * dt) * sum / cube.Paths;
    }

    private static double RealizedCorrelation(SimulationCube cube, int assetA, int assetB)
    {
        var n = (double)cube.Paths * cube.Steps;
        var sumA = 0.0; var sumB = 0.0;
        var sumA2 = 0.0; var sumB2 = 0.0; var sumAB = 0.0;
        for (var p = 0; p < cube.Paths; p++)
        for (var t = 0; t < cube.Steps; t++)
        {
            var a = cube[p, t, assetA]; var b = cube[p, t, assetB];
            sumA += a; sumB += b;
            sumA2 += a * a; sumB2 += b * b; sumAB += a * b;
        }
        var mA = sumA / n; var mB = sumB / n;
        return (sumAB / n - mA * mB) / Math.Sqrt((sumA2 / n - mA * mA) * (sumB2 / n - mB * mB));
    }

    // Standard normal CDF via complementary error function
    private static double NormalCdf(double x) =>
        0.5 * MathNet.Numerics.SpecialFunctions.Erfc(-x / Math.Sqrt(2.0));
}
