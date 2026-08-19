using Aegis.Instruments;
using NodaTime;
using RandomSimulator;

namespace MCPricer.Bonds;

/// <summary>
/// Monte Carlo pricer for a fixed-rate callable bond under the two-factor
/// Hull-White (G2++) short-rate model, using Longstaff-Schwartz (2001)
/// regression to value the issuer's early-redemption option.
///
/// ── Stochastic process (risk-neutral measure Q, money-market numeraire) ───────
///   dx(t) = -A1 x(t) dt + Sigma1 dW1(t),  x(0)=0
///   dy(t) = -A2 y(t) dt + Sigma2 dW2(t),  y(0)=0
///   dW1 dW2 = Rho dt
///   r(t)  = x(t) + y(t) + φ(t)            (see <see cref="G2ppAnalytics.Phi"/>)
///   B(t)  = exp(∫₀ᵗ r(s) ds)               (money-market account)
///   Price = E^Q[ Σ_i CF_i / B(t_i) ]
///
/// x, y are simulated exactly (their OU transition is Gaussian in closed
/// form); ∫r ds is approximated by the trapezoidal rule on the simulation
/// grid, so discounting is exact only in the dt→0 limit — use enough fine
/// steps per year (recommended: >= 50) for coupon/annual-scale bonds.
///
/// ── Early-redemption option (Longstaff-Schwartz) ──────────────────────────────
/// At each call date the issuer redeems at CallPrice·Notional whenever doing
/// so is cheaper than the regression-estimated continuation value (a
/// polynomial in the state (x,y): 1, x, y, x², y², xy). The *realized* path
/// value (not the regression estimate) is carried backward when the issuer
/// does not call, per Longstaff-Schwartz — this avoids look-ahead bias.
///
/// Known limitation: like all LSM implementations, using the same paths for
/// both the regression fit and the valuation introduces a small, well-documented
/// upward bias in the price (the in-sample regression "sees" the payoff it is
/// trying to predict). This is not corrected here; a bias-free estimate would
/// require an independent out-of-sample path set for the final valuation pass.
///
/// ── Benchmark ──────────────────────────────────────────────────────────────────
/// <see cref="AnalyticalBulletBondPrice"/> gives the model-independent value of
/// the bond ignoring call optionality (a straight sum of curve discount
/// factors). Because φ(t) is calibrated so the model reprices the input curve
/// exactly, an empty call schedule (or call prices high enough that the
/// option is never exercised) must make the Monte Carlo price converge to
/// this benchmark — this is the primary numerical sanity check for the engine.
/// </summary>
public sealed class CallableBondMCPricer
{
    private readonly CallableBondDefinition _bond;
    private readonly LocalDate _valuationDate;
    private readonly Pillars _curve;
    private readonly G2ppParameters _model;
    private readonly SimulationCube _cube;

    private readonly EventNode[] _events;
    private readonly int[] _eventGridIndex;
    private readonly double[] _phi;
    private readonly double _dt;

    public CallableBondMCPricer(
        CallableBondDefinition bond,
        LocalDate valuationDate,
        Pillars discountCurve,
        G2ppParameters model,
        SimulationCube cube)
    {
        ArgumentNullException.ThrowIfNull(bond);
        ArgumentNullException.ThrowIfNull(discountCurve);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(cube);
        if (bond.MaturityDate <= valuationDate)
            throw new ArgumentException("Bond must mature after the valuation date.", nameof(bond));
        if (cube.Assets != 2)
            throw new ArgumentException(
                "SimulationCube must have exactly 2 assets (the G2++ x and y factors).", nameof(cube));
        if (Math.Abs(cube.Correlation[0, 1] - model.Rho) > 1e-9)
            throw new ArgumentException(
                $"SimulationCube correlation[0,1]={cube.Correlation[0, 1]:F6} must match " +
                $"the model's Rho={model.Rho:F6} — the cube's built-in Cholesky correlation " +
                "is what correlates the two OU factors' driving noise.",
                nameof(cube));

        _bond          = bond;
        _valuationDate = valuationDate;
        _curve         = discountCurve;
        _model         = model;
        _cube          = cube;

        (_events, _eventGridIndex, _dt) = BuildEventGrid(bond, valuationDate, cube.Steps);

        _phi = new double[cube.Steps + 1];
        for (var i = 0; i <= cube.Steps; i++)
            _phi[i] = G2ppAnalytics.Phi(discountCurve, valuationDate, model, i * _dt);
    }

    /// <summary>
    /// Prices the callable bond by simulating the two G2++ factors forward
    /// across all paths, then applying Longstaff-Schwartz backward induction
    /// over the merged coupon/call event timeline.
    /// </summary>
    public PricingResult Price()
    {
        var paths      = _cube.Paths;
        var numEvents  = _events.Length;
        var xAtEvent   = new double[paths, numEvents];
        var yAtEvent   = new double[paths, numEvents];
        var dAtEvent   = new double[paths, numEvents];

        SimulateForward(xAtEvent, yAtEvent, dAtEvent);

        var pathPV = new double[paths];
        for (var k = numEvents - 1; k >= 0; k--)
        {
            var node = _events[k];

            if (node.IsCallable)
            {
                var coeffs = RegressContinuation(xAtEvent, yAtEvent, pathPV, k, paths);
                for (var p = 0; p < paths; p++)
                {
                    var x = xAtEvent[p, k];
                    var y = yAtEvent[p, k];
                    var continuationEstimate =
                        coeffs[0] + coeffs[1] * x + coeffs[2] * y + coeffs[3] * x * x + coeffs[4] * y * y + coeffs[5] * x * y;
                    var callValuePv = node.CallPrice * _bond.Notional * dAtEvent[p, k];

                    if (callValuePv < continuationEstimate)
                        pathPV[p] = callValuePv;
                }
            }

            if (node.CashflowAmount != 0.0)
            {
                for (var p = 0; p < paths; p++)
                    pathPV[p] += node.CashflowAmount * dAtEvent[p, k];
            }
        }

        return BuildPricingResult(pathPV);
    }

    /// <summary>
    /// DV01: sensitivity to a 1bp parallel shift of the discount curve,
    /// computed by central finite difference bump-and-reprice on the same
    /// SimulationCube (common random numbers reduce the finite-difference
    /// variance). Note the curve shift also perturbs φ(t) — recomputed
    /// inside each bumped pricer's constructor — so this correctly captures
    /// both the discounting and the model-drift sensitivity to the curve.
    /// </summary>
    public double ComputeDv01(double eps = 0.0001)
    {
        var up = new CallableBondMCPricer(_bond, _valuationDate, ZeroCurve.Shift(_curve, +eps), _model, _cube).Price().Price;
        var dn = new CallableBondMCPricer(_bond, _valuationDate, ZeroCurve.Shift(_curve, -eps), _model, _cube).Price().Price;
        return (up - dn) / (2.0 * eps);
    }

    /// <summary>
    /// Closed-form value of the bond ignoring call optionality — the sum of
    /// each cashflow discounted off the input curve. Model-independent (any
    /// arbitrage-free short-rate model calibrated to this curve must agree
    /// with it once vol → 0 or the call is never in the money); see the
    /// class-level "Benchmark" note.
    /// </summary>
    public static double AnalyticalBulletBondPrice(
        CallableBondDefinition bond, LocalDate valuationDate, Pillars discountCurve)
    {
        var cashflows = bond.BuildCashflowSchedule(valuationDate);
        var price = 0.0;
        foreach (var cf in cashflows)
            price += cf.Amount * ZeroCurve.DiscountFactor(discountCurve, valuationDate, cf.YearsFromValuation);
        return price;
    }

    // ── Forward simulation ─────────────────────────────────────────────────────

    private void SimulateForward(double[,] xAtEvent, double[,] yAtEvent, double[,] dAtEvent)
    {
        var a1 = _model.A1; var s1 = _model.Sigma1;
        var a2 = _model.A2; var s2 = _model.Sigma2;
        var dt = _dt;
        var steps = _cube.Steps;

        var decay1 = Math.Exp(-a1 * dt);
        var decay2 = Math.Exp(-a2 * dt);
        var vol1 = s1 * Math.Sqrt((1.0 - Math.Exp(-2.0 * a1 * dt)) / (2.0 * a1));
        var vol2 = s2 * Math.Sqrt((1.0 - Math.Exp(-2.0 * a2 * dt)) / (2.0 * a2));

        Parallel.For(0, _cube.Paths, path =>
        {
            var x = 0.0;
            var y = 0.0;
            var discount = 1.0;
            var rPrev = _phi[0];
            var eventPtr = 0;

            for (var i = 1; i <= steps; i++)
            {
                var z1 = _cube[path, i - 1, 0];
                var z2 = _cube[path, i - 1, 1];
                x = x * decay1 + vol1 * z1;
                y = y * decay2 + vol2 * z2;

                var rCur = x + y + _phi[i];
                discount *= Math.Exp(-0.5 * (rPrev + rCur) * dt);
                rPrev = rCur;

                while (eventPtr < _eventGridIndex.Length && _eventGridIndex[eventPtr] == i)
                {
                    xAtEvent[path, eventPtr] = x;
                    yAtEvent[path, eventPtr] = y;
                    dAtEvent[path, eventPtr] = discount;
                    eventPtr++;
                }
            }
        });
    }

    // ── Longstaff-Schwartz continuation regression ────────────────────────────

    // Basis: 1, x, y, x², y², xy — the standard low-order polynomial basis for
    // two-factor Gaussian short-rate models (Longstaff & Schwartz, 2001, §3).
    private const int BasisSize = 6;

    private static double[] RegressContinuation(
        double[,] xAtEvent, double[,] yAtEvent, double[] target, int eventIndex, int paths)
    {
        var ata = new double[BasisSize, BasisSize];
        var atb = new double[BasisSize];
        var basis = new double[BasisSize];

        for (var i = 0; i < paths; i++)
        {
            var x = xAtEvent[i, eventIndex];
            var y = yAtEvent[i, eventIndex];
            basis[0] = 1.0; basis[1] = x; basis[2] = y; basis[3] = x * x; basis[4] = y * y; basis[5] = x * y;

            var t = target[i];
            for (var r = 0; r < BasisSize; r++)
            {
                atb[r] += basis[r] * t;
                for (var c = 0; c < BasisSize; c++)
                    ata[r, c] += basis[r] * basis[c];
            }
        }

        return SolveLinearSystem(ata, atb, BasisSize);
    }

    // Gaussian elimination with partial pivoting on a small dense n x n system.
    private static double[] SolveLinearSystem(double[,] a, double[] b, int n)
    {
        for (var col = 0; col < n; col++)
        {
            var pivot = col;
            for (var row = col + 1; row < n; row++)
                if (Math.Abs(a[row, col]) > Math.Abs(a[pivot, col]))
                    pivot = row;

            if (pivot != col)
            {
                for (var c = 0; c < n; c++)
                    (a[col, c], a[pivot, c]) = (a[pivot, c], a[col, c]);
                (b[col], b[pivot]) = (b[pivot], b[col]);
            }

            var diag = a[col, col];
            if (Math.Abs(diag) < 1e-14)
                continue; // Degenerate basis direction (e.g. near-zero factor variance); leave coefficient at 0.

            for (var row = col + 1; row < n; row++)
            {
                var factor = a[row, col] / diag;
                for (var c = col; c < n; c++)
                    a[row, c] -= factor * a[col, c];
                b[row] -= factor * b[col];
            }
        }

        var x = new double[n];
        for (var row = n - 1; row >= 0; row--)
        {
            var sum = b[row];
            for (var c = row + 1; c < n; c++)
                sum -= a[row, c] * x[c];
            x[row] = Math.Abs(a[row, row]) < 1e-14 ? 0.0 : sum / a[row, row];
        }
        return x;
    }

    // ── Event grid construction ───────────────────────────────────────────────

    private readonly record struct EventNode(double Years, double CashflowAmount, bool IsCallable, double CallPrice);

    private static (EventNode[] events, int[] gridIndex, double dt) BuildEventGrid(
        CallableBondDefinition bond, LocalDate valuationDate, int fineSteps)
    {
        var cashflows = bond.BuildCashflowSchedule(valuationDate);
        var byDate = new SortedDictionary<LocalDate, EventNode>();
        foreach (var cf in cashflows)
            byDate[cf.Date] = new EventNode(cf.YearsFromValuation, cf.Amount, false, 0.0);

        foreach (var call in bond.CallSchedule)
        {
            if (call.Date <= valuationDate)
                continue;

            var years  = Period.Between(valuationDate, call.Date, PeriodUnits.Days).Days / 365.0;
            var amount = byDate.TryGetValue(call.Date, out var existing) ? existing.CashflowAmount : 0.0;
            byDate[call.Date] = new EventNode(years, amount, true, call.CallPrice);
        }

        if (byDate.Count == 0)
            throw new ArgumentException("Bond has no cashflows after the valuation date.", nameof(bond));

        var events = byDate.Values.ToArray();
        var maturityYears = events[^1].Years;
        var dt = maturityYears / fineSteps;
        var gridIndex = events
            .Select(e => Math.Clamp((int)Math.Round(e.Years / dt), 1, fineSteps))
            .ToArray();

        return (events, gridIndex, dt);
    }

    private static PricingResult BuildPricingResult(double[] pathPV)
    {
        var n = pathPV.Length;
        var sum = 0.0;
        var sumSq = 0.0;
        foreach (var v in pathPV)
        {
            sum   += v;
            sumSq += v * v;
        }

        var mean      = sum / n;
        var sampleVar = (sumSq / n - mean * mean) / (n - 1);
        var stderr    = Math.Sqrt(Math.Max(0.0, sampleVar));
        const double z95 = 1.9599639845400536; // Φ⁻¹(0.975), full double precision.

        return new PricingResult(
            Price:                   mean,
            StandardError:           stderr,
            ConfidenceIntervalLower: mean - z95 * stderr,
            ConfidenceIntervalUpper: mean + z95 * stderr,
            Paths:                   n);
    }
}
