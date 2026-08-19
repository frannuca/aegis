---
name: quant-pricing
description: Use when implementing or reviewing derivatives pricing models, Monte Carlo engines, SABR volatility models, Greeks, calibration, or analytical option formulas.
---

# Quant Pricing Skill

When working on pricing code:

1. Identify the product payoff.
2. Identify the underlying stochastic process.
3. Identify the pricing measure.
4. Derive or cite the pricing formula.
5. Implement the smallest testable component first.
6. Add deterministic tests.
7. Compare against analytical formulas where possible.
8. Check numerical edge cases.

## Required tests

For every pricing model, include:

- ATM option test
- deep ITM test
- deep OTM test
- zero maturity test
- near-zero volatility test
- monotonicity checks
- finite-difference Greek sanity checks

## Monte Carlo requirements

Monte Carlo pricers must expose:

- number of paths
- random seed
- standard error
- confidence interval
- antithetic option if available
- control variate option when useful

## Review checklist

Flag:

- unstable divisions
- missing discounting
- wrong forward/spot usage
- wrong measure
- inconsistent day count
- incorrect volatility units
- missing annuity for IR options
- calibration overfitting
