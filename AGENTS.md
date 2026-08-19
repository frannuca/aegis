# Aegis Quant Pricing Engine

## Purpose

This project implements a cross-asset quant pricing library for:

- FX options
- Interest-rate options
- Equity options
- SABR-based volatility modelling
- Monte Carlo simulation
- Analytical pricing formulas
- Greeks and calibration utilities

## Core engineering rules

- Prefer simple, testable numerical code.
- Never change pricing formulas without adding or updating tests.
- Every model must have:
  - mathematical description
  - implementation
  - unit tests
  - numerical sanity checks
  - benchmark example

## Languages

Primary languages:

- C++ for high-performance pricing kernels
- C# for orchestration, APIs, portfolio structures, and integration
- Python for prototypes, notebooks, calibration experiments, and validation

## Quant rules

When implementing pricing code:

- State the stochastic process.
- State the numeraire/measure when relevant.
- State model assumptions.
- Include edge cases:
  - zero volatility
  - zero maturity
  - deep ITM/OTM
  - negative rates where relevant
  - shifted lognormal cases

## Numerical rules

- Avoid hidden magic constants.
- Use deterministic seeds in tests.
- Prefer stable algorithms over clever shortcuts.
- Validate Monte Carlo prices against analytical formulas when available.
- Report confidence intervals for Monte Carlo tests.

## SABR conventions

Use Hagan-style SABR unless explicitly stated otherwise.

Support:

- lognormal SABR
- normal SABR
- shifted lognormal SABR
- beta = 0, beta = 1 special cases
- calibration to smile quotes

## Code style

- Keep pricing formulas isolated from product definitions.
- Avoid mixing market data loading with pricing logic.
- Prefer immutable model parameter objects.
- Validate inputs aggressively.
- No silent fallback when calibration fails.

## Testing

Before completing a change, run:

```bash
dotnet test
ctest
pytest
