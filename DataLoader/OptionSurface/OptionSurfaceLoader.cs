namespace DataLoader.OptionSurface;

using DataLoader.Database;
using DataLoader.Models;
using DataLoader.Providers;
using Microsoft.Extensions.Logging;

/// <summary>
/// Downloads the full implied-vol surface for a bond futures options market and
/// persists every contract's daily settlement prices to market_data.time_series.
///
/// Usage:
///   var loader = new OptionSurfaceLoader(repo, loggerFactory);
///   var result = await loader.LoadAsync("RX", atm: 131.50, pct: 0.04,
///                    minExpiry: new DateOnly(2026, 9, 1),
///                    maxExpiry: new DateOnly(2027, 3, 1));
/// </summary>
public sealed class OptionSurfaceLoader
{
    // Courtesy delay between HTTP requests to avoid provider rate-limiting.
    private static readonly TimeSpan RequestDelay = TimeSpan.FromMilliseconds(150);

    private readonly MarketDataRepository _repo;
    private readonly ILoggerFactory       _loggerFactory;
    private readonly ILogger<OptionSurfaceLoader> _logger;

    public OptionSurfaceLoader(MarketDataRepository repo, ILoggerFactory loggerFactory)
    {
        _repo          = repo;
        _loggerFactory = loggerFactory;
        _logger        = loggerFactory.CreateLogger<OptionSurfaceLoader>();
    }

    /// <param name="futureCode">Future code, e.g. RX, DU, TY. See FutureSpecs.KnownCodes.</param>
    /// <param name="atm">
    ///   ATM / forward price. When null the loader tries to auto-resolve:
    ///   (1) latest price in the database, (2) Yahoo Finance (CME futures only).
    ///   Eurex futures require an explicit value.
    /// </param>
    /// <param name="pct">Half-width of the strike grid as a fraction, e.g. 0.04 = ±4 %.</param>
    /// <param name="minExpiry">First expiry month to include (day component ignored).</param>
    /// <param name="maxExpiry">Last expiry month to include (day component ignored).</param>
    /// <param name="historyFrom">
    ///   How far back to fetch settlement prices. Defaults to 1 year before minExpiry.
    /// </param>
    public async Task<SurfaceLoadResult> LoadAsync(
        string    futureCode,
        double?   atm,
        double    pct,
        DateOnly  minExpiry,
        DateOnly  maxExpiry,
        DateOnly? historyFrom      = null,
        string?   providerOverride = null,
        CancellationToken ct       = default)
    {
        var spec     = FutureSpecs.Get(futureCode);
        var provider = providerOverride ?? spec.Provider;

        var atmResolved = atm ?? await ResolveAtmAsync(spec, ct);

        if (providerOverride is not null)
            _logger.LogInformation("Provider overridden: {Spec} → {Override}", spec.Provider, provider);

        _logger.LogInformation(
            "{Code}: ATM={Atm}, ±{Pct:P0}, expiries {Min:yyyyMM}–{Max:yyyyMM}, step={Step}",
            futureCode, atmResolved, pct, minExpiry, maxExpiry, spec.StrikeStep);

        var strikes  = ComputeStrikes(atmResolved, pct, spec.StrikeStep);
        var expiries = ComputeExpiries(spec, minExpiry, maxExpiry);

        _logger.LogInformation(
            "Grid: {E} expiries × {S} strikes × 2 types = {T} contracts",
            expiries.Count, strikes.Count, expiries.Count * strikes.Count * 2);

        var contracts = (
            from expiry in expiries
            from strike in strikes
            from type   in (string[])["C", "P"]
            select new OptionContract(futureCode, expiry, type, strike, spec.StrikeStep, provider)
        ).ToList();

        var fetchFrom = historyFrom ?? minExpiry.AddYears(-1);

        return await FetchAndPersistAsync(spec, provider, contracts, fetchFrom, ct);
    }

    // ── Grid helpers ──────────────────────────────────────────────────────────

    // Generates all strikes in [atm*(1-pct), atm*(1+pct)] aligned to step.
    private static List<double> ComputeStrikes(double atm, double pct, double step)
    {
        var lower = Math.Floor(atm * (1.0 - pct) / step) * step;
        var upper = Math.Ceiling(atm * (1.0 + pct) / step) * step;

        var strikes = new List<double>();
        for (var k = lower; k <= upper + step * 0.5; k += step)
            strikes.Add(Math.Round(k, 8));   // suppress floating-point drift

        return strikes;
    }

    // Enumerates the first-of-month DateOnly for each listed expiry month in [from, to].
    private static List<DateOnly> ComputeExpiries(FutureSpec spec, DateOnly from, DateOnly to)
    {
        var expiries = new List<DateOnly>();
        var d = new DateOnly(from.Year, from.Month, 1);

        while (d <= new DateOnly(to.Year, to.Month, 1))
        {
            if (spec.ExpiryMonths.Contains(d.Month))
                expiries.Add(d);
            d = d.AddMonths(1);
        }

        return expiries;
    }

    // ── ATM resolution ────────────────────────────────────────────────────────

    private async Task<double> ResolveAtmAsync(FutureSpec spec, CancellationToken ct)
    {
        // 1 — try latest stored price for this future's underlying
        var dbAtm = await _repo.GetLatestPriceAsync(spec.Code, spec.Provider, ct);
        if (dbAtm.HasValue)
        {
            _logger.LogInformation("ATM resolved from DB: {Atm} ({Code}/{Provider})",
                dbAtm.Value, spec.Code, spec.Provider);
            return dbAtm.Value;
        }

        // 2 — try Yahoo Finance (CME futures only)
        if (spec.YahooTicker is not null)
        {
            _logger.LogInformation("Resolving ATM from Yahoo Finance ({Ticker})…", spec.YahooTicker);
            await using var yahoo = ProviderFactory.Create("yahoo", _loggerFactory);
            var now    = DateTimeOffset.UtcNow;
            var points = await yahoo.FetchAsync(
                new FetchRequest(spec.YahooTicker, "yahoo", "1d", now.AddDays(-7), now), ct);

            if (points.Count > 0)
            {
                _logger.LogInformation("ATM resolved from Yahoo: {Atm}", points[^1].Value);
                return points[^1].Value;
            }
        }

        throw new InvalidOperationException(
            $"Cannot auto-resolve ATM for '{spec.Code}' " +
            (spec.YahooTicker is null
                ? $"({spec.Provider} futures are not on Yahoo Finance). "
                : "(Yahoo Finance returned no data). ") +
            "Pass --atm <price> explicitly.");
    }

    // ── Fetch and persist ─────────────────────────────────────────────────────

    private async Task<SurfaceLoadResult> FetchAndPersistAsync(
        FutureSpec           spec,
        string               providerName,
        List<OptionContract> contracts,
        DateOnly             historyFrom,
        CancellationToken    ct)
    {
        var succeeded = 0;
        var rowsTotal = 0;
        var failed    = new List<string>();

        var fetchFrom = new DateTimeOffset(historyFrom.ToDateTime(TimeOnly.MinValue), TimeSpan.Zero);
        var fetchTo   = DateTimeOffset.UtcNow;

        // One shared provider instance for the entire surface load (avoids per-contract
        // session establishment; for Yahoo this means one crumb covers all requests).
        await using var provider = ProviderFactory.Create(providerName, _loggerFactory);

        foreach (var contract in contracts)
        {
            if (ct.IsCancellationRequested) break;

            try
            {
                // Translate to the provider's native ticker format (e.g. Yahoo option tickers
                // differ from our internal format); the internal ticker is still used for DB storage.
                var fetchTicker = contract.TickerForProvider(providerName, spec.YahooOptionsSymbol);
                if (fetchTicker != contract.Ticker)
                    _logger.LogDebug("{Internal} → {External}", contract.Ticker, fetchTicker);

                var request = new FetchRequest(fetchTicker, providerName, "1d", fetchFrom, fetchTo);
                var points  = await provider.FetchAsync(request, ct);

                if (points.Count == 0)
                {
                    _logger.LogDebug("{Ticker}: no data", contract.Ticker);
                }
                else
                {
                    var secId  = await _repo.GetOrCreateSecurityIdAsync(
                        contract.Ticker, providerName, ct);   // internal ticker in DB
                    var rows   = await _repo.UpsertTimeSeriesAsync(secId, points, ct);
                    rowsTotal += rows;
                    succeeded++;
                    _logger.LogInformation("✓ {Ticker}: {Rows} rows", contract.Ticker, rows);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError("✗ {Ticker}: {Msg}", contract.Ticker, ex.Message);
                failed.Add(contract.Ticker);
            }

            await Task.Delay(RequestDelay, ct);
        }

        var result = new SurfaceLoadResult(contracts.Count, succeeded, rowsTotal, failed);
        _logger.LogInformation(
            "Surface complete — {Ok}/{Total} contracts, {Rows} rows stored, {Fail} failed",
            succeeded, contracts.Count, rowsTotal, failed.Count);
        return result;
    }
}

public sealed record SurfaceLoadResult(
    int                    Attempted,
    int                    Succeeded,
    int                    RowsStored,
    IReadOnlyList<string>  Failed);
