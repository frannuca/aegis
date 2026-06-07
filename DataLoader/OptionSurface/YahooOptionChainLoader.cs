namespace DataLoader.OptionSurface;

using System.Globalization;
using DataLoader.Database;
using DataLoader.Models;
using DataLoader.Providers.Yahoo;
using Microsoft.Extensions.Logging;

/// <summary>
/// Downloads today's option chain snapshot from Yahoo Finance (v7/finance/options)
/// and persists each contract's mid price into market_data.time_series.
///
/// Works with any Yahoo-listed underlying: equities, ETFs, or index options.
/// For interest-rate surfaces, use treasury ETF proxies:
///   TLT  ≈ US/RX (20+ year)   |   IEF  ≈ TY/OE (7-10 year)   |   SHY  ≈ TU/DU (1-3 year)
///
/// Note: Yahoo Finance does not carry CME bond futures options (ZN, ZF, ZT, ZB).
/// </summary>
public sealed class YahooOptionChainLoader
{
    private static readonly TimeSpan RequestDelay = TimeSpan.FromMilliseconds(200);

    private readonly MarketDataRepository _repo;
    private readonly ILogger<YahooOptionChainLoader> _logger;

    public YahooOptionChainLoader(MarketDataRepository repo, ILoggerFactory loggerFactory)
    {
        _repo   = repo;
        _logger = loggerFactory.CreateLogger<YahooOptionChainLoader>();
    }

    /// <param name="yahooSymbol">
    ///   Symbol to query from Yahoo Finance (e.g. "TLT", "IEF", "SHY", "SPY").
    /// </param>
    /// <param name="tickerPrefix">
    ///   Prefix for the internal DB ticker, e.g. "TLT" or "TY".
    ///   Normally equals <paramref name="yahooSymbol"/>; set differently only when
    ///   aliasing an ETF proxy under a future code.
    /// </param>
    /// <param name="minExpiry">Earliest expiry month to include (day component ignored).</param>
    /// <param name="maxExpiry">Latest expiry month to include (day component ignored).</param>
    public async Task<SnapshotLoadResult> LoadTodayAsync(
        string            yahooSymbol,
        string            tickerPrefix,
        DateOnly          minExpiry,
        DateOnly          maxExpiry,
        CancellationToken ct = default)
    {
        var today = DateTimeOffset.UtcNow;

        await using var yahoo = new YahooFinanceProvider(
            Microsoft.Extensions.Logging.Abstractions.NullLogger<YahooFinanceProvider>.Instance);

        // 1. Discover all expiry timestamps Yahoo currently lists
        _logger.LogInformation("Querying available expiries for {Symbol}...", yahooSymbol);
        var allExpiries = await yahoo.GetOptionExpiriesAsync(yahooSymbol, ct);

        // 2. Filter to the requested [minExpiry, maxExpiry] window (month granularity)
        var minTs = new DateTimeOffset(minExpiry.Year, minExpiry.Month, 1, 0, 0, 0, TimeSpan.Zero)
                        .ToUnixTimeSeconds();
        var maxTs = new DateTimeOffset(maxExpiry.Year, maxExpiry.Month,
                        DateTime.DaysInMonth(maxExpiry.Year, maxExpiry.Month),
                        23, 59, 59, TimeSpan.Zero).ToUnixTimeSeconds();

        var filteredExpiries = allExpiries.Where(ts => ts >= minTs && ts <= maxTs).ToList();

        _logger.LogInformation(
            "{Symbol}: {Total} expiries available, {Kept} within {Min:yyyyMM}-{Max:yyyyMM}",
            yahooSymbol, allExpiries.Count, filteredExpiries.Count, minExpiry, maxExpiry);

        if (filteredExpiries.Count == 0)
        {
            if (allExpiries.Count == 0)
                _logger.LogWarning(
                    "Yahoo Finance returned no option expiries for '{Symbol}'. " +
                    "CME bond futures (ZN, ZF, ZT, ZB) are not listed -- " +
                    "use a treasury ETF proxy instead: TLT (approx 20Y), IEF (approx 10Y), SHY (approx 2Y).",
                    yahooSymbol);
            else
                _logger.LogWarning(
                    "No expiries found within {Min:yyyyMM}-{Max:yyyyMM} " +
                    "(Yahoo listed {Total} expiries outside that window).",
                    minExpiry, maxExpiry, allExpiries.Count);
            return new SnapshotLoadResult(0, 0, 0, []);
        }

        // 3. Fetch chain per expiry and persist
        var succeeded = 0;
        var skipped   = 0;
        var failed    = new List<string>();

        foreach (var expiryTs in filteredExpiries)
        {
            if (ct.IsCancellationRequested) break;

            var expiryDate = DateOnly.FromDateTime(
                DateTimeOffset.FromUnixTimeSeconds(expiryTs).UtcDateTime);

            _logger.LogInformation("Fetching chain for {Symbol} expiry {Expiry}", yahooSymbol, expiryDate);

            try
            {
                var chain = await yahoo.FetchOptionChainAsync(yahooSymbol, expiryTs, ct);

                if (chain is null)
                {
                    _logger.LogWarning("No chain data for {Symbol} {Expiry}", yahooSymbol, expiryDate);
                    continue;
                }

                var (ok, skip, fail) = await PersistChainAsync(
                    tickerPrefix, expiryDate, chain.Calls, "C", today, ct);
                succeeded += ok; skipped += skip; failed.AddRange(fail);

                (ok, skip, fail) = await PersistChainAsync(
                    tickerPrefix, expiryDate, chain.Puts, "P", today, ct);
                succeeded += ok; skipped += skip; failed.AddRange(fail);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to fetch chain for {Symbol} {Expiry}",
                    yahooSymbol, expiryDate);
                failed.Add($"{tickerPrefix}-{expiryDate:yyyyMM}");
            }

            await Task.Delay(RequestDelay, ct);
        }

        var result = new SnapshotLoadResult(succeeded, skipped, failed.Count, failed);
        _logger.LogInformation(
            "Snapshot complete -- {Ok} stored, {Skip} skipped (no valid price), {Fail} failed",
            succeeded, skipped, failed.Count);
        return result;
    }

    // ---- Helpers ---------------------------------------------------------------

    private async Task<(int Stored, int Skipped, List<string> Failed)> PersistChainAsync(
        string                        tickerPrefix,
        DateOnly                      expiryDate,
        IEnumerable<YahooOptionQuote>? quotes,
        string                        type,
        DateTimeOffset                timestamp,
        CancellationToken             ct)
    {
        if (quotes is null) return (0, 0, []);

        int stored = 0, skipped = 0;
        var failed = new List<string>();

        foreach (var q in quotes)
        {
            var mid = Mid(q);
            if (mid <= 0)
            {
                skipped++;
                continue;
            }

            var ticker = BuildInternalTicker(tickerPrefix, expiryDate, type, q.Strike);

            try
            {
                var secId = await _repo.GetOrCreateSecurityIdAsync(ticker, "yahoo", ct);
                await _repo.UpsertTimeSeriesAsync(secId, [new TimeSeriesPoint(timestamp, mid)], ct);
                stored++;
                _logger.LogDebug("+ {Ticker} = {Mid:F6}", ticker, mid);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "x {Ticker}: {Msg}", ticker, ex.Message);
                failed.Add(ticker);
            }
        }

        return (stored, skipped, failed);
    }

    // Mid = (bid+ask)/2 when both are positive, otherwise last price.
    private static double Mid(YahooOptionQuote q) =>
        q.Bid > 0 && q.Ask > 0 ? (q.Bid + q.Ask) / 2.0 : q.LastPrice;

    // Internal ticker: {PREFIX}-{YYYYMM}-{C|P}-{STRIKE}
    // e.g. TLT-202709-C-092.00, IEF-202706-P-098.50
    private static string BuildInternalTicker(
        string prefix, DateOnly expiry, string type, double strike) =>
        $"{prefix}-{expiry:yyyyMM}-{type}-{FormatStrike(strike)}";

    // Auto-detect the minimum decimal places needed to represent the strike exactly (min 2).
    private static string FormatStrike(double strike)
    {
        for (var d = 0; d <= 6; d++)
        {
            var scaled = strike * Math.Pow(10, d);
            if (Math.Abs(scaled - Math.Round(scaled)) < 1e-9)
                return strike.ToString($"F{Math.Max(2, d)}", CultureInfo.InvariantCulture);
        }
        return strike.ToString("F4", CultureInfo.InvariantCulture);
    }
}

public sealed record SnapshotLoadResult(
    int                   Stored,
    int                   Skipped,
    int                   Failed,
    IReadOnlyList<string> FailedTickers);
