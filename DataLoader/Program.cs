using System.CommandLine;
using System.CommandLine.Invocation;
using System.CommandLine.Parsing;
using System.Globalization;
using System.Text.Json;
using DataLoader.Database;
using DataLoader.OptionSurface;
using DataLoader.Providers;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

// ── Configuration ─────────────────────────────────────────────────────────────
// DB credentials are read from appsettings.json or env vars with the prefix
// DATALOADER_  (e.g. DATALOADER_Database__Username, DATALOADER_Database__Password).
// Fill in appsettings.json before first run.

var config = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json", optional: true)
    .AddEnvironmentVariables("DATALOADER_")
    .Build();

var loggerFactory = LoggerFactory.Create(b =>
    b.AddConsole().SetMinimumLevel(LogLevel.Information));

var log = loggerFactory.CreateLogger("dataloader");

DatabaseSettings LoadDbSettings()
{
    var s = config.GetSection("Database").Get<DatabaseSettings>() ?? new DatabaseSettings();
    if (string.IsNullOrWhiteSpace(s.Username))
        log.LogWarning(
            "Database username is empty — set it in appsettings.json or via " +
            "DATALOADER_Database__Username environment variable.");
    return s;
}

// ── Shared option factories ────────────────────────────────────────────────────

static DateTimeOffset ParseUtcDate(ArgumentResult r)
{
    var s = r.Tokens[0].Value;
    if (DateTimeOffset.TryParse(s, CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var dto))
        return dto;
    r.ErrorMessage = $"Cannot parse '{s}' as a date. Use ISO 8601, e.g. 2024-01-01 or 2024-01-01T00:00:00Z.";
    return default;
}

Option<DateTimeOffset> MakeFromOption() =>
    new("--from",
        parseArgument: ParseUtcDate,
        isDefault: true,
        description: "Start date UTC (default: 1 year ago). E.g. 2024-01-01");

Option<DateTimeOffset> MakeToOption() =>
    new("--to",
        parseArgument: ParseUtcDate,
        isDefault: true,
        description: "End date UTC (default: today). E.g. 2024-12-31");

// ── fetch command ──────────────────────────────────────────────────────────────

var tickerOpt = new Option<string>(
    "--ticker",
    "FX ticker in Yahoo Finance format, e.g. EURUSD=X or GBPUSD=X")
{ IsRequired = true };

var intervalOpt = new Option<string>(
    "--interval",
    "Bar interval: 1d | 1h | 30m | 15m | 5m | 1m  (intraday lookback limits apply)");
intervalOpt.SetDefaultValue("1d");

var providerOpt = new Option<string>("--provider", "Data provider (default: yahoo)");
providerOpt.SetDefaultValue("yahoo");

var fromOpt = MakeFromOption();
var toOpt   = MakeToOption();

fromOpt.SetDefaultValue(DateTimeOffset.UtcNow.AddYears(-1));
toOpt.SetDefaultValue(DateTimeOffset.UtcNow);

var fetchCmd = new Command("fetch", "Download one FX ticker and upsert into the database");
fetchCmd.AddOption(tickerOpt);
fetchCmd.AddOption(intervalOpt);
fetchCmd.AddOption(fromOpt);
fetchCmd.AddOption(toOpt);
fetchCmd.AddOption(providerOpt);

fetchCmd.SetHandler(async (InvocationContext ctx) =>
{
    var ticker   = ctx.ParseResult.GetValueForOption(tickerOpt)!;
    var interval = ctx.ParseResult.GetValueForOption(intervalOpt)!;
    var from     = ctx.ParseResult.GetValueForOption(fromOpt);
    var to       = ctx.ParseResult.GetValueForOption(toOpt);
    var provider = ctx.ParseResult.GetValueForOption(providerOpt)!;
    var ct       = ctx.GetCancellationToken();

    ctx.ExitCode = await RunFetch(ticker, interval, from, to, provider, ct) ? 0 : 1;
});

// ── fetch-batch command ────────────────────────────────────────────────────────

var batchFileOpt = new Option<FileInfo>(
    "--config",
    "Path to a JSON batch config file")
{ IsRequired = true };

var batchFromOpt = MakeFromOption();
var batchToOpt   = MakeToOption();
batchFromOpt.SetDefaultValue(DateTimeOffset.UtcNow.AddYears(-1));
batchToOpt.SetDefaultValue(DateTimeOffset.UtcNow);

var fetchBatchCmd = new Command("fetch-batch", "Download multiple FX tickers from a JSON config file");
fetchBatchCmd.AddOption(batchFileOpt);
fetchBatchCmd.AddOption(batchFromOpt);
fetchBatchCmd.AddOption(batchToOpt);

fetchBatchCmd.SetHandler(async (InvocationContext ctx) =>
{
    var configFile  = ctx.ParseResult.GetValueForOption(batchFileOpt)!;
    var defaultFrom = ctx.ParseResult.GetValueForOption(batchFromOpt);
    var defaultTo   = ctx.ParseResult.GetValueForOption(batchToOpt);
    var ct          = ctx.GetCancellationToken();

    if (!configFile.Exists)
    {
        log.LogError("Config file not found: {Path}", configFile.FullName);
        ctx.ExitCode = 1;
        return;
    }

    BatchConfig? batch;
    try
    {
        await using var stream = configFile.OpenRead();
        batch = await JsonSerializer.DeserializeAsync<BatchConfig>(
            stream, new JsonSerializerOptions(JsonSerializerDefaults.Web), ct);
    }
    catch (Exception ex)
    {
        log.LogError(ex, "Failed to parse batch config: {Path}", configFile.FullName);
        ctx.ExitCode = 1;
        return;
    }

    if (batch?.Jobs is not { Count: > 0 })
    {
        log.LogWarning("No jobs defined in {Path}", configFile.FullName);
        return;
    }

    var failed = false;
    foreach (var job in batch.Jobs)
    {
        if (string.IsNullOrWhiteSpace(job.Ticker)) continue;  // skip _note sentinel entries

        var from     = job.From       ?? batch.From ?? defaultFrom;
        var to       = job.To         ?? batch.To   ?? defaultTo;
        var interval = job.Interval   ?? batch.Interval ?? "1d";
        var provider = job.Provider   ?? batch.Provider ?? "yahoo";

        if (!await RunFetch(job.Ticker, interval, from, to, provider, ct))
            failed = true;
    }

    ctx.ExitCode = failed ? 1 : 0;
});

// ── fetch-surface command ──────────────────────────────────────────────────────

static DateOnly ParseYearMonth(ArgumentResult r)
{
    var s = r.Tokens[0].Value.Trim();
    // Accept YYYY-MM or YYYY-MM-DD
    if (s.Length == 7) s += "-01";
    if (DateOnly.TryParseExact(s, "yyyy-MM-dd", out var d)) return d;
    r.ErrorMessage = $"Cannot parse '{r.Tokens[0].Value}' as a month. Use YYYY-MM or YYYY-MM-DD.";
    return default;
}

var sfFutureOpt = new Option<string>(
    "--future",
    $"Future code: {string.Join(", ", FutureSpecs.KnownCodes)}")
{ IsRequired = true };

var sfPctOpt = new Option<double>(
    "--pct",
    "Half-width of strike grid as a fraction, e.g. 0.04 = ±4 % around ATM")
{ IsRequired = true };

var sfMinExpiryOpt = new Option<DateOnly>(
    "--min-expiry",
    parseArgument: ParseYearMonth,
    isDefault: false,
    description: "Earliest option expiry month, e.g. 2026-09")
{ IsRequired = true };

var sfMaxExpiryOpt = new Option<DateOnly>(
    "--max-expiry",
    parseArgument: ParseYearMonth,
    isDefault: false,
    description: "Latest option expiry month, e.g. 2027-06")
{ IsRequired = true };

var sfAtmOpt = new Option<double?>(
    "--atm",
    "ATM / forward price. Optional: auto-resolved from Yahoo Finance for CME futures, " +
    "or from the last stored price in the DB. Required for Eurex futures (RX, DU, OE).");

var sfProviderOpt = new Option<string?>(
    "--provider",
    "Override the data provider (e.g. --provider yahoo). Defaults to the exchange defined in FutureSpec.");

var sfFromOpt = MakeFromOption();
sfFromOpt.SetDefaultValue(DateTimeOffset.UtcNow.AddYears(-1));

var fetchSurfaceCmd = new Command(
    "fetch-surface",
    "Download the full option surface for a bond future and upsert settlement prices into the database");
fetchSurfaceCmd.AddOption(sfFutureOpt);
fetchSurfaceCmd.AddOption(sfPctOpt);
fetchSurfaceCmd.AddOption(sfMinExpiryOpt);
fetchSurfaceCmd.AddOption(sfMaxExpiryOpt);
fetchSurfaceCmd.AddOption(sfAtmOpt);
fetchSurfaceCmd.AddOption(sfProviderOpt);
fetchSurfaceCmd.AddOption(sfFromOpt);

fetchSurfaceCmd.SetHandler(async (InvocationContext ctx) =>
{
    var futureCode       = ctx.ParseResult.GetValueForOption(sfFutureOpt)!;
    var pct              = ctx.ParseResult.GetValueForOption(sfPctOpt);
    var minExpiry        = ctx.ParseResult.GetValueForOption(sfMinExpiryOpt);
    var maxExpiry        = ctx.ParseResult.GetValueForOption(sfMaxExpiryOpt);
    var atm              = ctx.ParseResult.GetValueForOption(sfAtmOpt);
    var providerOverride = ctx.ParseResult.GetValueForOption(sfProviderOpt);
    var histFrom         = ctx.ParseResult.GetValueForOption(sfFromOpt);
    var ct               = ctx.GetCancellationToken();

    var db     = LoadDbSettings();
    var repo   = new MarketDataRepository(db, loggerFactory.CreateLogger<MarketDataRepository>());
    var loader = new OptionSurfaceLoader(repo, loggerFactory);

    try
    {
        var result = await loader.LoadAsync(
            futureCode, atm, pct, minExpiry, maxExpiry,
            historyFrom: DateOnly.FromDateTime(histFrom.UtcDateTime),
            providerOverride: providerOverride,
            ct);

        ctx.ExitCode = result.Failed.Count == result.Attempted ? 1 : 0;
    }
    catch (Exception ex)
    {
        log.LogError(ex, "fetch-surface failed: {Msg}", ex.Message);
        ctx.ExitCode = 1;
    }
});

// ── fetch-surface-snapshot command ────────────────────────────────────────────
// Downloads today's full option chain from Yahoo Finance (v7/finance/options).
// Discovers all listed strikes automatically — no strike grid required.
// Stores mid price (or last price when bid/ask unavailable) for each contract.
//
// Yahoo Finance does NOT carry CME bond futures options (ZN, ZF, ZT, ZB).
// Use treasury ETF proxies for interest-rate vol surfaces:
//   --symbol TLT   (iShares 20+ Year Treasury, proxy for US / Euro-Bund)
//   --symbol IEF   (iShares 7-10 Year Treasury, proxy for TY / Euro-Bobl)
//   --symbol SHY   (iShares 1-3 Year Treasury,  proxy for TU / Euro-Schatz)
//
// Usage:
//   fetch-surface-snapshot --symbol TLT --min-expiry 2026-09 --max-expiry 2027-06
//   fetch-surface-snapshot --future TY --min-expiry 2026-09 --max-expiry 2027-06
//   fetch-surface-snapshot --future TY --symbol IEF --min-expiry 2026-09 --max-expiry 2027-06
//     (queries IEF on Yahoo but stores tickers as TY-...)

var snapSymbolOpt = new Option<string?>(
    "--symbol",
    "Yahoo Finance symbol to query (e.g. TLT, IEF, SHY, SPY). " +
    "Required unless --future maps to a Yahoo options symbol. " +
    "When both --future and --symbol are provided, --symbol overrides the Yahoo query " +
    "while --future sets the internal ticker prefix.");

var snapFutureOpt = new Option<string?>(
    "--future",
    $"Future code ({string.Join(", ", FutureSpecs.KnownCodes)}). " +
    "Sets the internal ticker prefix and, if the future has a Yahoo options symbol, " +
    "the Yahoo query symbol (overridable by --symbol).");

var snapMinExpiryOpt = new Option<DateOnly>(
    "--min-expiry",
    parseArgument: ParseYearMonth,
    isDefault: false,
    description: "Earliest expiry month to include, e.g. 2026-09")
{ IsRequired = true };

var snapMaxExpiryOpt = new Option<DateOnly>(
    "--max-expiry",
    parseArgument: ParseYearMonth,
    isDefault: false,
    description: "Latest expiry month to include, e.g. 2027-06")
{ IsRequired = true };

var fetchSnapshotCmd = new Command(
    "fetch-surface-snapshot",
    "Download today's full option chain from Yahoo Finance and upsert mid prices into the database. " +
    "Use --symbol TLT / IEF / SHY for treasury ETF proxies (Yahoo does not list CME bond futures options).");
fetchSnapshotCmd.AddOption(snapSymbolOpt);
fetchSnapshotCmd.AddOption(snapFutureOpt);
fetchSnapshotCmd.AddOption(snapMinExpiryOpt);
fetchSnapshotCmd.AddOption(snapMaxExpiryOpt);

fetchSnapshotCmd.SetHandler(async (InvocationContext ctx) =>
{
    var symbolArg  = ctx.ParseResult.GetValueForOption(snapSymbolOpt);
    var futureCode = ctx.ParseResult.GetValueForOption(snapFutureOpt);
    var minExpiry  = ctx.ParseResult.GetValueForOption(snapMinExpiryOpt);
    var maxExpiry  = ctx.ParseResult.GetValueForOption(snapMaxExpiryOpt);
    var ct         = ctx.GetCancellationToken();

    // Resolve (yahooSymbol, tickerPrefix) from the arguments.
    // --symbol overrides the Yahoo query; --future sets the ticker prefix.
    string? yahooSymbol, tickerPrefix;

    if (futureCode is not null)
    {
        FutureSpec spec;
        try { spec = FutureSpecs.Get(futureCode); }
        catch (Exception ex)
        {
            log.LogError("Unknown future code '{Code}': {Msg}", futureCode, ex.Message);
            ctx.ExitCode = 1;
            return;
        }

        tickerPrefix = spec.Code;
        yahooSymbol  = symbolArg ?? spec.YahooOptionsSymbol;

        if (yahooSymbol is null)
        {
            log.LogError(
                "'{Code}' has no Yahoo options symbol and --symbol was not provided. " +
                "Yahoo Finance does not carry options for {Provider} futures. " +
                "Try: --future {Code} --symbol IEF  (or TLT / SHY).",
                futureCode, spec.Provider, futureCode);
            ctx.ExitCode = 1;
            return;
        }
    }
    else if (symbolArg is not null)
    {
        yahooSymbol  = symbolArg;
        tickerPrefix = symbolArg;
    }
    else
    {
        log.LogError("Provide --symbol <YAHOO_SYMBOL> or --future <CODE> (or both).");
        ctx.ExitCode = 1;
        return;
    }

    var db     = LoadDbSettings();
    var repo   = new MarketDataRepository(db, loggerFactory.CreateLogger<MarketDataRepository>());
    var loader = new YahooOptionChainLoader(repo, loggerFactory);

    try
    {
        var result = await loader.LoadTodayAsync(yahooSymbol, tickerPrefix, minExpiry, maxExpiry, ct);
        ctx.ExitCode = result.Failed > 0 && result.Stored == 0 ? 1 : 0;
    }
    catch (Exception ex)
    {
        log.LogError(ex, "fetch-surface-snapshot failed: {Msg}", ex.Message);
        ctx.ExitCode = 1;
    }
});

// ── Root command ───────────────────────────────────────────────────────────────

var root = new RootCommand("Aegis market data loader — fetches FX rates and stores them in PostgreSQL");
root.AddCommand(fetchCmd);
root.AddCommand(fetchBatchCmd);
root.AddCommand(fetchSurfaceCmd);
root.AddCommand(fetchSnapshotCmd);

return await root.InvokeAsync(args);

// ── Implementation ─────────────────────────────────────────────────────────────

async Task<bool> RunFetch(
    string ticker, string interval,
    DateTimeOffset from, DateTimeOffset to,
    string providerName, CancellationToken ct)
{
    try
    {
        await using var provider = ProviderFactory.Create(providerName, loggerFactory);

        var db      = LoadDbSettings();
        var repo    = new MarketDataRepository(db, loggerFactory.CreateLogger<MarketDataRepository>());
        var request = new FetchRequest(ticker, provider.Name, interval, from, to);

        var points = await provider.FetchAsync(request, ct);
        if (points.Count == 0)
        {
            log.LogWarning("No data returned for {Ticker} — nothing stored", ticker);
            return true;
        }

        var securityId = await repo.GetOrCreateSecurityIdAsync(ticker, provider.Name, ct);
        var stored     = await repo.UpsertTimeSeriesAsync(securityId, points, ct);
        log.LogInformation("✓ {Ticker}: {Stored} rows stored", ticker, stored);
        return true;
    }
    catch (Exception ex)
    {
        log.LogError(ex, "✗ Failed to fetch/store {Ticker}", ticker);
        return false;
    }
}

// ── Batch config model ─────────────────────────────────────────────────────────

sealed class BatchConfig
{
    public DateTimeOffset? From     { get; init; }
    public DateTimeOffset? To       { get; init; }
    public string?         Provider { get; init; }  // default provider for all jobs
    public string?         Interval { get; init; }  // default interval for all jobs
    public List<BatchJob>? Jobs     { get; init; }
}

sealed class BatchJob
{
    public string          Ticker   { get; init; } = string.Empty;
    public string?         Interval { get; init; }
    public string?         Provider { get; init; }
    public DateTimeOffset? From     { get; init; }  // overrides top-level From for this job
    public DateTimeOffset? To       { get; init; }  // overrides top-level To for this job
}
