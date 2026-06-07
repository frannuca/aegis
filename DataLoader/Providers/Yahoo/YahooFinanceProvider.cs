namespace DataLoader.Providers.Yahoo;

using System.Net;
using System.Text.Json;
using DataLoader.Models;
using Microsoft.Extensions.Logging;

// Yahoo Finance V8 chart API — requires a session crumb for authenticated requests.
// FX tickers use the =X suffix convention (e.g. EURUSD=X, GBPUSD=X).
//
// Intraday lookback limits imposed by Yahoo Finance (per request):
//   1h        → 729 days   |   30m/15m/5m/2m → 59 days   |   1m → 6 days
// Ranges exceeding these limits are automatically split into chunks.
public sealed class YahooFinanceProvider : IMarketDataProvider
{
    public string Name => "yahoo";

    private static readonly JsonSerializerOptions JsonOpts = new(JsonSerializerDefaults.Web);

    private readonly HttpClient _http;
    private readonly ILogger<YahooFinanceProvider> _logger;
    private readonly SemaphoreSlim _sessionLock = new(1, 1);
    private string? _crumb;

    public YahooFinanceProvider(ILogger<YahooFinanceProvider> logger)
    {
        _logger = logger;
        var handler = new HttpClientHandler
        {
            CookieContainer = new CookieContainer(),
            UseCookies = true,
            AllowAutoRedirect = true,
        };
        _http = new HttpClient(handler) { Timeout = TimeSpan.FromSeconds(30) };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36");
        _http.DefaultRequestHeaders.Accept.ParseAdd("application/json,*/*");
    }

    public async Task<IReadOnlyList<TimeSeriesPoint>> FetchAsync(FetchRequest request, CancellationToken ct = default)
    {
        await EnsureSessionAsync(ct);

        var chunkDays = MaxChunkDays(request.Interval);
        var totalDays = (request.To - request.From).TotalDays;

        if (totalDays <= chunkDays)
            return await FetchChunkWithRetryAsync(request, ct);

        // Split into chunks that respect the Yahoo Finance per-request limit
        _logger.LogInformation(
            "{Ticker} [{Interval}]: {Days:F0}-day range exceeds {Max}-day limit — splitting into chunks",
            request.Ticker, request.Interval, totalDays, chunkDays);

        var all = new List<TimeSeriesPoint>();
        var chunkFrom = request.From;

        while (chunkFrom < request.To)
        {
            var chunkTo = chunkFrom.AddDays(chunkDays);
            if (chunkTo > request.To) chunkTo = request.To;

            var chunk = await FetchChunkWithRetryAsync(request with { From = chunkFrom, To = chunkTo }, ct);
            all.AddRange(chunk);
            chunkFrom = chunkTo;
        }

        _logger.LogInformation("{Ticker}: {Total} points total across all chunks", request.Ticker, all.Count);
        return all;
    }

    private async Task<IReadOnlyList<TimeSeriesPoint>> FetchChunkWithRetryAsync(FetchRequest request, CancellationToken ct)
    {
        var points = await FetchInternalAsync(request, ct);

        // Crumb may have expired; refresh once and retry
        if (points is null)
        {
            _logger.LogWarning("Session appears expired — refreshing crumb and retrying for {Ticker}", request.Ticker);
            _crumb = null;
            await EnsureSessionAsync(ct);
            points = await FetchInternalAsync(request, ct)
                     ?? throw new InvalidOperationException($"No data for {request.Ticker} after session refresh.");
        }

        return points;
    }

    private async Task<IReadOnlyList<TimeSeriesPoint>?> FetchInternalAsync(FetchRequest request, CancellationToken ct)
    {
        var period1 = request.From.ToUnixTimeSeconds();
        var period2 = request.To.ToUnixTimeSeconds();
        var url = $"https://query1.finance.yahoo.com/v8/finance/chart/{Uri.EscapeDataString(request.Ticker)}" +
                  $"?interval={request.Interval}&period1={period1}&period2={period2}" +
                  $"&crumb={Uri.EscapeDataString(_crumb!)}";

        _logger.LogDebug("GET {Ticker} [{Interval}] {From:yyyy-MM-dd} → {To:yyyy-MM-dd}",
            request.Ticker, request.Interval, request.From, request.To);

        var resp = await _http.GetAsync(url, ct);

        if (resp.StatusCode is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden)
            return null; // signals caller to refresh session

        if (!resp.IsSuccessStatusCode)
        {
            var body = await resp.Content.ReadAsStringAsync(ct);
            throw new HttpRequestException(
                $"Yahoo Finance HTTP {(int)resp.StatusCode} for {request.Ticker} " +
                $"[{request.Interval}] {request.From:yyyy-MM-dd}→{request.To:yyyy-MM-dd}: {body}");
        }

        var json = await resp.Content.ReadAsStringAsync(ct);
        var envelope = JsonSerializer.Deserialize<YahooChartEnvelope>(json, JsonOpts);

        if (envelope?.Chart?.Error is { } err)
            throw new InvalidOperationException($"Yahoo Finance error [{err.Code}]: {err.Description}");

        var result = envelope?.Chart?.Result?.FirstOrDefault();
        if (result?.Timestamps is null || result.Indicators?.Quote is null)
        {
            _logger.LogWarning("Empty payload for {Ticker} {From:yyyy-MM-dd}→{To:yyyy-MM-dd}",
                request.Ticker, request.From, request.To);
            return [];
        }

        var quote = result.Indicators.Quote.FirstOrDefault();
        if (quote?.Close is null)
            return [];

        var points = new List<TimeSeriesPoint>(result.Timestamps.Count);
        for (var i = 0; i < result.Timestamps.Count; i++)
        {
            if (i >= quote.Close.Count || quote.Close[i] is not { } close)
                continue;
            points.Add(new TimeSeriesPoint(DateTimeOffset.FromUnixTimeSeconds(result.Timestamps[i]), close));
        }

        _logger.LogInformation("  {Ticker} [{Interval}] {From:yyyy-MM-dd}→{To:yyyy-MM-dd}: {Count} points",
            request.Ticker, request.Interval, request.From, request.To, points.Count);
        return points;
    }

    // ── Option chain (v7) ────────────────────────────────────────────────────────

    /// <summary>
    /// Returns all expiry timestamps available for <paramref name="symbol"/> on Yahoo Finance.
    /// Call once to discover the universe, then call <see cref="FetchOptionChainAsync"/>
    /// per desired expiry.
    /// </summary>
    internal async Task<IReadOnlyList<long>> GetOptionExpiriesAsync(
        string symbol, CancellationToken ct = default)
    {
        await EnsureSessionAsync(ct);
        var result = await FetchOptionChainPageAsync(symbol, expiryTimestamp: null, ct);
        return result?.ExpirationDates ?? [];
    }

    /// <summary>
    /// Fetches the full call and put chain for one expiry date.
    /// Returns null if Yahoo returns no data for that expiry.
    /// </summary>
    internal async Task<YahooOptionExpiry?> FetchOptionChainAsync(
        string symbol, long expiryTimestamp, CancellationToken ct = default)
    {
        await EnsureSessionAsync(ct);
        var result = await FetchOptionChainPageAsync(symbol, expiryTimestamp, ct);
        return result?.Options?.FirstOrDefault();
    }

    private async Task<YahooOptionChainResult?> FetchOptionChainPageAsync(
        string symbol, long? expiryTimestamp, CancellationToken ct)
    {
        var url = $"https://query1.finance.yahoo.com/v7/finance/options/{Uri.EscapeDataString(symbol)}" +
                  $"?crumb={Uri.EscapeDataString(_crumb!)}";
        if (expiryTimestamp.HasValue)
            url += $"&date={expiryTimestamp.Value}";

        _logger.LogDebug("GET option chain {Symbol} expiry={Expiry}", symbol, expiryTimestamp);

        var resp = await _http.GetAsync(url, ct);

        if (!resp.IsSuccessStatusCode)
        {
            var body = await resp.Content.ReadAsStringAsync(ct);
            throw new HttpRequestException(
                $"Yahoo Finance option chain HTTP {(int)resp.StatusCode} for {symbol}: {body}");
        }

        var json     = await resp.Content.ReadAsStringAsync(ct);
        var envelope = JsonSerializer.Deserialize<YahooOptionChainEnvelope>(json, JsonOpts);

        if (envelope?.OptionChain?.Error is { } err)
            throw new InvalidOperationException($"Yahoo Finance error [{err.Code}]: {err.Description}");

        return envelope?.OptionChain?.Result?.FirstOrDefault();
    }

    // ── Session ───────────────────────────────────────────────────────────────

    private async Task EnsureSessionAsync(CancellationToken ct)
    {
        if (_crumb is not null)
            return;

        await _sessionLock.WaitAsync(ct);
        try
        {
            if (_crumb is not null)
                return;

            await _http.GetAsync("https://fc.yahoo.com/", ct);

            var crumbResp = await _http.GetAsync(
                "https://query1.finance.yahoo.com/v1/test/getcrumb", ct);

            if (!crumbResp.IsSuccessStatusCode)
                throw new InvalidOperationException(
                    $"Failed to obtain Yahoo Finance crumb (HTTP {(int)crumbResp.StatusCode}). " +
                    "The session endpoint may have changed or your IP is being rate-limited.");

            _crumb = await crumbResp.Content.ReadAsStringAsync(ct);
            _logger.LogDebug("Yahoo Finance session ready");
        }
        finally
        {
            _sessionLock.Release();
        }
    }

    // Conservative per-request day limits; Yahoo's documented limits are:
    //   1h → 730d, but 729 avoids off-by-one edge cases.
    //   sub-hour → 60d, using 59 for the same reason.
    private static int MaxChunkDays(string interval) => interval switch
    {
        "1d" or "5d" or "1wk" or "1mo" or "3mo" => int.MaxValue / 2,
        "1h"                                      => 729,
        "30m" or "15m" or "5m" or "2m"           => 59,
        "1m"                                      => 6,
        _                                         => 59,
    };

    public ValueTask DisposeAsync()
    {
        _http.Dispose();
        _sessionLock.Dispose();
        return ValueTask.CompletedTask;
    }
}
