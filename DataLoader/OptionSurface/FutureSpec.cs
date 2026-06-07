namespace DataLoader.OptionSurface;

/// <summary>
/// Static description of a bond futures contract relevant for options and SABR calibration.
/// </summary>
/// <param name="Code">Internal code used in option tickers (e.g. RX, DU, TY).</param>
/// <param name="Provider">Data provider name for options on this future.</param>
/// <param name="StrikeStep">Minimum strike increment on the exchange.</param>
/// <param name="ExpiryMonths">Calendar months that carry listed expiries (1-12).</param>
/// <param name="YahooTicker">
///   Yahoo Finance ticker for the continuous futures price used to auto-resolve ATM.
///   Null for Eurex futures (not available on Yahoo) — ATM must be supplied explicitly.
/// </param>
/// <param name="YahooOptionsSymbol">
///   Yahoo Finance root symbol for options on this future (e.g. "ZN" for TY).
///   Combined with the computed expiry date and strike to build option tickers in Yahoo's
///   format: {symbol}{YYMMDD}{C|P}{strike×1000, 8 digits}.  Null when Yahoo Finance
///   does not carry options for this future.
/// </param>
public sealed record FutureSpec(
    string  Code,
    string  Provider,
    double  StrikeStep,
    int[]   ExpiryMonths,
    string? YahooTicker,
    string? YahooOptionsSymbol = null);

public static class FutureSpecs
{
    private static readonly Dictionary<string, FutureSpec> Registry =
        new(StringComparer.OrdinalIgnoreCase)
        {
            // ── Eurex ─────────────────────────────────────────────────────────────
            // Options: OGBL (RX), OGBS (DU), OGBM (OE)
            // ATM must be supplied via --atm (Eurex futures are not on Yahoo Finance)
            ["RX"] = new("RX", "eurex", 0.50, [3, 6, 9, 12], null),   // Euro-Bund
            ["DU"] = new("DU", "eurex", 0.25, [3, 6, 9, 12], null),   // Euro-Schatz
            ["OE"] = new("OE", "eurex", 0.25, [3, 6, 9, 12], null),   // Euro-Bobl

            // ── CME ───────────────────────────────────────────────────────────────
            // ATM auto-resolved from Yahoo Finance when --atm is omitted.
            // YahooOptionsSymbol: root used to build Yahoo option tickers at fetch time.
            ["TY"] = new("TY", "cme", 0.25,   [3, 6, 9, 12], "ZN=F", "ZN"),  // 10Y T-Note
            ["FV"] = new("FV", "cme", 0.125,  [3, 6, 9, 12], "ZF=F", "ZF"),  // 5Y T-Note
            ["TU"] = new("TU", "cme", 0.0625, [3, 6, 9, 12], "ZT=F", "ZT"),  // 2Y T-Note
            ["US"] = new("US", "cme", 0.50,   [3, 6, 9, 12], "ZB=F", "ZB"),  // 30Y T-Bond
        };

    public static FutureSpec Get(string code) =>
        Registry.TryGetValue(code, out var s) ? s
            : throw new ArgumentException(
                $"Unknown future '{code}'. Known codes: {string.Join(", ", Registry.Keys)}");

    public static IEnumerable<string> KnownCodes => Registry.Keys;
}
