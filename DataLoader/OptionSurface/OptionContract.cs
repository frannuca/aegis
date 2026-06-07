namespace DataLoader.OptionSurface;

using System.Globalization;

/// <summary>
/// A single listed option contract uniquely identified by (future, expiry, type, strike).
///
/// Ticker  — internal storage key in market_data.universe.
///           Format: {FUTURE}-{YYYYMM}-{C|P}-{STRIKE}  e.g. TY-202706-P-112.75
///
/// TickerForProvider — provider-specific format used in the HTTP request.
///           Yahoo Finance: {root}{YYMMDD}{C|P}{strike×1000, 8 digits}
///           e.g. ZN270625P00112750
/// </summary>
public sealed record OptionContract(
    string   FutureCode,
    DateOnly Expiry,
    string   Type,          // "C" or "P"
    double   Strike,
    double   StrikeStep,
    string   Provider)
{
    /// <summary>Internal DB key — provider-agnostic.</summary>
    public string Ticker =>
        $"{FutureCode}-{Expiry:yyyyMM}-{Type}-{FormatStrike(Strike, StrikeStep)}";

    /// <summary>
    /// Returns the ticker as the given provider expects it in its API.
    /// Falls back to <see cref="Ticker"/> when no translation is defined.
    /// </summary>
    public string TickerForProvider(string provider, string? yahooOptionsSymbol) =>
        provider == "yahoo" && yahooOptionsSymbol is not null
            ? BuildYahooTicker(yahooOptionsSymbol)
            : Ticker;

    // Yahoo Finance option ticker format:
    //   {root}{YYMMDD}{C|P}{strike × 1000, zero-padded to 8 digits}
    // Expiry date: last Friday of the contract month (standard CME quarterly rule).
    private string BuildYahooTicker(string root)
    {
        var expiry     = LastFridayOfMonth(Expiry);
        var strikeInt  = (long)Math.Round(Strike * 1000);
        return $"{root}{expiry:yyMMdd}{Type}{strikeInt:D8}";
    }

    private static DateOnly LastFridayOfMonth(DateOnly month)
    {
        var d = new DateOnly(month.Year, month.Month,
            DateTime.DaysInMonth(month.Year, month.Month));
        while (d.DayOfWeek != DayOfWeek.Friday)
            d = d.AddDays(-1);
        return d;
    }

    // Minimum decimal places needed to represent the tick size exactly (≥ 2).
    private static string FormatStrike(double strike, double step)
    {
        for (var d = 0; d <= 8; d++)
        {
            var scaled = step * Math.Pow(10, d);
            if (Math.Abs(scaled - Math.Round(scaled)) < 1e-9)
                return strike.ToString($"F{Math.Max(2, d)}", CultureInfo.InvariantCulture);
        }
        return strike.ToString("F4", CultureInfo.InvariantCulture);
    }
}
