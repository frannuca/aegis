namespace DataLoader.Providers.Yahoo;

using System.Text.Json.Serialization;

internal sealed class YahooChartEnvelope
{
    [JsonPropertyName("chart")]
    public YahooChart? Chart { get; init; }
}

internal sealed class YahooChart
{
    [JsonPropertyName("result")]
    public List<YahooChartResult>? Result { get; init; }

    [JsonPropertyName("error")]
    public YahooApiError? Error { get; init; }
}

internal sealed class YahooChartResult
{
    [JsonPropertyName("timestamp")]
    public List<long>? Timestamps { get; init; }

    [JsonPropertyName("indicators")]
    public YahooIndicators? Indicators { get; init; }
}

internal sealed class YahooIndicators
{
    [JsonPropertyName("quote")]
    public List<YahooQuote>? Quote { get; init; }
}

internal sealed class YahooQuote
{
    [JsonPropertyName("close")]
    public List<double?>? Close { get; init; }
}

internal sealed class YahooApiError
{
    [JsonPropertyName("code")]
    public string? Code { get; init; }

    [JsonPropertyName("description")]
    public string? Description { get; init; }
}
