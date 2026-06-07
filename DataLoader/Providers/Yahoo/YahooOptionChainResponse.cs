namespace DataLoader.Providers.Yahoo;

using System.Text.Json.Serialization;

internal sealed class YahooOptionChainEnvelope
{
    [JsonPropertyName("optionChain")]
    public YahooOptionChain? OptionChain { get; init; }
}

internal sealed class YahooOptionChain
{
    [JsonPropertyName("result")]
    public List<YahooOptionChainResult>? Result { get; init; }

    [JsonPropertyName("error")]
    public YahooApiError? Error { get; init; }
}

internal sealed class YahooOptionChainResult
{
    [JsonPropertyName("underlyingSymbol")]
    public string? UnderlyingSymbol { get; init; }

    [JsonPropertyName("expirationDates")]
    public List<long>? ExpirationDates { get; init; }

    [JsonPropertyName("options")]
    public List<YahooOptionExpiry>? Options { get; init; }
}

internal sealed class YahooOptionExpiry
{
    [JsonPropertyName("expirationDate")]
    public long ExpirationDate { get; init; }

    [JsonPropertyName("calls")]
    public List<YahooOptionQuote>? Calls { get; init; }

    [JsonPropertyName("puts")]
    public List<YahooOptionQuote>? Puts { get; init; }
}

internal sealed class YahooOptionQuote
{
    [JsonPropertyName("contractSymbol")]
    public string? ContractSymbol { get; init; }

    [JsonPropertyName("strike")]
    public double Strike { get; init; }

    [JsonPropertyName("lastPrice")]
    public double LastPrice { get; init; }

    [JsonPropertyName("bid")]
    public double Bid { get; init; }

    [JsonPropertyName("ask")]
    public double Ask { get; init; }

    [JsonPropertyName("volume")]
    public int? Volume { get; init; }

    [JsonPropertyName("openInterest")]
    public int? OpenInterest { get; init; }

    [JsonPropertyName("impliedVolatility")]
    public double ImpliedVolatility { get; init; }

    [JsonPropertyName("expiration")]
    public long Expiration { get; init; }
}
