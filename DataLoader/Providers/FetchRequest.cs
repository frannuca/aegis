namespace DataLoader.Providers;

public sealed record FetchRequest(
    string Ticker,
    string Provider,
    string Interval,
    DateTimeOffset From,
    DateTimeOffset To);
