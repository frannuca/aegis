namespace DataLoader.Models;

public sealed record TimeSeriesPoint(DateTimeOffset Time, double Value);
