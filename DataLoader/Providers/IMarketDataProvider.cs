namespace DataLoader.Providers;

using DataLoader.Models;

public interface IMarketDataProvider : IAsyncDisposable
{
    string Name { get; }
    Task<IReadOnlyList<TimeSeriesPoint>> FetchAsync(FetchRequest request, CancellationToken ct = default);
}
