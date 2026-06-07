namespace DataLoader.Providers;

using DataLoader.Providers.Yahoo;
using Microsoft.Extensions.Logging;

public static class ProviderFactory
{
    public static IMarketDataProvider Create(string name, ILoggerFactory loggerFactory) =>
        name.ToLowerInvariant() switch
        {
            "yahoo" => new YahooFinanceProvider(loggerFactory.CreateLogger<YahooFinanceProvider>()),
            _       => throw new NotSupportedException(
                           $"Unknown provider '{name}'. Supported: yahoo")
        };
}
