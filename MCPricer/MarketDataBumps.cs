using Aegis.Instruments;

namespace MCPricer;

/// <summary>
/// Produces bumped copies of IMarketData for bump-and-reprice Greeks.
/// Each method clones the concrete market-data object and shifts one parameter.
/// </summary>
public static class MarketDataBumps
{
    public static IMarketData BumpSpot(IMarketData market, double eps)
    {
        switch (market)
        {
            case FxMarketData fx:
            {
                var c = fx.Clone();
                c.Spot += eps;
                return c;
            }
            case EqMarketData eq:
            {
                var c = eq.Clone();
                c.Spot += eps;
                return c;
            }
            default:
                throw new NotSupportedException(
                    $"BumpSpot is not implemented for market data type {market.GetType().Name}.");
        }
    }

    public static IMarketData BumpDiscountCurve(IMarketData market, double eps)
    {
        switch (market)
        {
            case FxMarketData fx:
            {
                var c = fx.Clone();
                c.DomesticRate = ZeroCurve.Shift(c.DomesticRate, eps);
                return c;
            }
            case EqMarketData eq:
            {
                var c = eq.Clone();
                c.RiskFreeRate = ZeroCurve.Shift(c.RiskFreeRate, eps);
                return c;
            }
            default:
                throw new NotSupportedException(
                    $"BumpDiscountCurve is not implemented for market data type {market.GetType().Name}.");
        }
    }

    public static IMarketData BumpCarryCurve(IMarketData market, double eps)
    {
        switch (market)
        {
            case FxMarketData fx:
            {
                var c = fx.Clone();
                c.ForeignRate = ZeroCurve.Shift(c.ForeignRate, eps);
                return c;
            }
            case EqMarketData eq:
            {
                var c = eq.Clone();
                c.DividendYield = ZeroCurve.Shift(c.DividendYield, eps);
                return c;
            }
            default:
                throw new NotSupportedException(
                    $"BumpCarryCurve is not implemented for market data type {market.GetType().Name}.");
        }
    }
}
