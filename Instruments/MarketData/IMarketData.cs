namespace Aegis.Instruments;

/// <summary>
/// Common market-data interface shared by FX and equity MC pricers.
///
/// Mapping:
///   FxMarketData  → DiscountCurve = DomesticRate,  CarryCurve = ForeignRate
///   EqMarketData  → DiscountCurve = RiskFreeRate,  CarryCurve = DividendYield
///
/// Dividends is non-empty only for equity market data.
/// </summary>
public interface IMarketData
{
    double Spot { get; }
    Pillars DiscountCurve { get; }
    Pillars CarryCurve { get; }
    VolatilitySurface VolSurface { get; }
    IReadOnlyList<CashDividend> Dividends { get; }
}
