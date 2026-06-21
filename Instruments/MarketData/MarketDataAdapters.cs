namespace Aegis.Instruments;

public partial class FxMarketData : IMarketData
{
    Pillars IMarketData.DiscountCurve => DomesticRate;
    Pillars IMarketData.CarryCurve    => ForeignRate;
    IReadOnlyList<CashDividend> IMarketData.Dividends => [];
}

public partial class EqMarketData : IMarketData
{
    Pillars IMarketData.DiscountCurve => RiskFreeRate;
    Pillars IMarketData.CarryCurve    => DividendYield;
    // Dividends is the proto RepeatedField<CashDividend>, which implements IReadOnlyList<T>.
    IReadOnlyList<CashDividend> IMarketData.Dividends => Dividends;
}
