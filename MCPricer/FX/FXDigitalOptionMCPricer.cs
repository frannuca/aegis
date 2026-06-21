using Aegis.Instruments;
using RandomSimulator;
using NodaTime;

namespace MCPricer.FX;

/// <summary>
/// FX digital ("binary") option pricer — cash-or-nothing or asset-or-nothing,
/// call or put.
///
/// In-the-money condition uses the same convention as the vanilla payoff:
///   call ⇒ S(T) > K,   put ⇒ S(T) < K
///
/// Payoff at T:
///   CASH_OR_NOTHING:  payout · 1{S(T) ITM}
///   ASSET_OR_NOTHING: S(T)   · 1{S(T) ITM}
///
/// Only the terminal spot S(T) = spotPath[^1] is used; intermediate values are
/// simulated but irrelevant to the payoff (European, terminal-only).
///
/// Analytical benchmark: <see cref="AnalyticalPricers.DigitalBlackScholes"/>
/// (Garman-Kohlhagen carry convention: rate = r_d, carry = r_f).
///
/// Edge cases handled:
///   σ = 0: path is deterministic; payoff is determined by the forward
///   F = S·e^{(r_d − r_f)·T} versus K (see DigitalBlackScholes XML doc).
/// </summary>
public sealed class FXDigitalOptionMCPricer : GbmOptionMCPricer
{
    private readonly double                _strike;
    private readonly double                _phi;   // +1 for call, −1 for put
    private readonly double                _payout;
    private readonly DigitalSettlementType _settlementType;

    public FXDigitalOptionMCPricer(
        Option         option,
        LocalDate       valuationDate,
        double         volatility,
        SimulationCube cube)
        : base(option, valuationDate, volatility, cube)
    {
        if (option.Strike <= 0)
            throw new ArgumentException("Strike must be positive.", nameof(option));
        if (option.KindCase != Option.KindOneofCase.Digital)
            throw new ArgumentException("Option.Kind must be Digital.", nameof(option));

        var digital = option.Digital;
        if (digital.SettlementType == DigitalSettlementType.Unspecified)
            throw new ArgumentException("DigitalOption.SettlementType must be specified.", nameof(option));
        if (digital.SettlementType == DigitalSettlementType.CashOrNothing && digital.Payout <= 0)
            throw new ArgumentException("DigitalOption.Payout must be positive for cash-or-nothing settlement.", nameof(option));

        _strike         = option.Strike;
        _phi            = option.OptionType == OptionType.Call ? 1.0 : -1.0;
        _payout         = digital.Payout;
        _settlementType = digital.SettlementType;
    }

    protected override double EvaluatePayoff(ReadOnlySpan<double> spotPath)
    {
        var terminal = spotPath[^1];
        if (_phi * (terminal - _strike) <= 0.0)
            return 0.0;

        return _settlementType == DigitalSettlementType.AssetOrNothing ? terminal : _payout;
    }

    protected override MCBasePricer CreateBumped(BumpType bump, double epsilon)
    {
        var vol = Volatility;
        switch (bump)
        {
            case BumpType.VolUp:   vol += epsilon;                          break;
            case BumpType.VolDown: vol  = Math.Max(0.0, vol - epsilon);    break;
            default: throw new NotSupportedException($"Bump {bump} not supported by {GetType().Name}. Use ComputeGreeks(market, ...) for spot and rate bumps.");
        }
        return new FXDigitalOptionMCPricer(OptionDef, ValuationDate, vol, Cube);
    }
}
