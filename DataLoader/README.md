# DataLoader

CLI tool that downloads FX market data from external providers and stores it in the Aegis PostgreSQL database.

---

## Prerequisites

| Requirement | Version |
|---|---|
| .NET SDK | 10.0+ (use `/usr/local/share/dotnet/dotnet`) |
| PostgreSQL | 14+ with `market_data` schema applied |

> **Note:** The shell `PATH` on this machine defaults to the VS Code SDK (`~/.vscode-dotnet-sdk`).  
> Prefix every `dotnet` command with `/usr/local/share/dotnet/dotnet`, or add it to your PATH:
> ```bash
> export PATH="/usr/local/share/dotnet:$PATH"
> ```

---

## Configuration

DB credentials live in `DataLoader/appsettings.json`:

```json
{
  "Database": {
    "Host": "localhost",
    "Port": 5432,
    "Database": "quant",
    "Username": "quant_app",
    "Password": "…"
  }
}
```

Environment variables override the file (useful for CI / Docker):

```bash
export DATALOADER_Database__Username=quant_app
export DATALOADER_Database__Password=secret
```

---

## Running

### Option A — development (from repo root)

```bash
/usr/local/share/dotnet/dotnet run --project DataLoader/DataLoader.csproj -- <command> [options]
```

### Option B — self-contained binary (recommended for daily use)

Build once:

```bash
/usr/local/share/dotnet/dotnet publish DataLoader/DataLoader.csproj \
  -c Release \
  -r osx-arm64 \
  --self-contained true \
  -p:PublishSingleFile=true \
  -o ~/bin
```

Then just call `dataloader` from anywhere (assuming `~/bin` is on your PATH):

```bash
dataloader fetch --ticker EURUSD=X --interval 1d --from 2024-01-01 --to 2024-12-31
```

> For Intel Mac use `-r osx-x64`; for Linux `-r linux-x64`.

---

## Commands

### `fetch` — single ticker

```
dataloader fetch --ticker <TICKER> [--interval <INTERVAL>] [--from <DATE>] [--to <DATE>] [--provider <PROVIDER>]
```

| Option | Default | Description |
|---|---|---|
| `--ticker` | *(required)* | Yahoo Finance ticker, e.g. `EURUSD=X` |
| `--interval` | `1d` | Bar size — see table below |
| `--from` | 1 year ago | Start date UTC, e.g. `2024-01-01` |
| `--to` | today | End date UTC, e.g. `2024-12-31` |
| `--provider` | `yahoo` | Data provider |

**Examples:**

```bash
# One year of daily EUR/USD
dataloader fetch --ticker EURUSD=X --interval 1d --from 2024-01-01 --to 2024-12-31

# Last 30 days of hourly GBP/USD
dataloader fetch --ticker GBPUSD=X --interval 1h --from 2024-05-01 --to 2024-05-31

# 5-minute intraday USD/JPY (max 60 days lookback)
dataloader fetch --ticker USDJPY=X --interval 5m --from 2024-05-20 --to 2024-06-05
```

---

### `fetch-batch` — multiple tickers from a JSON file

```
dataloader fetch-batch --config <FILE> [--from <DATE>] [--to <DATE>]
```

`--from` / `--to` in the command line are used as defaults; they are overridden per-job by `from`/`to` in the JSON file.

**Example config (`batch-example.json`):**

```json
{
  "from": "2024-01-01T00:00:00Z",
  "to":   "2024-12-31T23:59:59Z",
  "jobs": [
    { "ticker": "EURUSD=X", "interval": "1d" },
    { "ticker": "GBPUSD=X", "interval": "1d" },
    { "ticker": "USDJPY=X", "interval": "1d" },
    { "ticker": "EURUSD=X", "interval": "1h" }
  ]
}
```

```bash
dataloader fetch-batch --config DataLoader/batch-example.json
```

---

## Yahoo Finance — FX tickers & interval limits

**FX ticker format:** `<BASE><QUOTE>=X`

| Pair | Ticker |
|---|---|
| EUR/USD | `EURUSD=X` |
| GBP/USD | `GBPUSD=X` |
| USD/JPY | `USDJPY=X` |
| AUD/USD | `AUDUSD=X` |
| USD/CHF | `USDCHF=X` |
| USD/CAD | `USDCAD=X` |

**Interval lookback limits (Yahoo Finance):**

| Interval | Max lookback |
|---|---|
| `1d` | unlimited |
| `1h` | 730 days |
| `30m`, `15m`, `5m` | 60 days |
| `2m` | 60 days |
| `1m` | 7 days |

---

## Database schema

New tickers are automatically registered in `market_data.universe` on first fetch.  
Subsequent fetches upsert into `market_data.time_series` (PK: `security_id, time`), so re-running a fetch is safe.

```
market_data.universe        market_data.time_series
─────────────────────       ───────────────────────────────
id        bigserial PK      security_id  bigint FK → universe.id
ticker    varchar            time         timestamptz
provider  varchar            value        double precision
```

---

## Adding a new provider

1. Create `Providers/<Name>/<Name>Provider.cs` implementing `IMarketDataProvider`.
2. Add the provider name to the `switch` expression in `Program.cs → RunFetch`.
