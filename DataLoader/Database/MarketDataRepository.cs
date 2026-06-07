namespace DataLoader.Database;

using DataLoader.Models;
using Microsoft.Extensions.Logging;
using Npgsql;
using NpgsqlTypes;

public sealed class MarketDataRepository(DatabaseSettings settings, ILogger<MarketDataRepository> logger)
{
    private readonly string _connectionString = settings.ConnectionString;

    /// <summary>
    /// Returns the universe.id for the given (provider, ticker), inserting a new row if it does not exist.
    /// </summary>
    public async Task<long> GetOrCreateSecurityIdAsync(string ticker, string provider, CancellationToken ct = default)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync(ct);

        // DO UPDATE SET ticker = ticker is a no-op that forces RETURNING to fire on conflict
        await using var cmd = new NpgsqlCommand(
            """
            INSERT INTO market_data.universe (ticker, provider)
            VALUES (@ticker, @provider)
            ON CONFLICT (provider, ticker) DO UPDATE SET ticker = EXCLUDED.ticker
            RETURNING id
            """, conn);

        cmd.Parameters.AddWithValue("ticker", ticker);
        cmd.Parameters.AddWithValue("provider", provider);

        var id = (long)(await cmd.ExecuteScalarAsync(ct))!;
        logger.LogDebug("Security {Provider}:{Ticker} → universe.id = {Id}", provider, ticker, id);
        return id;
    }

    /// <summary>
    /// Bulk-upserts time-series rows.  Uses a COPY → temp-table → INSERT ON CONFLICT pipeline
    /// so it stays fast even for large intraday datasets.
    /// </summary>
    public async Task<int> UpsertTimeSeriesAsync(
        long securityId,
        IReadOnlyList<TimeSeriesPoint> points,
        CancellationToken ct = default)
    {
        if (points.Count == 0)
            return 0;

        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync(ct);
        await using var tx = await conn.BeginTransactionAsync(ct);

        // Staging table lives only for this transaction
        await using (var createCmd = new NpgsqlCommand(
                         """
                         CREATE TEMP TABLE ts_staging (
                             time        TIMESTAMPTZ      NOT NULL,
                             security_id BIGINT           NOT NULL,
                             value       DOUBLE PRECISION  NOT NULL
                         ) ON COMMIT DROP
                         """, conn, tx))
        {
            await createCmd.ExecuteNonQueryAsync(ct);
        }

        // Binary COPY into the staging table (fast path)
        await using (var writer = await conn.BeginBinaryImportAsync(
                         "COPY ts_staging (time, security_id, value) FROM STDIN (FORMAT BINARY)", ct))
        {
            foreach (var p in points)
            {
                await writer.StartRowAsync(ct);
                await writer.WriteAsync(p.Time, NpgsqlDbType.TimestampTz, ct);
                await writer.WriteAsync(securityId, NpgsqlDbType.Bigint, ct);
                await writer.WriteAsync(p.Value, NpgsqlDbType.Double, ct);
            }

            await writer.CompleteAsync(ct);
        }

        // Upsert from staging into the permanent table
        int count;
        await using (var upsertCmd = new NpgsqlCommand(
                         """
                         INSERT INTO market_data.time_series (security_id, time, value)
                         SELECT security_id, time, value FROM ts_staging
                         ON CONFLICT (security_id, time) DO UPDATE SET value = EXCLUDED.value
                         """, conn, tx))
        {
            count = await upsertCmd.ExecuteNonQueryAsync(ct);
        }

        await tx.CommitAsync(ct);
        logger.LogInformation("Upserted {Count} rows for security_id={Id}", count, securityId);
        return count;
    }

    /// <summary>
    /// Returns the most recent value in time_series for the given (provider, ticker),
    /// or null if no data exists. Used for ATM auto-resolution.
    /// </summary>
    public async Task<double?> GetLatestPriceAsync(
        string ticker, string provider, CancellationToken ct = default)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync(ct);

        await using var cmd = new NpgsqlCommand(
            """
            SELECT ts.value
            FROM market_data.time_series ts
            JOIN market_data.universe u ON u.id = ts.security_id
            WHERE u.ticker = @ticker AND u.provider = @provider
            ORDER BY ts.time DESC
            LIMIT 1
            """, conn);

        cmd.Parameters.AddWithValue("ticker", ticker);
        cmd.Parameters.AddWithValue("provider", provider);

        var result = await cmd.ExecuteScalarAsync(ct);
        return result is double v ? v : null;
    }
}