namespace RandomSimulator;

using System.Globalization;
using System.Numerics;
using System.Reflection;

public sealed class SobolSequence
{
    private const int WordBits = 32;
    private const double TwoPow32Inv = 1.0 / 4_294_967_296.0;

    // Change this if your embedded file name is different.
    private const string EmbeddedDirectionFileName = "new-joe-kuo-6.21201";

    private static readonly uint[][] V = LoadEmbeddedDirectionVectors();

    private readonly int _dim;
    private readonly uint[] _x;
    private ulong _n;

    public static int MaxDimensions => V.Length;

    public SobolSequence(int dimension)
    {
        if (dimension < 1 || dimension > MaxDimensions)
            throw new ArgumentOutOfRangeException(
                nameof(dimension),
                $"Sobol dimension must be in [1, {MaxDimensions}].");

        _dim = dimension;
        _x = new uint[dimension];
        _n = 0;
    }

    public void NextPoint(Span<double> point)
    {
        if (point.Length < _dim)
            throw new ArgumentException($"Buffer too small: need {_dim}, got {point.Length}.");

        _n++;

        var c = TrailingZeros(_n);

        if (c >= WordBits)
            throw new InvalidOperationException(
                $"32-bit Sobol generator supports at most 2^{WordBits} - 1 points.");

        for (var d = 0; d < _dim; d++)
        {
            _x[d] ^= V[d][c];
            point[d] = _x[d] * TwoPow32Inv;
        }
    }

    public void Reset()
    {
        _x.AsSpan().Clear();
        _n = 0;
    }

    private static uint[][] LoadEmbeddedDirectionVectors()
    {
        var assembly = typeof(SobolSequence).Assembly;

        var resourceName = assembly
            .GetManifestResourceNames()
            .SingleOrDefault(x =>
                x.EndsWith(EmbeddedDirectionFileName, StringComparison.OrdinalIgnoreCase));

        if (resourceName is null)
        {
            var available = string.Join(Environment.NewLine, assembly.GetManifestResourceNames());

            throw new InvalidOperationException(
                $"Embedded Sobol direction file '{EmbeddedDirectionFileName}' was not found. " +
                $"Available embedded resources:{Environment.NewLine}{available}");
        }

        using var stream = assembly.GetManifestResourceStream(resourceName)
            ?? throw new InvalidOperationException(
                $"Could not open embedded Sobol direction resource '{resourceName}'.");

        using var reader = new StreamReader(stream);

        var rows = ParseDirectionRows(reader);

        return BuildDirectionVectorsFromRows(rows);
    }

    private static List<SobolDirectionRow> ParseDirectionRows(TextReader reader)
    {
        var rows = new List<SobolDirectionRow>();

        string? rawLine;

        while ((rawLine = reader.ReadLine()) is not null)
        {
            var line = rawLine.Trim();

            if (line.Length == 0)
                continue;

            if (line.StartsWith("#", StringComparison.Ordinal))
                continue;

            if (line.StartsWith("d", StringComparison.OrdinalIgnoreCase))
                continue;

            var parts = line.Split(
                (char[]?)null,
                StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            if (parts.Length < 4)
                continue;

            var dimension = int.Parse(parts[0], CultureInfo.InvariantCulture);
            var s = int.Parse(parts[1], CultureInfo.InvariantCulture);
            var a = uint.Parse(parts[2], CultureInfo.InvariantCulture);

            if (s < 1 || s > WordBits)
                throw new InvalidOperationException(
                    $"Dimension {dimension}: invalid polynomial degree s={s}.");

            if (parts.Length != 3 + s)
                throw new InvalidOperationException(
                    $"Dimension {dimension}: expected {s} m-values, got {parts.Length - 3}.");

            var m = new uint[s];

            for (var i = 0; i < s; i++)
                m[i] = uint.Parse(parts[3 + i], CultureInfo.InvariantCulture);

            rows.Add(new SobolDirectionRow(dimension, s, a, m));
        }

        if (rows.Count == 0)
            throw new InvalidOperationException("Embedded Sobol direction file contains no usable rows.");

        return rows;
    }

    private static uint[][] BuildDirectionVectorsFromRows(IReadOnlyList<SobolDirectionRow> rows)
    {
        var ordered = rows
            .OrderBy(r => r.Dimension)
            .ToList();

        if (ordered[0].Dimension != 2)
            throw new InvalidOperationException(
                $"Sobol direction file must start at dimension 2. Found {ordered[0].Dimension}.");

        for (var i = 0; i < ordered.Count; i++)
        {
            var expected = i + 2;

            if (ordered[i].Dimension != expected)
                throw new InvalidOperationException(
                    $"Sobol direction file has missing dimension {expected}. Found {ordered[i].Dimension}.");
        }

        var vectors = new uint[ordered.Count + 1][];

        vectors[0] = BuildDimensionOneDirectionVector();

        for (var i = 0; i < ordered.Count; i++)
        {
            var row = ordered[i];
            vectors[i + 1] = ComputeDirectionVector(row.S, row.A, row.M);
        }

        return vectors;
    }

    private static uint[] BuildDimensionOneDirectionVector()
    {
        var v = new uint[WordBits];

        for (var k = 0; k < WordBits; k++)
            v[k] = 1u << (WordBits - 1 - k);

        return v;
    }

    private static uint[] ComputeDirectionVector(int s, uint a, uint[] m)
    {
        var v = new uint[WordBits];

        for (var k = 0; k < s; k++)
        {
            if ((m[k] & 1u) == 0)
                throw new ArgumentException($"m[{k}] must be odd.");

            if (m[k] >= (1u << (k + 1)))
                throw new ArgumentException($"m[{k}] must be smaller than 2^{k + 1}.");

            v[k] = m[k] << (WordBits - 1 - k);
        }

        for (var k = s; k < WordBits; k++)
        {
            var value = v[k - s] ^ (v[k - s] >> s);

            for (var j = 1; j < s; j++)
            {
                if (((a >> (j - 1)) & 1u) != 0)
                    value ^= v[k - j];
            }

            v[k] = value;
        }

        return v;
    }

    private static int TrailingZeros(ulong n) =>
        BitOperations.TrailingZeroCount(n);

    private sealed record SobolDirectionRow(
        int Dimension,
        int S,
        uint A,
        uint[] M);
}