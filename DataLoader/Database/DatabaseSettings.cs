namespace DataLoader.Database;

public sealed class DatabaseSettings
{
    public string Host     { get; init; } = "localhost";
    public int    Port     { get; init; } = 5432;
    public string Database { get; init; } = "quant";
    public string Username { get; init; } = string.Empty;
    public string Password { get; init; } = string.Empty;

    public string ConnectionString =>
        $"Host={Host};Port={Port};Database={Database};Username={Username};Password={Password}";
}
