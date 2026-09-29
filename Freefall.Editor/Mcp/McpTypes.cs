using System.ComponentModel;

namespace Freefall.Editor.Mcp
{
    // Argument shapes shared by the MCP tools. Serialized camelCase ({x,y,z}) to match the command routes.

    [Description("3D vector")]
    public record Vec3(float X, float Y, float Z);

    [Description("Quaternion (x,y,z,w)")]
    public record Quat(float X, float Y, float Z, float W);

    [Description("World-space ground position (x, z)")]
    public record PointXZ(float X, float Z);

    [Description("Inclusive range; omit either end for unbounded/default")]
    public record MinMax(float? Min = null, float? Max = null);

    [Description("Pixel rectangle")]
    public record PixelRect(int X, int Y, int Width, int Height);

    public record WeightedAsset(
        [property: Description("Prefab (preferred) or Mesh GUID")] string Guid,
        [property: Description("Relative selection weight, default 1")] float? Weight = null);

    public record ScatterArea(
        [property: Description("Circle center on the ground plane")] PointXZ Center,
        [property: Description("Circle radius in world units, default 100")] float? Radius = null);

    public enum ScreenshotFormat { Jpeg, Png }
}
