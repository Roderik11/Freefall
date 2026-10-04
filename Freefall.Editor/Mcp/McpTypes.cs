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

    public record BatchItem(
        [property: Description("Prefab (preferred) or Mesh GUID, or '@Exact Name' (resolves the prefab of that name)")] string Guid,
        string? Name = null,
        [property: Description("Local to the group/parent when one is given, else world")] Vec3? Position = null,
        [property: Description("Quaternion; wins over rotationEuler")] Quat? Rotation = null,
        [property: Description("Euler degrees (x=pitch, y=yaw, z=roll)")] Vec3? RotationEuler = null,
        Vec3? Scale = null);

    public record BatchGroup(
        [property: Description("Name of the new parent entity")] string Name,
        [property: Description("World position of the group")] Vec3? Position = null,
        [property: Description("Euler degrees of the group (yaw turns the whole assembly)")] Vec3? RotationEuler = null);

    public enum ScreenshotFormat { Jpeg, Png }
}
