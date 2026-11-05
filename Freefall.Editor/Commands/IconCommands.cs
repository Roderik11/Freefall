using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using Freefall.Assets;
using Freefall.Graphics;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// POST /api/icon/render — draw an item's meshes into a PNG with a transparent background (Graphics.IconRenderer):
    /// a studio of lights, orthographic, framed to fill width × height.
    /// Body: {"width", "height", "out": absolute .png path,
    ///        "parts": [{"mesh": guid, "materials": [guid or null by slot], "position": [x,y,z], "rotation": [x,y,z,w],
    ///                   "scale": [x,y,z] or a number, "side": "Both"|"Positive"|"Negative", "sideCut": 0,
    ///                   "alphaTest": bool (default: on for a skinned material, as the skinned G-buffer cuts)}],
    ///        "orient": "long" (stand a long item up, tip on top, flat side to the camera) or "none",
    ///        or "principal" (stand it along the way its vertices spread most, the end towards "upHint" [x,y,z] on top,
    ///        the side nearest "frontHint" to the camera: a vambrace on a bent arm),
    ///        "up": "near" (the end nearer the grip on top: a staff), "front": "near" (the side nearer the grip to the
    ///        camera: a curved shield), "yaw", "pitch", "roll" (degrees, after orient), "margin" (0.04),
    ///        "exposure" (1), "interior" (1: how dark insides go), "supersample" (4)}
    /// Answers the path, the size and the share of the picture covered.
    /// </summary>
    [CommandRoute("POST", "/api/icon/render")]
    public class RenderIconCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required: {\"width\", \"height\", \"out\", \"parts\": [...]}");
            using var doc = context.ParseBody();
            var root = doc.RootElement;
            try
            {
                var parts = IconRequest.Parts(root);
                int width = root.TryGetProperty("width", out var w) ? w.GetInt32() : 128;
                int height = root.TryGetProperty("height", out var h) ? h.GetInt32() : 128;
                string? output = root.TryGetProperty("out", out var o) ? o.GetString() : null;
                if (string.IsNullOrEmpty(output) || !Path.IsPathRooted(output))
                    return CommandResult.BadRequest("'out' must be an absolute .png path");

                var view = IconRequest.View(root, parts);
                var pixels = IconRenderer.Shared.Render(parts, view, width, height);
                if (pixels == null)
                    return CommandResult.Error(500, "Nothing was drawn (no mesh on the GPU, or the device was lost)");

                int covered = 0;
                for (int i = 3; i < pixels.Length; i += 4) if (pixels[i] > 0) covered++;
                TextureReadback.SavePng(output, pixels, width, height);
                return CommandResult.Json(new { path = output, width, height, coverage = Math.Round((double)covered / (width * height), 4) });
            }
            catch (Exception ex)
            {
                return CommandResult.BadRequest(ex.GetBaseException().Message);
            }
        }
    }

    /// <summary>
    /// POST /api/icon/measure — the bounds of each part as /api/icon/render would draw it, in the item's space (after
    /// its transform, its kept half only): {"parts": [{"min", "max"}], "min", "max"}. For placing the halves of a pair.
    /// </summary>
    [CommandRoute("POST", "/api/icon/measure")]
    public class MeasureIconCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required: {\"parts\": [...]}");
            using var doc = context.ParseBody();
            try
            {
                var parts = IconRequest.Parts(doc.RootElement);
                var each = parts.Select(p => IconRenderer.Bounds(new[] { p }, Matrix4x4.Identity, out var mn, out var mx)
                    ? new { min = new[] { mn.X, mn.Y, mn.Z }, max = new[] { mx.X, mx.Y, mx.Z } } : null).ToList();
                IconRenderer.Bounds(parts, Matrix4x4.Identity, out var min, out var max);
                return CommandResult.Json(new { parts = each, min = new[] { min.X, min.Y, min.Z }, max = new[] { max.X, max.Y, max.Z } });
            }
            catch (Exception ex)
            {
                return CommandResult.BadRequest(ex.GetBaseException().Message);
            }
        }
    }

    internal static class IconRequest
    {
        public static List<IconPart> Parts(JsonElement root)
        {
            if (!root.TryGetProperty("parts", out var list) || list.ValueKind != JsonValueKind.Array)
                throw new InvalidOperationException("'parts' must be an array of {\"mesh\", \"materials\", ...}");
            var parts = new List<IconPart>();
            foreach (var item in list.EnumerateArray())
            {
                string? meshGuid = item.TryGetProperty("mesh", out var m) ? m.GetString() : null;
                var mesh = string.IsNullOrEmpty(meshGuid) ? null : Engine.Assets.LoadByGuid<Mesh>(meshGuid);
                if (mesh == null) throw new InvalidOperationException($"No mesh {meshGuid}");

                var materials = new List<Material?>();
                if (item.TryGetProperty("materials", out var mats) && mats.ValueKind == JsonValueKind.Array)
                    foreach (var g in mats.EnumerateArray())
                        materials.Add(g.ValueKind == JsonValueKind.String && !string.IsNullOrEmpty(g.GetString())
                            ? Engine.Assets.LoadByGuid<Material>(g.GetString()!) : null);

                var position = item.TryGetProperty("position", out var p) ? Vec3(p) : Vector3.Zero;
                var rotation = item.TryGetProperty("rotation", out var r) ? Quat(r) : Quaternion.Identity;
                var scale = Vector3.One;
                if (item.TryGetProperty("scale", out var s))
                    scale = s.ValueKind == JsonValueKind.Number ? new Vector3(s.GetSingle()) : Vec3(s);

                var part = new IconPart(mesh, materials,
                    Matrix4x4.CreateScale(scale) * Matrix4x4.CreateFromQuaternion(rotation) * Matrix4x4.CreateTranslation(position));
                if (item.TryGetProperty("side", out var side) && side.ValueKind == JsonValueKind.String)
                    part.Side = Enum.Parse<IconSide>(side.GetString()!, true);
                if (item.TryGetProperty("sideCut", out var cut)) part.SideCut = cut.GetSingle();
                part.AlphaTest = item.TryGetProperty("alphaTest", out var at)
                    ? at.GetBoolean()
                    : materials.Any(x => x?.Effect?.Name?.Contains("skinned", StringComparison.OrdinalIgnoreCase) == true);
                parts.Add(part);
            }
            return parts;
        }

        public static IconView View(JsonElement root, IReadOnlyList<IconPart> parts)
        {
            var view = new IconView();
            float yaw = root.TryGetProperty("yaw", out var y) ? y.GetSingle() : 0f;
            float pitch = root.TryGetProperty("pitch", out var pi) ? pi.GetSingle() : 0f;
            float roll = root.TryGetProperty("roll", out var ro) ? ro.GetSingle() : 0f;
            bool nearUp = root.TryGetProperty("up", out var up) && up.GetString() == "near";
            bool nearFront = root.TryGetProperty("front", out var front) && front.GetString() == "near";
            string? orientName = root.TryGetProperty("orient", out var orient) ? orient.GetString() : null;
            var frame = orientName switch
            {
                "long" => IconRenderer.LongAxisFrame(parts, nearUp, nearFront),
                "principal" => IconRenderer.PrincipalFrame(parts,
                    root.TryGetProperty("upHint", out var uh) ? Vec3(uh) : Vector3.UnitY,
                    root.TryGetProperty("frontHint", out var fh) ? Vec3(fh) : -Vector3.UnitZ),
                _ => Matrix4x4.Identity,
            };
            view.Rotation = frame * IconView.Turn(yaw, pitch, roll);
            if (root.TryGetProperty("interior", out var interior)) view.Interior = interior.GetSingle();
            if (root.TryGetProperty("margin", out var margin)) view.Margin = margin.GetSingle();
            if (root.TryGetProperty("exposure", out var exposure)) view.Exposure = exposure.GetSingle();
            if (root.TryGetProperty("supersample", out var ss)) view.Supersample = ss.GetInt32();
            return view;
        }

        static Vector3 Vec3(JsonElement e) => new(e[0].GetSingle(), e[1].GetSingle(), e[2].GetSingle());
        static Quaternion Quat(JsonElement e) => new(e[0].GetSingle(), e[1].GetSingle(), e[2].GetSingle(), e[3].GetSingle());
    }
}
