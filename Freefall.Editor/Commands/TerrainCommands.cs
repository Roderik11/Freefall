using System;
using System.Linq;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// Terrain height query: returns the terrain Y at a given (X, Z) world position.
    /// Supports single point and batch queries.
    /// </summary>
    [CommandRoute("GET", "/api/terrain/height")]
    public class TerrainHeightCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var terrain = FindTerrainRenderer();
            if (terrain == null)
                return CommandResult.NotFound("No TerrainRenderer found in scene");

            var query = CommandHelpers.ParseQueryString(context.Path);

            if (!query.TryGetValue("x", out var xs) || !query.TryGetValue("z", out var zs))
                return CommandResult.BadRequest("Required query params: x, z (e.g. /api/terrain/height?x=780&z=1020)");

            if (!float.TryParse(xs, System.Globalization.CultureInfo.InvariantCulture, out float x) ||
                !float.TryParse(zs, System.Globalization.CultureInfo.InvariantCulture, out float z))
                return CommandResult.BadRequest("x and z must be valid numbers");

            float y = terrain.GetHeight(new Vector3(x, 0, z));

            return CommandResult.Json(new
            {
                x,
                y,
                z,
                terrainOrigin = CommandHelpers.Vec3(terrain.Transform.Position)
            });
        }

        internal static TerrainRenderer FindTerrainRenderer()
        {
            foreach (var entity in EntityManager.Entities)
            {
                var tr = entity.GetComponent<TerrainRenderer>();
                if (tr != null) return tr;
            }
            return null;
        }
    }

    /// <summary>
    /// Batch height query for multiple points.
    /// POST /api/terrain/heights with body: {"points": [{"x":780,"z":1020}, ...]}
    /// </summary>
    [CommandRoute("POST", "/api/terrain/heights")]
    public class TerrainHeightBatchCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var terrain = TerrainHeightCommand.FindTerrainRenderer();
            if (terrain == null)
                return CommandResult.NotFound("No TerrainRenderer found in scene");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'points' array");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            if (!root.TryGetProperty("points", out var pointsArr))
                return CommandResult.BadRequest("Body must contain 'points' array of {x, z} objects");

            var results = new System.Collections.Generic.List<object>();
            foreach (var pt in pointsArr.EnumerateArray())
            {
                float x = pt.TryGetProperty("x", out var xp) ? xp.GetSingle() : 0;
                float z = pt.TryGetProperty("z", out var zp) ? zp.GetSingle() : 0;
                float y = terrain.GetHeight(new Vector3(x, 0, z));
                results.Add(new { x, y, z });
            }

            return CommandResult.Json(new { count = results.Count, heights = results });
        }
    }

    /// <summary>
    /// Returns terrain metadata: dimensions, resolution, height range, world bounds.
    /// GET /api/terrain/info
    /// </summary>
    [CommandRoute("GET", "/api/terrain/info")]
    public class TerrainInfoCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var terrainRenderer = TerrainHeightCommand.FindTerrainRenderer();
            if (terrainRenderer == null)
                return CommandResult.NotFound("No TerrainRenderer found in scene");

            var terrain = terrainRenderer.Terrain;
            if (terrain == null)
                return CommandResult.NotFound("TerrainRenderer has no Terrain asset assigned");

            var origin = terrainRenderer.Transform.Position;
            var size = terrain.TerrainSize;

            return CommandResult.Json(new
            {
                name = terrain.Name,
                guid = terrain.Guid,
                terrainSize = new { x = size.X, z = size.Y },
                maxHeight = terrain.MaxHeight,
                heightmapResolution = terrain.HeightmapResolution,
                origin = CommandHelpers.Vec3(origin),
                worldBounds = new
                {
                    min = CommandHelpers.Vec3(origin),
                    max = CommandHelpers.Vec3(origin + new Vector3(size.X, terrain.MaxHeight, size.Y))
                }
            });
        }
    }

    /// <summary>
    /// Samples a rectangular area of the terrain and returns height statistics + slope.
    /// POST /api/terrain/sample
    /// Body: {"center":{"x":500,"z":500}, "size":{"x":40,"z":30}, "rotation":0, "density":2}
    /// - rotation: degrees around Y axis (default 0)
    /// - density: samples per unit (default 1.0)
    /// </summary>
    [CommandRoute("POST", "/api/terrain/sample")]
    public class TerrainSampleCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var terrain = TerrainHeightCommand.FindTerrainRenderer();
            if (terrain == null)
                return CommandResult.NotFound("No TerrainRenderer found in scene");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'center' and 'size'");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            if (!root.TryGetProperty("center", out var centerProp) ||
                !root.TryGetProperty("size", out var sizeProp))
                return CommandResult.BadRequest("Body must contain 'center' {x,z} and 'size' {x,z}");

            float cx = centerProp.TryGetProperty("x", out var cxp) ? cxp.GetSingle() : 0;
            float cz = centerProp.TryGetProperty("z", out var czp) ? czp.GetSingle() : 0;
            float sx = sizeProp.TryGetProperty("x", out var sxp) ? sxp.GetSingle() : 10;
            float sz = sizeProp.TryGetProperty("z", out var szp) ? szp.GetSingle() : 10;

            float rotDeg = root.TryGetProperty("rotation", out var rotProp) ? rotProp.GetSingle() : 0;
            float density = root.TryGetProperty("density", out var denProp) ? denProp.GetSingle() : 1.0f;

            // Clamp density to reasonable range
            density = Math.Clamp(density, 0.1f, 10f);

            float rotRad = rotDeg * (MathF.PI / 180f);
            float cosR = MathF.Cos(rotRad);
            float sinR = MathF.Sin(rotRad);

            int samplesX = Math.Max(2, (int)(sx * density));
            int samplesZ = Math.Max(2, (int)(sz * density));
            int totalSamples = samplesX * samplesZ;

            float halfX = sx * 0.5f;
            float halfZ = sz * 0.5f;

            float minH = float.MaxValue, maxH = float.MinValue;
            double sumH = 0;
            var heights = new float[totalSamples];
            var positions = new Vector3[totalSamples];
            int idx = 0;

            for (int iz = 0; iz < samplesZ; iz++)
            {
                for (int ix = 0; ix < samplesX; ix++)
                {
                    // Local offset from center
                    float lx = -halfX + sx * ix / (samplesX - 1);
                    float lz = -halfZ + sz * iz / (samplesZ - 1);

                    // Rotate around Y
                    float wx = cx + lx * cosR - lz * sinR;
                    float wz = cz + lx * sinR + lz * cosR;

                    float h = terrain.GetHeight(new Vector3(wx, 0, wz));
                    heights[idx] = h;
                    positions[idx] = new Vector3(wx, h, wz);

                    if (h < minH) minH = h;
                    if (h > maxH) maxH = h;
                    sumH += h;
                    idx++;
                }
            }

            float avgH = (float)(sumH / totalSamples);

            // Compute best-fit plane via least-squares (normal of the plane = slope direction)
            // Using the covariance method for a plane fit
            double sumLx = 0, sumLz = 0, sumLxLx = 0, sumLzLz = 0, sumLxLz = 0;
            double sumLxH = 0, sumLzH = 0;

            idx = 0;
            for (int iz = 0; iz < samplesZ; iz++)
            {
                for (int ix = 0; ix < samplesX; ix++)
                {
                    float lx = -halfX + sx * ix / (samplesX - 1);
                    float lz = -halfZ + sz * iz / (samplesZ - 1);
                    float h = heights[idx++];

                    sumLx += lx; sumLz += lz;
                    sumLxLx += lx * lx; sumLzLz += lz * lz; sumLxLz += lx * lz;
                    sumLxH += lx * h; sumLzH += lz * h;
                }
            }

            int n = totalSamples;
            // Solve: h = a*lx + b*lz + c  (least squares)
            double det = (sumLxLx * sumLzLz - sumLxLz * sumLxLz) * n
                       + (sumLxLz * sumLz - sumLzLz * sumLx) * sumLx
                       + (sumLxLz * sumLx - sumLxLx * sumLz) * sumLz;

            float slopeX = 0, slopeZ = 0;
            if (Math.Abs(det) > 1e-10)
            {
                // Simplified: for centered samples, sumLx ≈ 0, sumLz ≈ 0
                // But we do the full solve anyway
                double a = ((sumLzLz * n - sumLz * sumLz) * sumLxH
                          + (sumLxLz * sumLz - sumLzLz * sumLx) * sumH
                          + (sumLx * sumLz - sumLxLz * n) * sumLzH) / det;
                double b = ((sumLxLz * sumLx - sumLxLx * sumLz) * sumH
                          + (sumLxLx * n - sumLx * sumLx) * sumLzH
                          + (sumLx * sumLz - sumLxLz * n) * sumLxH) / det;

                slopeX = (float)a;
                slopeZ = (float)b;
            }

            float slopeAngle = MathF.Atan(MathF.Sqrt(slopeX * slopeX + slopeZ * slopeZ)) * (180f / MathF.PI);

            // Flatness: 1.0 = perfectly flat, 0.0 = very rough
            // Based on standard deviation of heights relative to the best-fit plane
            double sumSqDev = 0;
            idx = 0;
            for (int iz = 0; iz < samplesZ; iz++)
            {
                for (int ix = 0; ix < samplesX; ix++)
                {
                    float lx = -halfX + sx * ix / (samplesX - 1);
                    float lz = -halfZ + sz * iz / (samplesZ - 1);
                    float predicted = avgH + slopeX * lx + slopeZ * lz;
                    float diff = heights[idx++] - predicted;
                    sumSqDev += diff * diff;
                }
            }
            float stdDev = MathF.Sqrt((float)(sumSqDev / n));
            float flatness = 1.0f / (1.0f + stdDev); // 1 when stdDev=0, approaches 0 for rough terrain

            // Compute plane normal (from slope gradients)
            var planeNormal = Vector3.Normalize(new Vector3(-slopeX, 1, -slopeZ));

            return CommandResult.Json(new
            {
                samples = totalSamples,
                area = new { center = new { x = cx, z = cz }, size = new { x = sx, z = sz }, rotation = rotDeg },
                height = new { min = minH, max = maxH, average = avgH, range = maxH - minH },
                slope = new
                {
                    gradientX = slopeX,
                    gradientZ = slopeZ,
                    angle = slopeAngle,
                    normal = CommandHelpers.Vec3(planeNormal)
                },
                flatness,
                stdDev
            });
        }
    }

    /// <summary>
    /// Aligns an entity to the terrain surface using its mesh bounds.
    /// Samples terrain under the entity's footprint, computes best-fit plane,
    /// and adjusts Y position + rotation to match slope.
    /// POST /api/entity/{id}/alignToTerrain
    /// </summary>
    [CommandRoute("POST", "/api/entity/{id}/alignToTerrain")]
    public class AlignToTerrainCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var terrain = TerrainHeightCommand.FindTerrainRenderer();
            if (terrain == null)
                return CommandResult.NotFound("No TerrainRenderer found in scene");

            int entityId = context.GetInt("id");
            var entity = CommandHelpers.FindEntityById(entityId);
            if (entity == null)
                return CommandResult.NotFound($"Entity {entityId} not found");

            // Get mesh bounds for footprint sampling
            var meshRenderer = entity.GetComponent<MeshRenderer>();
            Vortice.Mathematics.BoundingBox? localBounds = null;

            if (meshRenderer?.Mesh != null)
                localBounds = meshRenderer.Mesh.BoundingBox;

            var pos = entity.Transform.Position;
            var scale = entity.Transform.Scale;
            var rot = entity.Transform.Rotation;

            float halfX, halfZ;
            if (localBounds.HasValue)
            {
                // Use actual mesh footprint (scaled)
                halfX = Math.Max(Math.Abs(localBounds.Value.Min.X), Math.Abs(localBounds.Value.Max.X)) * scale.X;
                halfZ = Math.Max(Math.Abs(localBounds.Value.Min.Z), Math.Abs(localBounds.Value.Max.Z)) * scale.Z;
                // Minimum 1m footprint
                halfX = Math.Max(halfX, 0.5f);
                halfZ = Math.Max(halfZ, 0.5f);
            }
            else
            {
                // No mesh — use a small default footprint
                halfX = 2f;
                halfZ = 2f;
            }

            // Get entity's Y rotation for oriented sampling
            float yaw = MathF.Atan2(
                2 * (rot.W * rot.Y + rot.X * rot.Z),
                1 - 2 * (rot.Y * rot.Y + rot.Z * rot.Z));
            float cosR = MathF.Cos(yaw);
            float sinR = MathF.Sin(yaw);

            // Sample a 3×3 grid under the footprint (9 points — corners + edges + center)
            const int grid = 3;
            var heights = new float[grid * grid];
            var localX = new float[grid * grid];
            var localZ = new float[grid * grid];
            int idx = 0;

            for (int iz = 0; iz < grid; iz++)
            {
                for (int ix = 0; ix < grid; ix++)
                {
                    float lx = -halfX + 2 * halfX * ix / (grid - 1);
                    float lz = -halfZ + 2 * halfZ * iz / (grid - 1);

                    float wx = pos.X + lx * cosR - lz * sinR;
                    float wz = pos.Z + lx * sinR + lz * cosR;

                    float h = terrain.GetHeight(new Vector3(wx, 0, wz));
                    heights[idx] = h;
                    localX[idx] = lx;
                    localZ[idx] = lz;
                    idx++;
                }
            }

            // Least-squares plane fit: h = a*lx + b*lz + c
            double sumLx = 0, sumLz2 = 0, sumLx2 = 0, sumLxLz = 0, sumLz = 0;
            double sumLxH = 0, sumLzH = 0, sumH = 0;

            for (int i = 0; i < idx; i++)
            {
                sumLx += localX[i]; sumLz += localZ[i];
                sumLx2 += localX[i] * localX[i];
                sumLz2 += localZ[i] * localZ[i];
                sumLxLz += localX[i] * localZ[i];
                sumLxH += localX[i] * heights[i];
                sumLzH += localZ[i] * heights[i];
                sumH += heights[i];
            }

            int n = idx;
            double det = (sumLx2 * sumLz2 - sumLxLz * sumLxLz) * n
                       + (sumLxLz * sumLz - sumLz2 * sumLx) * sumLx
                       + (sumLxLz * sumLx - sumLx2 * sumLz) * sumLz;

            float slopeX = 0, slopeZ = 0;
            float avgH = (float)(sumH / n);

            if (Math.Abs(det) > 1e-10)
            {
                slopeX = (float)(((sumLz2 * n - sumLz * sumLz) * sumLxH
                        + (sumLxLz * sumLz - sumLz2 * sumLx) * sumH
                        + (sumLx * sumLz - sumLxLz * n) * sumLzH) / det);
                slopeZ = (float)(((sumLxLz * sumLx - sumLx2 * sumLz) * sumH
                        + (sumLx2 * n - sumLx * sumLx) * sumLzH
                        + (sumLx * sumLz - sumLxLz * n) * sumLxH) / det);
            }

            // Compute rotation from slope normal
            var terrainNormal = Vector3.Normalize(new Vector3(-slopeX, 1, -slopeZ));
            var slopeRotation = RotationFromNormal(terrainNormal, yaw);

            // Set the entity's position (center height) and rotation
            entity.Transform.Position = new Vector3(pos.X, avgH, pos.Z);
            entity.Transform.Rotation = slopeRotation;

            float slopeAngle = MathF.Atan(MathF.Sqrt(slopeX * slopeX + slopeZ * slopeZ)) * (180f / MathF.PI);

            return CommandResult.Json(new
            {
                status = "aligned",
                id = entity.Id, uid = entity.UID.ToString(),
                name = entity.Name,
                position = CommandHelpers.Vec3(entity.Transform.Position),
                rotation = CommandHelpers.Quat(entity.Transform.Rotation),
                slope = new
                {
                    angle = slopeAngle,
                    normal = CommandHelpers.Vec3(terrainNormal)
                },
                footprint = new { halfX, halfZ },
                sampledHeights = new { min = heights.Min(), max = heights.Max(), average = avgH }
            });
        }

        /// <summary>
        /// Builds a quaternion that tilts an object so its local Y aligns with the terrain normal,
        /// while preserving the original yaw (heading) rotation.
        /// </summary>
        private static Quaternion RotationFromNormal(Vector3 normal, float yaw)
        {
            // Start with the yaw rotation around world Y
            var yawQuat = Quaternion.CreateFromAxisAngle(Vector3.UnitY, yaw);

            // Compute the tilt rotation from world up to terrain normal
            var up = Vector3.UnitY;
            var cross = Vector3.Cross(up, normal);
            float dot = Vector3.Dot(up, normal);

            Quaternion tilt;
            if (cross.LengthSquared() < 1e-8f)
            {
                // Normal is already up (or down) — no tilt needed
                tilt = Quaternion.Identity;
            }
            else
            {
                cross = Vector3.Normalize(cross);
                float angle = MathF.Acos(Math.Clamp(dot, -1f, 1f));
                tilt = Quaternion.CreateFromAxisAngle(cross, angle);
            }

            // Tilt first, then apply yaw
            return Quaternion.Normalize(tilt * yawQuat);
        }
    }

    /// <summary>
    /// Terrain brush stroke: paints ControlMaps along a polyline path.
    /// POST /api/terrain/brush
    /// Body: {"points":[{"x":500,"z":500}], "radius":30, "strength":0.5,
    ///        "falloff":1.0, "mode":"raise", "targetHeight":10,
    ///        "target":"height", "layerIndex":0}
    /// target: "height" (default), "splatmap", "density"
    /// layerIndex: which TextureLayer or Decoration index (default 0)
    /// </summary>
    [CommandRoute("POST", "/api/terrain/brush")]
    public class TerrainBrushCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var terrainRenderer = TerrainHeightCommand.FindTerrainRenderer();
            if (terrainRenderer == null)
                return CommandResult.NotFound("No TerrainRenderer found in scene");

            var terrain = terrainRenderer.Terrain;
            if (terrain == null)
                return CommandResult.NotFound("TerrainRenderer has no Terrain asset");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'points' array");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            if (!root.TryGetProperty("points", out var pointsArr))
                return CommandResult.BadRequest("Body must contain 'points' array of {x, z} objects");

            // Parse stroke points (world space → terrain UV)
            var origin = terrainRenderer.Transform.Position;
            var size = terrain.TerrainSize;
            var worldPoints = new System.Collections.Generic.List<Vector2>();

            foreach (var pt in pointsArr.EnumerateArray())
            {
                float wx = pt.TryGetProperty("x", out var xp) ? xp.GetSingle() : 0;
                float wz = pt.TryGetProperty("z", out var zp) ? zp.GetSingle() : 0;

                // World → terrain UV [0..1]
                float u = (wx - origin.X) / size.X;
                float v = (wz - origin.Z) / size.Y;
                worldPoints.Add(new Vector2(u, v));
            }

            if (worldPoints.Count == 0)
                return CommandResult.BadRequest("'points' array must contain at least one point");

            // Parse brush params
            float radius = root.TryGetProperty("radius", out var rp) ? rp.GetSingle() : 20f;
            float strength = root.TryGetProperty("strength", out var sp) ? sp.GetSingle() : 0.5f;
            float falloff = root.TryGetProperty("falloff", out var fp) ? fp.GetSingle() : 1.0f;
            float targetHeight = root.TryGetProperty("targetHeight", out var tp) ? tp.GetSingle() : 0;

            string modeStr = root.TryGetProperty("mode", out var mp) ? mp.GetString() : "raise";
            var mode = modeStr?.ToLowerInvariant() switch
            {
                "raise" => BrushMode.Raise,
                "lower" => BrushMode.Lower,
                "flatten" => BrushMode.Flatten,
                "smooth" => BrushMode.Smooth,
                _ => BrushMode.Raise
            };

            // Target: "height" (default), "splatmap", "density"
            string targetStr = root.TryGetProperty("target", out var tgt) ? tgt.GetString() : "height";
            int layerIndex = root.TryGetProperty("layerIndex", out var li) ? li.GetInt32() : 0;

            var target = targetStr?.ToLowerInvariant() switch
            {
                "splatmap" => TerrainBaker.ControlMapTarget.Splatmap,
                "density" => TerrainBaker.ControlMapTarget.Density,
                _ => TerrainBaker.ControlMapTarget.Height
            };

            // V-flip for splatmap/density targets is now handled by CS_PaintBrush (FlipV push constant)

            // Validate layer index
            if (target == TerrainBaker.ControlMapTarget.Splatmap &&
                (terrain.Layers == null || layerIndex < 0 || layerIndex >= terrain.Layers.Count))
                return CommandResult.BadRequest($"layerIndex {layerIndex} out of range (terrain has {terrain.Layers?.Count ?? 0} layers)");
            if (target == TerrainBaker.ControlMapTarget.Density &&
                (terrain.Decorations == null || layerIndex < 0 || layerIndex >= terrain.Decorations.Count))
                return CommandResult.BadRequest($"layerIndex {layerIndex} out of range (terrain has {terrain.Decorations?.Count ?? 0} decorations)");

            // Enqueue GPU brush stroke
            var uvPoints = worldPoints.ToArray();

            Debug.Log($"[TerrainBrushCmd] target={targetStr} layer={layerIndex} mode={modeStr} " +
                      $"pts={worldPoints.Count} radius={radius} strength={strength} " +
                      $"uv0=({uvPoints[0].X:F3},{uvPoints[0].Y:F3})");

            terrainRenderer.EnqueueBrushStroke(
                uvPoints, uvPoints.Length,
                (uint)mode, strength, radius, falloff, targetHeight,
                target, layerIndex);

            return CommandResult.Json(new
            {
                status = "brushed",
                target = targetStr,
                layerIndex,
                mode = modeStr,
                points = worldPoints.Count,
                radius,
                strength,
                falloff,
                targetHeight
            });
        }
    }

    /// <summary>
    /// Import a texture channel into a ControlMap.
    /// POST /api/terrain/import-channel
    /// Body: {"sourceGuid":"...", "channel":"r", "target":"splatmap", "layerIndex":0}
    /// channel: "r","g","b","a" (default "r")
    /// target: "height","splatmap","density" (default "splatmap")
    /// </summary>
    [CommandRoute("POST", "/api/terrain/import-channel")]
    public class TerrainImportChannelCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var terrainRenderer = TerrainHeightCommand.FindTerrainRenderer();
            if (terrainRenderer == null)
                return CommandResult.NotFound("No TerrainRenderer found in scene");

            var terrain = terrainRenderer.Terrain;
            if (terrain == null)
                return CommandResult.NotFound("TerrainRenderer has no Terrain asset");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            // Source texture — either from asset GUID or external file path
            Texture sourceTexture = null;
            string sourceGuid = root.TryGetProperty("sourceGuid", out var sg) ? sg.GetString() : null;
            string sourcePath = root.TryGetProperty("path", out var sp) ? sp.GetString() : null;

            if (!string.IsNullOrEmpty(sourceGuid))
            {
                sourceTexture = Engine.Assets.LoadByGuid<Texture>(sourceGuid);
                if (sourceTexture == null)
                    return CommandResult.NotFound($"Texture not found: {sourceGuid}");
            }
            else if (!string.IsNullOrEmpty(sourcePath))
            {
                // Load from external file
                if (!System.IO.File.Exists(sourcePath))
                    return CommandResult.NotFound($"File not found: {sourcePath}");
                try
                {
                    var cpuData = Texture.ParseFromFile(Engine.Device, sourcePath);
                    sourceTexture = Texture.CreateAsync(Engine.Device, cpuData);
                    // Flush the streaming upload so the GPU texture is ready
                    Graphics.StreamingManager.Instance?.Flush();
                    Debug.Log($"[TerrainImport] Loaded external file: {sourcePath} ({cpuData.Width}x{cpuData.Height}, {cpuData.Format})");
                }
                catch (Exception ex)
                {
                    return CommandResult.Error(500, $"Failed to load file: {ex.Message}");
                }
            }
            else
            {
                return CommandResult.BadRequest("'sourceGuid' or 'path' is required");
            }

            // Channel
            string channelStr = root.TryGetProperty("channel", out var ch) ? ch.GetString() : "r";
            int channelIndex = channelStr?.ToLowerInvariant() switch
            {
                "r" => 0, "g" => 1, "b" => 2, "a" => 3,
                _ => 0
            };

            // Target
            string targetStr = root.TryGetProperty("target", out var tgt) ? tgt.GetString() : "splatmap";
            int layerIndex = root.TryGetProperty("layerIndex", out var li) ? li.GetInt32() : 0;

            var target = targetStr?.ToLowerInvariant() switch
            {
                "height" => TerrainBaker.ControlMapTarget.Height,
                "density" => TerrainBaker.ControlMapTarget.Density,
                _ => TerrainBaker.ControlMapTarget.Splatmap
            };

            // Validate
            if (target == TerrainBaker.ControlMapTarget.Splatmap &&
                (terrain.Layers == null || layerIndex < 0 || layerIndex >= terrain.Layers.Count))
                return CommandResult.BadRequest($"layerIndex {layerIndex} out of range ({terrain.Layers?.Count ?? 0} layers)");
            if (target == TerrainBaker.ControlMapTarget.Density &&
                (terrain.Decorations == null || layerIndex < 0 || layerIndex >= terrain.Decorations.Count))
                return CommandResult.BadRequest($"layerIndex {layerIndex} out of range ({terrain.Decorations?.Count ?? 0} decorations)");

            terrainRenderer.EnqueueImportChannel(sourceTexture, channelIndex, target, layerIndex);

            return CommandResult.Json(new
            {
                status = "imported",
                source = !string.IsNullOrEmpty(sourceGuid) ? sourceGuid : sourcePath,
                channel = channelStr,
                target = targetStr,
                layerIndex
            });
        }
    }

    // ════════════════════════════════════════════════════════════════════
    //  TERRAIN LAYER MANAGEMENT
    // ════════════════════════════════════════════════════════════════════

    /// <summary>
    /// List all terrain layers.
    /// GET /api/terrain/layers
    /// </summary>
    [CommandRoute("GET", "/api/terrain/layers")]
    public class TerrainListLayersCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var tr = TerrainHeightCommand.FindTerrainRenderer();
            if (tr?.Terrain == null) return CommandResult.NotFound("No terrain");
            var t = tr.Terrain;

            var heightLayers = t.HeightLayers.Select((l, i) =>
            {
                object extra = l switch
                {
                    NoiseHeightLayer n => new { noiseType = n.Type.ToString(), n.Octaves, n.Frequency, n.Amplitude, n.Lacunarity, n.Persistence, offset = new { x = n.Offset.X, y = n.Offset.Y }, n.Seed },
                    ErosionHeightLayer e => new { e.Scale, e.Strength, e.GullyWeight, e.Detail, e.Octaves, e.Lacunarity, e.Gain, e.RidgeRounding, e.CreaseRounding, e.CellScale, e.Normalization, e.SlopeOnset, e.AssumedSlope, e.AssumedSlopeAmount },
                    ImportHeightLayer imp => new { source = imp.Source?.Name },
                    PaintHeightLayer p => new { hasControlMap = p.ControlMap != null },
                    _ => (object)null
                };
                return new
                {
                    index = i, type = l.GetType().Name, enabled = l.Enabled,
                    blendMode = l.BlendMode.ToString(), opacity = l.Opacity,
                    details = extra
                };
            });

            var textureLayers = (t.Layers ?? new()).Select((l, i) => new
            {
                index = i, diffuse = l.Diffuse?.Name, diffuseGuid = l.Diffuse?.Guid,
                normals = l.Normals?.Name, normalsGuid = l.Normals?.Guid,
                tiling = new { x = l.Tiling.X, y = l.Tiling.Y },
                hasControlMap = l.ControlMap != null
            });

            var decoLayers = (t.Decorations ?? new()).Select((d, i) => new
            {
                index = i, mode = d.Mode.ToString(), mesh = d.Mesh?.Name,
                density = d.Density, weight = d.Weight, hasControlMap = d.ControlMap != null
            });

            return CommandResult.Json(new { heightLayers, textureLayers, decorationLayers = decoLayers });
        }
    }

    /// <summary>
    /// Add a layer. POST /api/terrain/layers/add
    /// Body: {"category":"texture", "diffuse":"guid", "normals":"guid", "tiling":{"x":10,"y":10}}
    ///        {"category":"height", "type":"paint"}
    ///        {"category":"decoration", "mesh":"guid", "density":2.0}
    /// </summary>
    [CommandRoute("POST", "/api/terrain/layers/add")]
    public class TerrainAddLayerCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var tr = TerrainHeightCommand.FindTerrainRenderer();
            if (tr?.Terrain == null) return CommandResult.NotFound("No terrain");
            if (string.IsNullOrEmpty(context.Body)) return CommandResult.BadRequest("Body required");

            using var doc = context.ParseBody();
            var root = doc.RootElement;
            string category = root.TryGetProperty("category", out var c) ? c.GetString() : "texture";
            var t = tr.Terrain;

            switch (category?.ToLowerInvariant())
            {
                case "height":
                {
                    string htype = root.TryGetProperty("type", out var ht) ? ht.GetString() : "paint";
                    switch (htype?.ToLowerInvariant())
                    {
                        case "import":
                        {
                            string srcGuid = root.TryGetProperty("source", out var sg) ? sg.GetString() : null;
                            var src = !string.IsNullOrEmpty(srcGuid) ? Engine.Assets.LoadByGuid<Texture>(srcGuid) : null;
                            t.HeightLayers.Add(new ImportHeightLayer { Source = src });
                            break;
                        }
                        case "noise":
                        {
                            var nl = new NoiseHeightLayer();
                            if (root.TryGetProperty("noiseType", out var nt) && Enum.TryParse<NoiseType>(nt.GetString(), true, out var ntype)) nl.Type = ntype;
                            if (root.TryGetProperty("octaves", out var oc)) nl.Octaves = oc.GetInt32();
                            if (root.TryGetProperty("frequency", out var fr)) nl.Frequency = fr.GetSingle();
                            if (root.TryGetProperty("amplitude", out var am)) nl.Amplitude = am.GetSingle();
                            if (root.TryGetProperty("lacunarity", out var la)) nl.Lacunarity = la.GetSingle();
                            if (root.TryGetProperty("persistence", out var pe)) nl.Persistence = pe.GetSingle();
                            if (root.TryGetProperty("seed", out var se)) nl.Seed = se.GetInt32();
                            if (root.TryGetProperty("blendMode", out var bm) && Enum.TryParse<HeightBlendMode>(bm.GetString(), true, out var bmode)) nl.BlendMode = bmode;
                            if (root.TryGetProperty("opacity", out var op2)) nl.Opacity = op2.GetSingle();
                            if (root.TryGetProperty("offset", out var off))
                                nl.Offset = new Vector2(
                                    off.TryGetProperty("x", out var ox) ? ox.GetSingle() : 0,
                                    off.TryGetProperty("y", out var oy) ? oy.GetSingle() : 0);
                            t.HeightLayers.Add(nl);
                            break;
                        }
                        case "erosion":
                        {
                            var el = new ErosionHeightLayer();
                            if (root.TryGetProperty("scale", out var esc)) el.Scale = esc.GetSingle();
                            if (root.TryGetProperty("strength", out var est)) el.Strength = est.GetSingle();
                            if (root.TryGetProperty("gullyWeight", out var egw)) el.GullyWeight = egw.GetSingle();
                            if (root.TryGetProperty("detail", out var edt)) el.Detail = edt.GetSingle();
                            if (root.TryGetProperty("octaves", out var eoc)) el.Octaves = eoc.GetInt32();
                            if (root.TryGetProperty("lacunarity", out var ela)) el.Lacunarity = ela.GetSingle();
                            if (root.TryGetProperty("gain", out var egn)) el.Gain = egn.GetSingle();
                            if (root.TryGetProperty("ridgeRounding", out var err)) el.RidgeRounding = err.GetSingle();
                            if (root.TryGetProperty("creaseRounding", out var ecr)) el.CreaseRounding = ecr.GetSingle();
                            if (root.TryGetProperty("cellScale", out var ecs)) el.CellScale = ecs.GetSingle();
                            if (root.TryGetProperty("normalization", out var enm)) el.Normalization = enm.GetSingle();
                            if (root.TryGetProperty("slopeOnset", out var eso)) el.SlopeOnset = eso.GetSingle();
                            if (root.TryGetProperty("assumedSlope", out var eas)) el.AssumedSlope = eas.GetSingle();
                            if (root.TryGetProperty("assumedSlopeAmount", out var easa)) el.AssumedSlopeAmount = easa.GetSingle();
                            if (root.TryGetProperty("blendMode", out var bm2) && Enum.TryParse<HeightBlendMode>(bm2.GetString(), true, out var bmode2)) el.BlendMode = bmode2;
                            if (root.TryGetProperty("opacity", out var op3)) el.Opacity = op3.GetSingle();
                            t.HeightLayers.Add(el);
                            break;
                        }
                        default:
                            t.HeightLayers.Add(new PaintHeightLayer());
                            break;
                    }
                    t.MarkDirty();
                    return CommandResult.Json(new { status = "added", category, index = t.HeightLayers.Count - 1 });
                }
                case "texture":
                {
                    t.Layers ??= new();
                    string dGuid = root.TryGetProperty("diffuse", out var dg) ? dg.GetString() : null;
                    string nGuid = root.TryGetProperty("normals", out var ng) ? ng.GetString() : null;
                    float tx = 10f, ty = 10f;
                    if (root.TryGetProperty("tiling", out var tp))
                    {
                        tx = tp.TryGetProperty("x", out var txp) ? txp.GetSingle() : 10f;
                        ty = tp.TryGetProperty("y", out var typ) ? typ.GetSingle() : 10f;
                    }
                    var layer = new Terrain.TextureLayer
                    {
                        Diffuse = !string.IsNullOrEmpty(dGuid) ? Engine.Assets.LoadByGuid<Texture>(dGuid) : null,
                        Normals = !string.IsNullOrEmpty(nGuid) ? Engine.Assets.LoadByGuid<Texture>(nGuid) : null,
                        Tiling = new Vector2(tx, ty)
                    };
                    t.Layers.Add(layer);
                    t.MarkDirty();
                    return CommandResult.Json(new { status = "added", category, index = t.Layers.Count - 1, diffuse = layer.Diffuse?.Name, normals = layer.Normals?.Name });
                }
                case "decoration":
                {
                    t.Decorations ??= new();
                    string meshGuid = root.TryGetProperty("mesh", out var mg) ? mg.GetString() : null;
                    float density = root.TryGetProperty("density", out var dp) ? dp.GetSingle() : 1f;
                    var deco = new Terrain.Decoration
                    {
                        Mesh = !string.IsNullOrEmpty(meshGuid) ? Engine.Assets.LoadByGuid<Mesh>(meshGuid) : null,
                        Density = density
                    };
                    t.Decorations.Add(deco);
                    t.MarkDirty();
                    return CommandResult.Json(new { status = "added", category, index = t.Decorations.Count - 1, mesh = deco.Mesh?.Name });
                }
                default:
                    return CommandResult.BadRequest($"Unknown category '{category}'. Use: height, texture, decoration");
            }
        }
    }

    /// <summary>
    /// Remove a layer. POST /api/terrain/layers/remove
    /// Body: {"category":"texture", "index":2}
    /// </summary>
    [CommandRoute("POST", "/api/terrain/layers/remove")]
    public class TerrainRemoveLayerCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var tr = TerrainHeightCommand.FindTerrainRenderer();
            if (tr?.Terrain == null) return CommandResult.NotFound("No terrain");
            using var doc = context.ParseBody();
            var root = doc.RootElement;
            string category = root.TryGetProperty("category", out var c) ? c.GetString() : "";
            int index = root.TryGetProperty("index", out var ip) ? ip.GetInt32() : -1;
            var t = tr.Terrain;

            switch (category?.ToLowerInvariant())
            {
                case "height":
                    if (index < 0 || index >= t.HeightLayers.Count) return CommandResult.BadRequest("Index out of range");
                    t.HeightLayers.RemoveAt(index); break;
                case "texture":
                    if (t.Layers == null || index < 0 || index >= t.Layers.Count) return CommandResult.BadRequest("Index out of range");
                    t.Layers.RemoveAt(index); break;
                case "decoration":
                    if (t.Decorations == null || index < 0 || index >= t.Decorations.Count) return CommandResult.BadRequest("Index out of range");
                    t.Decorations.RemoveAt(index); break;
                default: return CommandResult.BadRequest("'category' required: height, texture, decoration");
            }
            t.MarkDirty();
            return CommandResult.Json(new { status = "removed", category, index });
        }
    }

    /// <summary>
    /// Set properties on a layer. POST /api/terrain/layers/set
    /// Body: {"category":"texture", "index":0, "diffuse":"guid", "tiling":{"x":5,"y":5}}
    ///        {"category":"height", "index":0, "enabled":false, "opacity":0.5}
    ///        {"category":"decoration", "index":0, "density":3.0}
    /// </summary>
    [CommandRoute("POST", "/api/terrain/layers/set")]
    public class TerrainSetLayerCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var tr = TerrainHeightCommand.FindTerrainRenderer();
            if (tr?.Terrain == null) return CommandResult.NotFound("No terrain");
            using var doc = context.ParseBody();
            var root = doc.RootElement;
            string category = root.TryGetProperty("category", out var c) ? c.GetString() : "";
            int index = root.TryGetProperty("index", out var ip) ? ip.GetInt32() : 0;
            var t = tr.Terrain;

            switch (category?.ToLowerInvariant())
            {
                case "height":
                {
                    if (index < 0 || index >= t.HeightLayers.Count) return CommandResult.BadRequest("Index out of range");
                    var hl = t.HeightLayers[index];
                    if (root.TryGetProperty("enabled", out var ep)) hl.Enabled = ep.GetBoolean();
                    if (root.TryGetProperty("opacity", out var op)) hl.Opacity = op.GetSingle();
                    if (root.TryGetProperty("blendMode", out var bm) && Enum.TryParse<HeightBlendMode>(bm.GetString(), true, out var mode))
                        hl.BlendMode = mode;

                    // Noise-specific properties
                    if (hl is NoiseHeightLayer nl)
                    {
                        if (root.TryGetProperty("noiseType", out var nt) && Enum.TryParse<NoiseType>(nt.GetString(), true, out var ntype)) nl.Type = ntype;
                        if (root.TryGetProperty("octaves", out var oc)) nl.Octaves = oc.GetInt32();
                        if (root.TryGetProperty("frequency", out var fr)) nl.Frequency = fr.GetSingle();
                        if (root.TryGetProperty("amplitude", out var am)) nl.Amplitude = am.GetSingle();
                        if (root.TryGetProperty("lacunarity", out var la)) nl.Lacunarity = la.GetSingle();
                        if (root.TryGetProperty("persistence", out var pe)) nl.Persistence = pe.GetSingle();
                        if (root.TryGetProperty("seed", out var se)) nl.Seed = se.GetInt32();
                        if (root.TryGetProperty("offset", out var off))
                            nl.Offset = new Vector2(
                                off.TryGetProperty("x", out var ox) ? ox.GetSingle() : nl.Offset.X,
                                off.TryGetProperty("y", out var oy) ? oy.GetSingle() : nl.Offset.Y);

                        // Terrace params
                        if (root.TryGetProperty("terraceSteps", out var ts)) nl.TerraceSteps = ts.GetInt32();
                        if (root.TryGetProperty("terraceSmoothness", out var tsm)) nl.TerraceSmoothness = tsm.GetSingle();

                        // Spatial mask (radial falloff)
                        if (root.TryGetProperty("maskCenter", out var mc))
                            nl.MaskCenter = new Vector2(
                                mc.TryGetProperty("x", out var mcx) ? mcx.GetSingle() : nl.MaskCenter.X,
                                mc.TryGetProperty("y", out var mcy) ? mcy.GetSingle() : nl.MaskCenter.Y);
                        if (root.TryGetProperty("maskRadius", out var mr)) nl.MaskRadius = mr.GetSingle();
                        if (root.TryGetProperty("maskFalloff", out var mf)) nl.MaskFalloff = mf.GetSingle();
                    }

                    // Erosion-specific properties
                    if (hl is ErosionHeightLayer el)
                    {
                        if (root.TryGetProperty("scale", out var esc)) el.Scale = esc.GetSingle();
                        if (root.TryGetProperty("strength", out var est)) el.Strength = est.GetSingle();
                        if (root.TryGetProperty("gullyWeight", out var egw)) el.GullyWeight = egw.GetSingle();
                        if (root.TryGetProperty("detail", out var edt)) el.Detail = edt.GetSingle();
                        if (root.TryGetProperty("octaves", out var eoc)) el.Octaves = eoc.GetInt32();
                        if (root.TryGetProperty("lacunarity", out var ela)) el.Lacunarity = ela.GetSingle();
                        if (root.TryGetProperty("gain", out var egn)) el.Gain = egn.GetSingle();
                        if (root.TryGetProperty("ridgeRounding", out var err)) el.RidgeRounding = err.GetSingle();
                        if (root.TryGetProperty("creaseRounding", out var ecr)) el.CreaseRounding = ecr.GetSingle();
                        if (root.TryGetProperty("cellScale", out var ecs)) el.CellScale = ecs.GetSingle();
                        if (root.TryGetProperty("normalization", out var enm)) el.Normalization = enm.GetSingle();
                        if (root.TryGetProperty("slopeOnset", out var eso)) el.SlopeOnset = eso.GetSingle();
                        if (root.TryGetProperty("assumedSlope", out var eas)) el.AssumedSlope = eas.GetSingle();
                        if (root.TryGetProperty("assumedSlopeAmount", out var easa)) el.AssumedSlopeAmount = easa.GetSingle();
                    }
                    break;
                }
                case "texture":
                {
                    if (t.Layers == null || index < 0 || index >= t.Layers.Count) return CommandResult.BadRequest("Index out of range");
                    var tl = t.Layers[index];
                    if (root.TryGetProperty("diffuse", out var dg)) tl.Diffuse = Engine.Assets.LoadByGuid<Texture>(dg.GetString());
                    if (root.TryGetProperty("normals", out var ng)) tl.Normals = Engine.Assets.LoadByGuid<Texture>(ng.GetString());
                    if (root.TryGetProperty("tiling", out var tp))
                        tl.Tiling = new Vector2(
                            tp.TryGetProperty("x", out var txp) ? txp.GetSingle() : tl.Tiling.X,
                            tp.TryGetProperty("y", out var typ) ? typ.GetSingle() : tl.Tiling.Y);
                    // Procedural auto-mask properties
                    if (root.TryGetProperty("heightRange", out var hr))
                        tl.HeightRange = new Vector2(
                            hr.TryGetProperty("min", out var hmin) ? hmin.GetSingle() :
                            hr.TryGetProperty("x", out var hx) ? hx.GetSingle() : tl.HeightRange.X,
                            hr.TryGetProperty("max", out var hmax) ? hmax.GetSingle() :
                            hr.TryGetProperty("y", out var hy) ? hy.GetSingle() : tl.HeightRange.Y);
                    if (root.TryGetProperty("slopeRange", out var sr))
                        tl.SlopeRange = new Vector2(
                            sr.TryGetProperty("min", out var smin) ? smin.GetSingle() :
                            sr.TryGetProperty("x", out var sx) ? sx.GetSingle() : tl.SlopeRange.X,
                            sr.TryGetProperty("max", out var smax) ? smax.GetSingle() :
                            sr.TryGetProperty("y", out var sy) ? sy.GetSingle() : tl.SlopeRange.Y);
                    if (root.TryGetProperty("heightBlend", out var hb)) tl.HeightBlend = hb.GetSingle();
                    if (root.TryGetProperty("slopeBlend", out var sb)) tl.SlopeBlend = sb.GetSingle();
                    if (root.TryGetProperty("proceduralWeight", out var pw)) tl.ProceduralWeight = pw.GetSingle();
                    break;
                }
                case "decoration":
                {
                    if (t.Decorations == null || index < 0 || index >= t.Decorations.Count) return CommandResult.BadRequest("Index out of range");
                    var dl = t.Decorations[index];
                    if (root.TryGetProperty("density", out var dp)) dl.Density = dp.GetSingle();
                    if (root.TryGetProperty("weight", out var wp)) dl.Weight = wp.GetSingle();
                    if (root.TryGetProperty("mesh", out var mg)) dl.Mesh = Engine.Assets.LoadByGuid<Mesh>(mg.GetString());
                    break;
                }
                default: return CommandResult.BadRequest("'category' required: height, texture, decoration");
            }
            t.MarkDirty();
            return CommandResult.Json(new { status = "updated", category, index });
        }
    }

    /// <summary>
    /// Reorder a layer. POST /api/terrain/layers/reorder
    /// Body: {"category":"texture", "from":2, "to":0}
    /// </summary>
    [CommandRoute("POST", "/api/terrain/layers/reorder")]
    public class TerrainReorderLayerCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var tr = TerrainHeightCommand.FindTerrainRenderer();
            if (tr?.Terrain == null) return CommandResult.NotFound("No terrain");
            using var doc = context.ParseBody();
            var root = doc.RootElement;
            string category = root.TryGetProperty("category", out var c) ? c.GetString() : "";
            int from = root.TryGetProperty("from", out var fp) ? fp.GetInt32() : -1;
            int to = root.TryGetProperty("to", out var tp) ? tp.GetInt32() : -1;
            var t = tr.Terrain;

            switch (category?.ToLowerInvariant())
            {
                case "height":
                    if (from < 0 || from >= t.HeightLayers.Count || to < 0 || to >= t.HeightLayers.Count) return CommandResult.BadRequest("out of range");
                    var hl = t.HeightLayers[from]; t.HeightLayers.RemoveAt(from); t.HeightLayers.Insert(to, hl); break;
                case "texture":
                    if (t.Layers == null || from < 0 || from >= t.Layers.Count || to < 0 || to >= t.Layers.Count) return CommandResult.BadRequest("out of range");
                    var tl = t.Layers[from]; t.Layers.RemoveAt(from); t.Layers.Insert(to, tl); break;
                case "decoration":
                    if (t.Decorations == null || from < 0 || from >= t.Decorations.Count || to < 0 || to >= t.Decorations.Count) return CommandResult.BadRequest("out of range");
                    var dl = t.Decorations[from]; t.Decorations.RemoveAt(from); t.Decorations.Insert(to, dl); break;
                default: return CommandResult.BadRequest("'category' required: height, texture, decoration");
            }
            t.MarkDirty();
            return CommandResult.Json(new { status = "reordered", category, from, to });
        }
    }
}
