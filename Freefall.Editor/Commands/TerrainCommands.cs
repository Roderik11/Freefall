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
    /// Returns terrain metadata: dimensions, resolution, height range, world bounds, and the palette —
    /// the layers and decorators the terrain renders, which are derived from the stamps in the scene
    /// (a TerrainLayer gets a channel when a SplatStamp references it, a TerrainDecorator a slot when
    /// a DecoStamp adds it).
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
                },
                layers = terrainRenderer.LayerPalette.Select((l, i) => new
                {
                    channel = i,
                    name = l.Name,
                    guid = l.Guid,
                    diffuse = l.Diffuse?.Name,
                }),
                decorators = terrainRenderer.DecoratorPalette.Select((d, i) => new
                {
                    slot = i,
                    name = d.Name,
                    guid = d.Guid,
                    variants = d.Variants.Count,
                }),
                warnings = terrainRenderer.PaletteWarnings,
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
}
