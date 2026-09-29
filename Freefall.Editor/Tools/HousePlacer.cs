using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;

namespace Freefall.Editor.Tools
{
    /// <summary>
    /// Cached measurements for a single house prefab.
    /// </summary>
    public class PrefabProfile
    {
        public Prefab Prefab;
        public float Width;     // facade width (along road)
        public float Depth;     // perpendicular to facade
        public float Height;    // vertical extent
        /// <summary>Yaw offset so the door faces +Z after rotation.</summary>
        public float DoorAngle;
    }

    /// <summary>
    /// Places house prefabs inside block polygons or along road splines.
    /// Uses prefab profiling, best-fit packing, and road avoidance.
    /// </summary>
    public static class HousePlacer
    {
        private const float SidewalkMargin = 1.2f;  // walkable strip at block edges
        private const float HouseGapMin = 0.2f;     // minimum gap between houses
        private const float HouseGapMax = 0.8f;     // maximum random gap
        private const float RoadClearance = 1f;      // extra clearance from road edges
        private const float PavementHeight = 0.1f;

        // ═══════════════════════════════════
        // ── Prefab Profiling ──
        // ═══════════════════════════════════

        /// <summary>
        /// Instantiate each unique prefab, measure bounding box, detect door facing, then destroy.
        /// </summary>
        private static List<PrefabProfile> ProfilePrefabs(List<Prefab> prefabs)
        {
            var profiles = new List<PrefabProfile>();
            var seen = new HashSet<Prefab>();

            foreach (var prefab in prefabs)
            {
                if (prefab == null || seen.Contains(prefab)) continue;
                seen.Add(prefab);

                var probe = prefab.Instantiate();
                if (probe == null) continue;

                var profile = new PrefabProfile { Prefab = prefab };

                // Compute combined bounding box from all mesh renderers in hierarchy
                // Use WorldPosition relative to root so nested children are correct
                var renderers = probe.GetComponentsInChildren<MeshRenderer>();
                var rootPos = probe.Transform.WorldPosition;
                var min = new Vector3(-4f, 0, -4f);
                var max = new Vector3(4f, 6f, 4f);

                if (renderers.Count > 0)
                {
                    min = new Vector3(float.MaxValue);
                    max = new Vector3(float.MinValue);
                    foreach (var mr in renderers)
                    {
                        if (mr.Mesh == null) continue;
                        var bb = mr.Mesh.BoundingBox;
                        // Full hierarchy offset: world pos relative to prefab root
                        var offset = mr.Entity.Transform.WorldPosition - rootPos;
                        min = Vector3.Min(min, bb.Min + offset);
                        max = Vector3.Max(max, bb.Max + offset);
                    }

                    var size = max - min;
                    profile.Width = MathF.Max(size.X, 0.5f);
                    profile.Depth = MathF.Max(size.Z, 0.5f);
                    profile.Height = MathF.Max(size.Y, 0.5f);
                }
                else
                {
                    profile.Width = 8f;
                    profile.Depth = 8f;
                    profile.Height = 6f;
                }

                // Door detection: which bounding box face has the door?
                profile.DoorAngle = DetectDoorAngle(probe, min, max);

                Debug.Log($"[HousePlacer] Prefab '{prefab.Name}': {profile.Width:F1}x{profile.Depth:F1}x{profile.Height:F1} door={profile.DoorAngle:F2}rad");

                probe.Destroy();
                profiles.Add(profile);
            }

            if (profiles.Count > 0)
            {
                Debug.Log($"[HousePlacer] Profiled {profiles.Count} prefabs");
            }

            return profiles;
        }

        /// <summary>
        /// Determine which bounding box face (±X or ±Z) the door is on.
        /// Returns a yaw offset so that rotating by (edgeYaw - DoorAngle)
        /// aligns the door side with the road.
        /// 0 = door on +Z face, π = -Z, π/2 = +X, -π/2 = -X.
        /// </summary>
        private static float DetectDoorAngle(Entity root, Vector3 bboxMin, Vector3 bboxMax)
        {
            var children = new List<Entity>();
            CollectAllChildren(root, children);

            var center = (bboxMin + bboxMax) * 0.5f;
            var halfExt = (bboxMax - bboxMin) * 0.5f;

            foreach (var child in children)
            {
                if (child.Name == null) continue;
                if (!child.Name.Contains("door", StringComparison.OrdinalIgnoreCase) &&
                    !child.Name.Contains("entrance", StringComparison.OrdinalIgnoreCase))
                    continue;

                var doorPos = child.Transform.Position;

                // Compute normalized distances from door to each bbox face
                // Whichever face the door is closest to = the front side
                float distPosZ = MathF.Abs(doorPos.Z - bboxMax.Z); // +Z face
                float distNegZ = MathF.Abs(doorPos.Z - bboxMin.Z); // -Z face
                float distPosX = MathF.Abs(doorPos.X - bboxMax.X); // +X face
                float distNegX = MathF.Abs(doorPos.X - bboxMin.X); // -X face

                float minDist = MathF.Min(MathF.Min(distPosZ, distNegZ), MathF.Min(distPosX, distNegX));

                if (minDist == distPosZ) return 0;                    // door on +Z face
                if (minDist == distNegZ) return MathF.PI;             // door on -Z face
                if (minDist == distPosX) return MathF.PI * 0.5f;      // door on +X face
                if (minDist == distNegX) return -MathF.PI * 0.5f;     // door on -X face
            }

            return 0; // Default: assume door faces +Z
        }

        private static void CollectAllChildren(Entity entity, List<Entity> result)
        {
            int count = entity.Transform.GetChildCount();
            for (int i = 0; i < count; i++)
            {
                var child = entity.Transform.GetChild(i)?.Entity;
                if (child != null)
                {
                    result.Add(child);
                    CollectAllChildren(child, result);
                }
            }
        }

        // ═══════════════════════════════════
        // ── Block-based placement ──
        // ═══════════════════════════════════

        public static void PlaceInBlocks(Entity root, WatabouData data, WatabouSettings settings, float scale, Random rng)
        {
            if (settings.Theme?.Houses == null || settings.Theme.Houses.Count == 0 || data.Buildings.Count == 0) return;

            var profiles = ProfilePrefabs(settings.Theme.Houses);
            if (profiles.Count == 0) return;

            // Sort profiles by width (narrowest first) for packing
            profiles.Sort((a, b) => a.Width.CompareTo(b.Width));
            float maxDepth = 0;
            foreach (var p in profiles) maxDepth = MathF.Max(maxDepth, p.Depth);

            var blocksRoot = new Entity("Blocks");
            blocksRoot.Transform.Parent = root.Transform;

            var housesRoot = new Entity("Houses");
            housesRoot.Transform.Parent = root.Transform;

            // Build road exclusion zones
            var roadSegments = BuildRoadExclusion(data, scale);

            // Compute city center and radius from wall boundary
            var boundary = GetBoundary(data, scale);
            var cityCenter = Vector2.Zero;
            float cityRadius = 1f;

            if (boundary != null && boundary.Count >= 3)
            {
                foreach (var p in boundary)
                    cityCenter += p;
                cityCenter /= boundary.Count;

                foreach (var p in boundary)
                    cityRadius = MathF.Max(cityRadius, Vector2.Distance(p, cityCenter));
            }

            // Track placed houses as oriented boxes
            var placed = new List<PlacedHouse>();
            int houseIndex = 0;

            for (int b = 0; b < data.Buildings.Count; b++)
            {
                var blockPoly = data.Buildings[b];
                if (blockPoly.Count < 3) continue;

                var scaledBlock = new List<Vector2>(blockPoly.Count);
                foreach (var p in blockPoly)
                    scaledBlock.Add(p * scale);

                // Create pavement
                CreatePavementEntity(blocksRoot, $"Block_{b}", scaledBlock, settings);

                // Fill block with houses
                for (int row = 0; ; row++)
                {
                    int placedThisRow = 0;
                    float inset = SidewalkMargin + maxDepth * 0.5f + row * (maxDepth + HouseGapMin);

                    var insetPoly = InsetPolygon(scaledBlock, inset);
                    if (insetPoly == null || insetPoly.Count < 3) break;

                    for (int e = 0; e < insetPoly.Count; e++)
                    {
                        var p0 = insetPoly[e];
                        var p1 = insetPoly[(e + 1) % insetPoly.Count];
                        var edgeVec = p1 - p0;
                        float edgeLen = edgeVec.Length();
                        if (edgeLen < profiles[0].Width) continue;

                        var edgeDir = edgeVec / edgeLen;
                        var outward = new Vector2(edgeDir.Y, -edgeDir.X);
                        float yaw = MathF.Atan2(outward.X, outward.Y);

                        // Pack houses along this edge
                        float cursor = HouseGapMin;
                        int lastProfileIdx = -1;

                        while (cursor < edgeLen - HouseGapMin)
                        {
                            float remaining = edgeLen - cursor - HouseGapMin;

                            // Centrality: 0 at edge, 1 at center
                            var approxCenter = p0 + edgeDir * (cursor + remaining * 0.5f);
                            float centrality = 1f - MathF.Min(1f, Vector2.Distance(approxCenter, cityCenter) / cityRadius);

                            // Pick best-fit prefab: biased by distance to city center
                            var profile = PickPrefab(profiles, remaining, lastProfileIdx, centrality, rng);
                            if (profile == null) break;

                            float hw = profile.Width;
                            float hd = profile.Depth;

                            var center = p0 + edgeDir * (cursor + hw * 0.5f);

                            // Check inside original block
                            if (!PointInPolygon(center, scaledBlock))
                            {
                                cursor += profiles[0].Width * 0.5f;
                                continue;
                            }

                            // Check road clearance
                            if (IntersectsRoad(center, hw, hd, yaw, roadSegments))
                            {
                                cursor += profiles[0].Width * 0.5f;
                                continue;
                            }

                            // Check collision with placed houses
                            var candidate = new PlacedHouse
                            {
                                Center = center,
                                HalfW = hw * 0.5f + HouseGapMin,
                                HalfD = hd * 0.5f + HouseGapMin,
                                Forward = outward
                            };

                            if (CollidesWithPlaced(candidate, placed))
                            {
                                cursor += profiles[0].Width * 0.5f;
                                continue;
                            }

                            // Place it
                            var house = profile.Prefab.Instantiate();
                            if (house != null)
                            {
                                house.Name = $"House_{houseIndex++}";
                                house.Transform.Parent = housesRoot.Transform;
                                house.Transform.Position = new Vector3(center.X, PavementHeight, center.Y);

                                float rotation = yaw - profile.DoorAngle + MathF.PI;
                                house.Transform.Rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitY, rotation);

                                placed.Add(new PlacedHouse
                                {
                                    Center = center,
                                    HalfW = hw * 0.5f,
                                    HalfD = hd * 0.5f,
                                    Forward = outward
                                });
                                placedThisRow++;
                                lastProfileIdx = profiles.IndexOf(profile);
                            }

                            cursor += hw + HouseGapMin + rng.NextSingle() * (HouseGapMax - HouseGapMin);
                        }
                    }

                    if (placedThisRow == 0) break;
                }

                // Grid fill pass: pack remaining gaps in the block interior
                float minPW = profiles[0].Width;
                float gridStep = minPW + HouseGapMin;
                var blockMin = new Vector2(float.MaxValue);
                var blockMax = new Vector2(float.MinValue);
                foreach (var p in scaledBlock)
                {
                    blockMin = Vector2.Min(blockMin, p);
                    blockMax = Vector2.Max(blockMax, p);
                }

                // Shrink to stay off the sidewalk edge
                blockMin += new Vector2(SidewalkMargin);
                blockMax -= new Vector2(SidewalkMargin);

                for (float gy = blockMin.Y; gy < blockMax.Y; gy += gridStep)
                {
                    for (float gx = blockMin.X; gx < blockMax.X; gx += gridStep)
                    {
                        var center = new Vector2(gx, gy);
                        if (!PointInPolygon(center, scaledBlock)) continue;

                        float centrality = 1f - MathF.Min(1f, Vector2.Distance(center, cityCenter) / cityRadius);
                        var profile = PickPrefabByCentrality(profiles, centrality, rng);
                        if (profile == null) continue;

                        float hw = profile.Width;
                        float hd = profile.Depth;

                        // Align to nearest block edge for natural orientation
                        float yaw = FindNearestEdgeYaw(center, scaledBlock);
                        var fwd = new Vector2(MathF.Sin(yaw), MathF.Cos(yaw));

                        if (IntersectsRoad(center, hw, hd, yaw, roadSegments)) continue;

                        var candidate = new PlacedHouse
                        {
                            Center = center,
                            HalfW = hw * 0.5f + HouseGapMin,
                            HalfD = hd * 0.5f + HouseGapMin,
                            Forward = fwd
                        };
                        if (CollidesWithPlaced(candidate, placed)) continue;

                        var house = profile.Prefab.Instantiate();
                        if (house == null) continue;

                        house.Name = $"House_{houseIndex++}";
                        house.Transform.Parent = housesRoot.Transform;
                        house.Transform.Position = new Vector3(center.X, PavementHeight, center.Y);

                        float rotation = yaw - profile.DoorAngle + MathF.PI;
                        house.Transform.Rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitY, rotation);

                        placed.Add(new PlacedHouse
                        {
                            Center = center,
                            HalfW = hw * 0.5f,
                            HalfD = hd * 0.5f,
                            Forward = fwd
                        });
                    }
                }
            }

            float minProfileH = float.MaxValue, maxProfileH = 0;
            foreach (var p in profiles) { minProfileH = MathF.Min(minProfileH, p.Height); maxProfileH = MathF.Max(maxProfileH, p.Height); }
            Debug.Log($"[HousePlacer] Placed {houseIndex} houses in {data.Buildings.Count} blocks. " +
                $"City center=({cityCenter.X:F0},{cityCenter.Y:F0}) radius={cityRadius:F0}. " +
                $"Prefab heights: {minProfileH:F1}–{maxProfileH:F1}m");
        }


        /// <summary>
        /// Pick a prefab that fits remaining space, biased by height centrality.
        /// </summary>
        private static PrefabProfile PickPrefab(List<PrefabProfile> profiles, float remainingSpace, int lastIdx, float centrality, Random rng)
        {
            var candidates = new List<PrefabProfile>();
            foreach (var p in profiles)
                if (p.Width <= remainingSpace)
                    candidates.Add(p);

            if (candidates.Count == 0) return null;

            // Prefer variety: avoid repeating the last prefab
            if (candidates.Count > 1 && lastIdx >= 0)
            {
                var lastPrefab = profiles[lastIdx].Prefab;
                var filtered = new List<PrefabProfile>();
                foreach (var c in candidates)
                    if (c.Prefab != lastPrefab) filtered.Add(c);
                if (filtered.Count > 0) candidates = filtered;
            }

            return PickByHeightCentrality(candidates, profiles, centrality, rng);
        }

        /// <summary>
        /// Pick a prefab purely by centrality (no edge-fit constraint).
        /// Used by grid fill pass.
        /// </summary>
        private static PrefabProfile PickPrefabByCentrality(List<PrefabProfile> profiles, float centrality, Random rng)
        {
            return PickByHeightCentrality(profiles, profiles, centrality, rng);
        }

        // Higher = stricter gradient. 10-15 is a good range.
        private const float HeightCentralitySharpness = 6f;

        /// <summary>
        /// Gaussian-weighted selection by height rank.
        /// Each prefab has a "preferred centrality" based on its height rank (0=shortest, 1=tallest).
        /// Weight = exp(-sharpness * (centrality - preferredCentrality)^2)
        /// This naturally places tall buildings at the center and short ones at the edges.
        /// </summary>
        private static PrefabProfile PickByHeightCentrality(
            List<PrefabProfile> candidates, List<PrefabProfile> allProfiles,
            float centrality, Random rng)
        {
            if (candidates.Count == 0) return null;
            if (candidates.Count == 1) return candidates[0];

            // Determine height rank of each candidate within the full profile set
            float globalMinH = float.MaxValue, globalMaxH = float.MinValue;
            foreach (var p in allProfiles)
            {
                globalMinH = MathF.Min(globalMinH, p.Height);
                globalMaxH = MathF.Max(globalMaxH, p.Height);
            }
            float hRange = globalMaxH - globalMinH;

            float totalWeight = 0;
            Span<float> weights = candidates.Count <= 64
                ? stackalloc float[candidates.Count]
                : new float[candidates.Count];

            for (int i = 0; i < candidates.Count; i++)
            {
                // Rank 0 = shortest in full set, 1 = tallest
                float rank = hRange > 0.1f
                    ? (candidates[i].Height - globalMinH) / hRange
                    : 0.5f;

                float diff = centrality - rank;
                float w = MathF.Exp(-HeightCentralitySharpness * diff * diff);
                weights[i] = w;
                totalWeight += w;
            }

            float roll = rng.NextSingle() * totalWeight;
            float accum = 0;
            for (int i = 0; i < candidates.Count; i++)
            {
                accum += weights[i];
                if (roll <= accum) return candidates[i];
            }

            return candidates[^1];
        }

        // ═══════════════════════════════════
        // ── Road-based placement (fallback) ──
        // ═══════════════════════════════════

        public static void PlaceAlongRoads(Entity root, WatabouData data, WatabouSettings settings, float scale, Random rng)
        {
            if (settings.Theme?.Houses == null || settings.Theme.Houses.Count == 0) return;

            var boundary = GetBoundary(data, scale);
            if (boundary == null || boundary.Count < 3)
            {
                Debug.Log("[HousePlacer] No wall or earth boundary — skipping");
                return;
            }

            var profiles = ProfilePrefabs(settings.Theme.Houses);
            if (profiles.Count == 0) return;

            float maxWidth = 0, maxDepth = 0;
            foreach (var p in profiles) { maxWidth = MathF.Max(maxWidth, p.Width); maxDepth = MathF.Max(maxDepth, p.Depth); }

            var housesRoot = new Entity("Houses");
            housesRoot.Transform.Parent = root.Transform;

            var roads = new List<WatabouPolyline>(data.Roads);
            roads.Sort((a, b) => PolylineLength(b, scale).CompareTo(PolylineLength(a, scale)));

            var placed = new List<PlacedHouse>();
            float stepAlong = maxWidth + HouseGapMin;
            int houseIndex = 0;

            for (int row = 0; ; row++)
            {
                int placedThisRow = 0;

                foreach (var road in roads)
                {
                    if (road.Points.Count < 2) continue;
                    float roadHalfWidth = road.Width * scale * 0.5f;
                    float offset = roadHalfWidth + SidewalkMargin + maxDepth * 0.5f + row * (maxDepth + HouseGapMin);

                    for (int side = -1; side <= 1; side += 2)
                    {
                        float distAccum = stepAlong * 0.5f;

                        for (int e = 0; e < road.Points.Count - 1; e++)
                        {
                            var p0 = road.Points[e] * scale;
                            var p1 = road.Points[e + 1] * scale;
                            var edgeVec = p1 - p0;
                            float edgeLen = edgeVec.Length();
                            if (edgeLen < 0.1f) continue;

                            var edgeDir = edgeVec / edgeLen;
                            var perpDir = new Vector2(-edgeDir.Y, edgeDir.X);

                            while (distAccum < edgeLen)
                            {
                                var roadPt = p0 + edgeDir * distAccum;
                                var center = roadPt + perpDir * (offset * side);

                                if (!PointInPolygon(center, boundary))
                                { distAccum += stepAlong; continue; }

                                var profile = profiles[rng.Next(profiles.Count)];
                                var candidate = new PlacedHouse
                                {
                                    Center = center,
                                    HalfW = profile.Width * 0.5f + HouseGapMin,
                                    HalfD = profile.Depth * 0.5f + HouseGapMin,
                                    Forward = perpDir * side
                                };

                                if (!CollidesWithPlaced(candidate, placed))
                                {
                                    var house = profile.Prefab.Instantiate();
                                    if (house != null)
                                    {
                                        house.Name = $"House_{houseIndex++}";
                                        house.Transform.Parent = housesRoot.Transform;
                                        house.Transform.Position = new Vector3(center.X, 0, center.Y);

                                        float yaw = MathF.Atan2(-perpDir.X * side, -perpDir.Y * side);
                                        float rotation = yaw - profile.DoorAngle + MathF.PI;
                                        house.Transform.Rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitY, rotation);

                                        placed.Add(new PlacedHouse
                                        {
                                            Center = center,
                                            HalfW = profile.Width * 0.5f,
                                            HalfD = profile.Depth * 0.5f,
                                            Forward = perpDir * side
                                        });
                                        placedThisRow++;
                                    }
                                }

                                distAccum += stepAlong;
                            }
                            distAccum -= edgeLen;
                        }
                    }
                }

                if (placedThisRow == 0) break;
            }

            Debug.Log($"[HousePlacer] Placed {houseIndex} houses along roads");
        }

        // ═══════════════════════════════════
        // ── Pavement creation ──
        // ═══════════════════════════════════

        private static void CreatePavementEntity(Entity parent, string name, List<Vector2> polygon, WatabouSettings settings)
        {
            var entity = new Entity(name);
            entity.Transform.Parent = parent.Transform;

            var spline = entity.AddComponent<Spline>();
            spline.Points.Clear();
            spline.Closed = true;
            spline.Tension = 1f;
            foreach (var p in polygon)
                spline.Points.Add(new Vector3(p.X, 0, p.Y));

            var rtMesh = entity.AddComponent<RuntimeMesh>();
            rtMesh.Height = PavementHeight;
            rtMesh.HeightMode = RuntimeMeshHeightMode.Surface;
            rtMesh.Smoothness = 1;

            var renderer = entity.AddComponent<MeshRenderer>();
            renderer.Material = settings.Theme?.SidewalkMaterial ?? InternalAssets.DefaultMaterial;

            if (settings.AddTerrainInfluence)
            {
                var heightStamp = entity.AddComponent<HeightStamp>();
                heightStamp.Radius = 1f;
                heightStamp.Falloff = 1f;

                var decoStamp = entity.AddComponent<DecoStamp>();
                decoStamp.Radius = 1f;
                decoStamp.Falloff = 1f;
            }
        }

        // ═══════════════════════════════════
        // ── Collision & Road Avoidance ──
        // ═══════════════════════════════════

        private struct PlacedHouse
        {
            public Vector2 Center;
            public float HalfW, HalfD; // half-extents (W along edge, D perpendicular)
            public Vector2 Forward;     // unit direction house faces (perpendicular to edge)
        }

        private struct RoadSegment
        {
            public Vector2 A, B;
            public float HalfWidth;
        }

        private static List<RoadSegment> BuildRoadExclusion(WatabouData data, float scale)
        {
            var segments = new List<RoadSegment>();
            foreach (var road in data.Roads)
            {
                float hw = road.Width * scale * 0.5f + RoadClearance;
                for (int i = 0; i < road.Points.Count - 1; i++)
                {
                    segments.Add(new RoadSegment
                    {
                        A = road.Points[i] * scale,
                        B = road.Points[i + 1] * scale,
                        HalfWidth = hw
                    });
                }
            }
            return segments;
        }

        /// <summary>Check if a house at center with given size/yaw intersects any road.</summary>
        private static bool IntersectsRoad(Vector2 center, float w, float d, float yaw, List<RoadSegment> roads)
        {
            float houseRadius = MathF.Sqrt(w * w + d * d) * 0.5f;

            foreach (var seg in roads)
            {
                // Quick: point-to-segment distance vs combined radius
                float dist = PointSegmentDist(center, seg.A, seg.B);
                if (dist < seg.HalfWidth + houseRadius * 0.7f) // 0.7 tightens for rectangular shapes
                    return true;
            }
            return false;
        }

        private static float PointSegmentDist(Vector2 p, Vector2 a, Vector2 b)
        {
            var ab = b - a;
            float len2 = ab.LengthSquared();
            if (len2 < 1e-6f) return Vector2.Distance(p, a);
            float t = MathF.Max(0, MathF.Min(1, Vector2.Dot(p - a, ab) / len2));
            var proj = a + ab * t;
            return Vector2.Distance(p, proj);
        }

        /// <summary>Approximate OBB overlap check using separating axis on the forward vectors.</summary>
        private static bool CollidesWithPlaced(PlacedHouse candidate, List<PlacedHouse> placed)
        {
            foreach (var other in placed)
            {
                var delta = candidate.Center - other.Center;
                float dist = delta.Length();
                float maxR = candidate.HalfW + candidate.HalfD + other.HalfW + other.HalfD;
                if (dist > maxR) continue; // quick reject

                // Separating axis test using both forward directions
                if (OBBOverlap(candidate, other))
                    return true;
            }
            return false;
        }

        private static bool OBBOverlap(PlacedHouse a, PlacedHouse b)
        {
            var aFwd = a.Forward;
            var aRight = new Vector2(-aFwd.Y, aFwd.X);
            var bFwd = b.Forward;
            var bRight = new Vector2(-bFwd.Y, bFwd.X);

            var delta = b.Center - a.Center;

            // Test 4 axes
            return
                AxisOverlap(delta, aFwd, a.HalfD, a.HalfW, b.HalfD, b.HalfW, aFwd, aRight, bFwd, bRight) &&
                AxisOverlap(delta, aRight, a.HalfD, a.HalfW, b.HalfD, b.HalfW, aFwd, aRight, bFwd, bRight) &&
                AxisOverlap(delta, bFwd, a.HalfD, a.HalfW, b.HalfD, b.HalfW, aFwd, aRight, bFwd, bRight) &&
                AxisOverlap(delta, bRight, a.HalfD, a.HalfW, b.HalfD, b.HalfW, aFwd, aRight, bFwd, bRight);
        }

        private static bool AxisOverlap(Vector2 delta, Vector2 axis,
            float aHalfD, float aHalfW, float bHalfD, float bHalfW,
            Vector2 aFwd, Vector2 aRight, Vector2 bFwd, Vector2 bRight)
        {
            float projDelta = MathF.Abs(Vector2.Dot(delta, axis));
            float projA = aHalfD * MathF.Abs(Vector2.Dot(aFwd, axis)) +
                          aHalfW * MathF.Abs(Vector2.Dot(aRight, axis));
            float projB = bHalfD * MathF.Abs(Vector2.Dot(bFwd, axis)) +
                          bHalfW * MathF.Abs(Vector2.Dot(bRight, axis));
            return projDelta < projA + projB;
        }

        // ═══════════════════════════════════
        // ── Geometry Helpers ──
        // ═══════════════════════════════════

        /// <summary>Find the nearest block edge to a point, return the outward-facing yaw.</summary>
        private static float FindNearestEdgeYaw(Vector2 point, List<Vector2> polygon)
        {
            float bestDist = float.MaxValue;
            Vector2 bestOutward = Vector2.UnitY;

            for (int i = 0; i < polygon.Count; i++)
            {
                var a = polygon[i];
                var b = polygon[(i + 1) % polygon.Count];
                float dist = PointSegmentDist(point, a, b);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    var edgeDir = Vector2.Normalize(b - a);
                    bestOutward = new Vector2(edgeDir.Y, -edgeDir.X);
                }
            }

            return MathF.Atan2(bestOutward.X, bestOutward.Y);
        }

        private static List<Vector2> InsetPolygon(List<Vector2> poly, float dist)
        {
            int n = poly.Count;
            if (n < 3) return null;

            var normals = new Vector2[n];
            for (int i = 0; i < n; i++)
            {
                var edge = poly[(i + 1) % n] - poly[i];
                var dir = Vector2.Normalize(edge);
                normals[i] = new Vector2(dir.Y, -dir.X);
            }

            float area = 0;
            for (int i = 0; i < n; i++)
            {
                var a = poly[i];
                var b = poly[(i + 1) % n];
                area += (b.X - a.X) * (b.Y + a.Y);
            }
            if (area < 0) // CCW — flip normals to point inward
            {
                for (int i = 0; i < n; i++)
                    normals[i] = -normals[i];
            }

            var result = new List<Vector2>();
            for (int i = 0; i < n; i++)
            {
                int prev = (i - 1 + n) % n;
                var a0 = poly[prev] + normals[prev] * dist;
                var a1 = poly[i] + normals[prev] * dist;
                var b0 = poly[i] + normals[i] * dist;
                var b1 = poly[(i + 1) % n] + normals[i] * dist;

                if (LineIntersect(a0, a1, b0, b1, out var pt))
                    result.Add(pt);
                else
                    result.Add(poly[i] + (normals[prev] + normals[i]) * 0.5f * dist);
            }

            return result.Count >= 3 ? result : null;
        }

        private static bool LineIntersect(Vector2 a0, Vector2 a1, Vector2 b0, Vector2 b1, out Vector2 result)
        {
            var dA = a1 - a0;
            var dB = b1 - b0;
            float denom = dA.X * dB.Y - dA.Y * dB.X;
            result = Vector2.Zero;
            if (MathF.Abs(denom) < 1e-8f) return false;
            var diff = b0 - a0;
            float t = (diff.X * dB.Y - diff.Y * dB.X) / denom;
            result = a0 + dA * t;
            return true;
        }

        private static List<Vector2> GetBoundary(WatabouData data, float scale)
        {
            if (data.Walls.Count > 0 && data.Walls[0].Points.Count >= 3)
            {
                var result = new List<Vector2>(data.Walls[0].Points.Count);
                foreach (var p in data.Walls[0].Points) result.Add(p * scale);
                return result;
            }
            if (data.EarthBoundary.Count >= 3)
            {
                var result = new List<Vector2>(data.EarthBoundary.Count);
                foreach (var p in data.EarthBoundary) result.Add(p * scale);
                return result;
            }
            return null;
        }

        private static bool PointInPolygon(Vector2 point, List<Vector2> polygon)
        {
            int n = polygon.Count;
            bool inside = false;
            for (int i = 0, j = n - 1; i < n; j = i++)
            {
                if ((polygon[i].Y > point.Y) != (polygon[j].Y > point.Y) &&
                    point.X < (polygon[j].X - polygon[i].X) * (point.Y - polygon[i].Y) /
                              (polygon[j].Y - polygon[i].Y) + polygon[i].X)
                    inside = !inside;
            }
            return inside;
        }

        private static float PolylineLength(WatabouPolyline line, float scale)
        {
            float len = 0;
            for (int i = 0; i < line.Points.Count - 1; i++)
                len += Vector2.Distance(line.Points[i] * scale, line.Points[i + 1] * scale);
            return len;
        }
    }
}
