using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graph;

namespace Freefall.PCG
{
    public enum SamplingMode
    {
        /// <summary>Sample at fixed distance intervals along the spline curve.</summary>
        EvenSpacing,

        /// <summary>Walk control point edges directly, subdividing by Spacing.</summary>
        PerEdge,

        /// <summary>
        /// Sample points within the spline's 2D area (requires closed spline).
        /// </summary>
        Area,

        /// <summary>
        /// Rows over the closed spline's area, Spacing apart and Spacing between the points of a row, the rows along
        /// the longest control edge: an orchard, a vineyard, a planted copse.
        /// </summary>
        Rows,

        /// <summary>
        /// Pieces end to end round the outline of the control points, moved EdgeInset in (closed splines) with
        /// mitred corners. The outline splits into runs where it turns 20° or more, and each run is divided evenly
        /// into segments no longer than Spacing, one point at the middle of each, facing along its run. The runs
        /// end at the corners, so a hedge or fence round a plot meets itself there instead of crossing.
        /// </summary>
        Outline
    }

    /// <summary>
    /// Samples points along a Spline component, producing a SamplePointSet
    /// with position and rotation (aligned to edge tangent, Y-up).
    /// 
    /// EvenSpacing: uses Catmull-Rom evaluation at fixed arc-length intervals (arc-length table, so
    ///              dense control points don't concentrate samples).
    /// PerEdge: walks each control point pair as a straight segment, subdividing
    ///          by Spacing. Best for polygon-like splines (e.g., Watabou walls).
    /// </summary>
    [Category("Sampler")]
    public class SplineSampler : Node
    {
        /// <summary>
        /// Source spline. Injected by PCGComponent from the entity's Spline component.
        /// </summary>
        [System.ComponentModel.Browsable(false)]
        public Spline Spline;

        public SamplingMode Mode = SamplingMode.PerEdge;

        /// <summary>Distance between samples in meters.</summary>
        [ValueRange(0.5f, 50f)]
        public float Spacing = 5f;

        /// <summary>
        /// PerEdge / EvenSpacing on a closed spline: move each sample this far toward the polygon interior
        /// (negative = outward), independent of winding. E.g. props on a sidewalk just inside a curb.
        /// Outline: the outline itself moves in (mitred corners). Area / Rows: points nearer the outline than this
        /// are left out, so what grows there keeps inside the area (bushes and trees in a yard).
        /// </summary>
        [ValueRange(-20f, 20f)]
        [Description("How far in from the outline (m). PerEdge / EvenSpacing: each sample moves in. Outline: the outline moves in, mitred at the corners. Area / Rows: points nearer the outline are left out.")]
        public float EdgeInset = 0f;

        /// <summary>
        /// PerEdge / EvenSpacing / Outline on a closed spline: turn samples so their local +X points out of the polygon
        /// (by default +X is the right-hand side of travel, which flips with winding). E.g. lantern arms over the street.
        /// </summary>
        public bool OrientOutward = false;

        /// <summary>Outline: a turn of the outline at least this sharp ends a run (degrees).</summary>
        private const float OutlineCornerDegrees = 20f;

        [Output]
        public SamplePointSet Output;

        public override void Process()
        {
            if (Spline == null || Spline.Points.Count < 2)
            {
                SetOutput("Output", SamplePointSet.Empty());
                return;
            }

            var result = Mode switch
            {
                SamplingMode.EvenSpacing => SampleEvenSpacing(Spline),
                SamplingMode.PerEdge => SamplePerEdge(Spline),
                SamplingMode.Area => SampleArea(Spline),
                SamplingMode.Rows => SampleRows(Spline),
                SamplingMode.Outline => SampleOutline(Spline),
                _ => SamplePointSet.Empty()
            };

            if ((EdgeInset != 0f || OrientOutward) && (Mode == SamplingMode.EvenSpacing || Mode == SamplingMode.PerEdge) && Spline.Closed)
                ApplyEdgeFrame(result, Spline);

            // Area and Rows keep EdgeInset clear of the outline (until 2026-11-17 they ignored it: a yard's bushes
            // set 2.8 m in grew over its curb)
            if (EdgeInset > 0f && (Mode == SamplingMode.Area || Mode == SamplingMode.Rows) && Spline.Closed)
                result = KeepClearOfOutline(result, Spline.Points, EdgeInset);

            SetOutput("Output", result);
            Debug.Log($"[SplineSampler] Produced {result.Count} samples (mode={Mode}, spacing={Spacing}m)");
        }


        /// <summary>
        /// Rows over a closed spline's area: points Spacing apart along rows Spacing apart, the rows running along its
        /// longest control edge, each point facing along its row. The grid is centred on the area, so the outer rows keep
        /// about half a spacing from the outline wherever it runs parallel to them.
        /// </summary>
        private SamplePointSet SampleRows(Spline spline)
        {
            if (!spline.Closed)
            {
                Debug.LogWarning("[SplineSampler] Rows sampling requires a closed spline.");
                return SamplePointSet.Empty();
            }

            var pts = spline.Points;
            int n = pts.Count;
            var along = Vector2.UnitX;
            float longest = 0f;
            for (int i = 0; i < n; i++)
            {
                var a = pts[i];
                var b = pts[(i + 1) % n];
                var edge = new Vector2(b.X - a.X, b.Z - a.Z);
                float lengthSq = edge.LengthSquared();
                if (lengthSq > longest) { longest = lengthSq; along = edge / MathF.Sqrt(lengthSq); }
            }
            var across = new Vector2(-along.Y, along.X);

            float u0 = float.MaxValue, u1 = float.MinValue, v0 = float.MaxValue, v1 = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                var p = new Vector2(pts[i].X, pts[i].Z);
                float u = Vector2.Dot(p, along), v = Vector2.Dot(p, across);
                u0 = MathF.Min(u0, u); u1 = MathF.Max(u1, u);
                v0 = MathF.Min(v0, v); v1 = MathF.Max(v1, v);
            }

            float s = MathF.Max(Spacing, 0.1f);
            int nu = (int)((u1 - u0) / s), nv = (int)((v1 - v0) / s);
            float uStart = (u0 + u1) * 0.5f - nu * s * 0.5f, vStart = (v0 + v1) * 0.5f - nv * s * 0.5f;
            var rotation = LookRotation(new Vector3(along.X, 0f, along.Y), Vector3.UnitY);

            var positions = new List<Vector3>();
            var rotations = new List<Quaternion>();
            for (int j = 0; j <= nv; j++)
                for (int i = 0; i <= nu; i++)
                {
                    var q = along * (uStart + i * s) + across * (vStart + j * s);
                    if (!PointInPolygonXZ(q.X, q.Y, pts, n)) continue;
                    positions.Add(new Vector3(q.X, 0f, q.Y));
                    rotations.Add(rotation);
                }
            return BuildResult(positions, rotations);
        }

        /// <summary>
        /// Sample points within the spline's 2D area. Only valid for closed splines.
        /// </summary>
        /// <param name="spline"></param>
        /// <returns></returns>
        private SamplePointSet SampleArea(Spline spline)
        {
            if (!spline.Closed)
            {
                Debug.LogWarning("[SplineSampler] Area sampling requires a closed spline.");
                return SamplePointSet.Empty();
            }

            var pts = spline.Points;
            int n = pts.Count;

            // AABB of control points in XZ
            float minX = float.MaxValue, minZ = float.MaxValue;
            float maxX = float.MinValue, maxZ = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                var p = pts[i];
                if (p.X < minX) minX = p.X;
                if (p.X > maxX) maxX = p.X;
                if (p.Z < minZ) minZ = p.Z;
                if (p.Z > maxZ) maxZ = p.Z;
            }

            float r = Spacing;
            float cellSize = r / MathF.Sqrt(2f);
            int gridW = Math.Max(1, (int)MathF.Ceiling((maxX - minX) / cellSize));
            int gridH = Math.Max(1, (int)MathF.Ceiling((maxZ - minZ) / cellSize));
            int[] grid = new int[gridW * gridH];
            Array.Fill(grid, -1);

            var rng = CreateRandom();
            var accepted = new List<Vector3>();
            var active = new List<int>();
            const int k = 30;

            // Seed: try center of AABB first, fall back to random attempts
            var seed2D = new Vector2((minX + maxX) * 0.5f, (minZ + maxZ) * 0.5f);
            if (!PointInPolygonXZ(seed2D.X, seed2D.Y, pts, n))
            {
                seed2D = default;
                for (int attempt = 0; attempt < 100; attempt++)
                {
                    float sx = minX + (float)rng.NextDouble() * (maxX - minX);
                    float sz = minZ + (float)rng.NextDouble() * (maxZ - minZ);
                    if (PointInPolygonXZ(sx, sz, pts, n))
                    {
                        seed2D = new Vector2(sx, sz);
                        break;
                    }
                }
                if (seed2D == default) return SamplePointSet.Empty();
            }

            accepted.Add(new Vector3(seed2D.X, 0, seed2D.Y));
            active.Add(0);
            int gx = (int)((seed2D.X - minX) / cellSize);
            int gz = (int)((seed2D.Y - minZ) / cellSize);
            if (gx >= 0 && gx < gridW && gz >= 0 && gz < gridH)
                grid[gz * gridW + gx] = 0;

            while (active.Count > 0)
            {
                int idx = rng.Next(active.Count);
                var current = accepted[active[idx]];
                bool found = false;

                for (int j = 0; j < k; j++)
                {
                    float angle = (float)(rng.NextDouble() * Math.PI * 2);
                    float dist = r + (float)rng.NextDouble() * r;
                    float cx = current.X + MathF.Cos(angle) * dist;
                    float cz = current.Z + MathF.Sin(angle) * dist;

                    if (cx < minX || cx > maxX || cz < minZ || cz > maxZ)
                        continue;

                    int ci = (int)((cx - minX) / cellSize);
                    int cj = (int)((cz - minZ) / cellSize);
                    if (ci < 0 || ci >= gridW || cj < 0 || cj >= gridH)
                        continue;

                    // Check neighbors in 5x5 grid window
                    bool tooClose = false;
                    int i0 = Math.Max(0, ci - 2), i1 = Math.Min(gridW - 1, ci + 2);
                    int j0 = Math.Max(0, cj - 2), j1 = Math.Min(gridH - 1, cj + 2);
                    for (int ni = i0; ni <= i1 && !tooClose; ni++)
                    {
                        for (int nj = j0; nj <= j1 && !tooClose; nj++)
                        {
                            int si = grid[nj * gridW + ni];
                            if (si < 0) continue;
                            var s = accepted[si];
                            float dx = cx - s.X;
                            float dz = cz - s.Z;
                            if (dx * dx + dz * dz < r * r)
                                tooClose = true;
                        }
                    }

                    if (tooClose) continue;
                    if (!PointInPolygonXZ(cx, cz, pts, n)) continue;

                    int newIdx = accepted.Count;
                    accepted.Add(new Vector3(cx, 0, cz));
                    active.Add(newIdx);
                    grid[cj * gridW + ci] = newIdx;
                    found = true;
                }

                if (!found)
                    active.RemoveAt(idx);
            }

            var rotations = new List<Quaternion>(accepted.Count);
            for (int i = 0; i < accepted.Count; i++)
                rotations.Add(Quaternion.Identity);

            return BuildResult(accepted, rotations);
        }

        /// <summary>
        /// Ray-cast point-in-polygon test on the XZ plane.
        /// </summary>
        private static bool PointInPolygonXZ(float px, float pz, IList<Vector3> polygon, int count)
        {
            bool inside = false;
            for (int i = 0, j = count - 1; i < count; j = i++)
            {
                float iz = polygon[i].Z, jz = polygon[j].Z;
                if ((iz > pz) != (jz > pz) &&
                    px < (polygon[j].X - polygon[i].X) * (pz - iz) / (jz - iz) + polygon[i].X)
                    inside = !inside;
            }
            return inside;
        }

        /// <summary>
        /// Sample using Catmull-Rom evaluation at even arc-length intervals.
        /// </summary>
        private SamplePointSet SampleEvenSpacing(Spline spline)
        {
            // Spline t is uniform per span, not per metre: short spans (dense control points) would get as many
            // samples as long ones. Build an arc-length table and invert it so samples are evenly spaced in distance.
            int steps = Math.Max(64, spline.SpanCount * 32);
            var ts = new float[steps + 1];
            var dist = new float[steps + 1];
            Vector3 prev = spline.GetPoint(0f);
            for (int i = 1; i <= steps; i++)
            {
                ts[i] = (float)i / steps;
                Vector3 curr = spline.GetPoint(ts[i]);
                dist[i] = dist[i - 1] + Vector3.Distance(prev, curr);
                prev = curr;
            }

            float totalLength = dist[steps];
            if (totalLength < 0.01f) return SamplePointSet.Empty();

            int intervals = Math.Max(1, (int)MathF.Round(totalLength / Spacing));
            float actualSpacing = totalLength / intervals;
            // Closed: the end coincides with the start, so skip it. Open: include both ends.
            int count = spline.Closed ? intervals : intervals + 1;

            var positions = new List<Vector3>(count);
            var rotations = new List<Quaternion>(count);

            int seg = 0;
            for (int i = 0; i < count; i++)
            {
                float s = Math.Min(i * actualSpacing, totalLength);
                while (seg < steps - 1 && dist[seg + 1] < s) seg++;
                float segLen = dist[seg + 1] - dist[seg];
                float f = segLen > 1e-5f ? (s - dist[seg]) / segLen : 0f;
                float t = ts[seg] + (ts[seg + 1] - ts[seg]) * f;

                positions.Add(spline.GetPoint(t));
                rotations.Add(LookRotation(spline.GetTangent(t), Vector3.UnitY));
            }

            return BuildResult(positions, rotations);
        }

        /// <summary>
        /// Walk each control point pair as a straight edge segment.
        /// Subdivides each edge by Spacing, producing evenly-spaced samples
        /// along each straight segment.
        /// </summary>
        private SamplePointSet SamplePerEdge(Spline spline)
        {
            int n = spline.Points.Count;
            int edgeCount = spline.Closed ? n : n - 1;

            var positions = new List<Vector3>();
            var rotations = new List<Quaternion>();

            for (int edge = 0; edge < edgeCount; edge++)
            {
                int next = (edge + 1) % n;
                var p0 = spline.Points[edge];
                var p1 = spline.Points[next];

                var edgeVec = p1 - p0;
                float edgeLen = edgeVec.Length();
                if (edgeLen < 0.01f) continue;

                var dir = edgeVec / edgeLen;
                var rot = LookRotation(dir, Vector3.UnitY);

                // How many segments fit on this edge
                int segments = Math.Max(1, (int)MathF.Round(edgeLen / Spacing));
                float actualSpacing = edgeLen / segments;

                for (int s = 0; s < segments; s++)
                {
                    float t = (s + 0.5f) * actualSpacing; // center of each segment
                    positions.Add(p0 + dir * t);
                    rotations.Add(rot);
                }
            }

            return BuildResult(positions, rotations);
        }

        /// <summary>
        /// Shift samples perpendicular to their edge direction toward the polygon interior (EdgeInset), and
        /// optionally turn them so local +X faces outward (OrientOutward).
        /// </summary>
        private void ApplyEdgeFrame(SamplePointSet set, Spline spline)
        {
            // Signed area in XZ (shoelace, x→right, z→up): > 0 = counter-clockwise = interior on the left
            float area = 0;
            var pts = spline.Points;
            for (int i = 0; i < pts.Count; i++)
            {
                var a = pts[i];
                var b = pts[(i + 1) % pts.Count];
                area += a.X * b.Z - b.X * a.Z;
            }
            float side = area > 0 ? 1f : -1f;

            for (int i = 0; i < set.Count; i++)
            {
                var fwd = Vector3.Transform(Vector3.UnitZ, set.rotation[i]);
                // Left of the travel direction in XZ (z-up 2D convention): (-fwd.Z, fwd.X)
                var left = Vector3.Normalize(new Vector3(-fwd.Z, 0, fwd.X));
                set.position[i] += left * (EdgeInset * side);

                // Local +X is the right of travel = outward only for counter-clockwise polygons
                if (OrientOutward && side < 0)
                    set.rotation[i] = Quaternion.Normalize(set.rotation[i] * Quaternion.CreateFromAxisAngle(Vector3.UnitY, MathF.PI));
            }
        }

        /// <summary>The points at least <paramref name="clearance"/> from every edge of the closed control polygon.</summary>
        private static SamplePointSet KeepClearOfOutline(SamplePointSet set, IList<Vector3> polygon, float clearance)
        {
            int n = polygon.Count;
            float clearanceSq = clearance * clearance;
            return set.Filter(i =>
            {
                var p = new Vector2(set.position[i].X, set.position[i].Z);
                for (int k = 0; k < n; k++)
                {
                    var a = new Vector2(polygon[k].X, polygon[k].Z);
                    var b = new Vector2(polygon[(k + 1) % n].X, polygon[(k + 1) % n].Z);
                    var ab = b - a;
                    float t = Math.Clamp(Vector2.Dot(p - a, ab) / Math.Max(ab.LengthSquared(), 1e-6f), 0f, 1f);
                    if (Vector2.DistanceSquared(p, a + ab * t) < clearanceSq) return false;
                }
                return true;
            });
        }

        /// <summary>
        /// Outline mode: the control polygon (moved EdgeInset in when closed) split into runs at its corners, each run
        /// divided evenly into segments no longer than Spacing, a point at the middle of each facing along the run.
        /// </summary>
        private SamplePointSet SampleOutline(Spline spline)
        {
            var points = new List<Vector3>(spline.Points.Count);
            foreach (var p in spline.Points)
                if (points.Count == 0 || Vector3.DistanceSquared(points[^1], p) > 1e-4f) points.Add(p);
            bool closed = spline.Closed && points.Count >= 3;
            if (closed && Vector3.DistanceSquared(points[0], points[^1]) <= 1e-4f) points.RemoveAt(points.Count - 1);
            int count = points.Count;
            var positions = new List<Vector3>();
            var rotations = new List<Quaternion>();
            if (count < 2) return BuildResult(positions, rotations);

            // Winding as ApplyEdgeFrame reads it: side > 0 = the inside is on the left of travel
            float side = 1f;
            if (closed)
            {
                float area = 0;
                for (int i = 0; i < count; i++)
                {
                    var a = points[i];
                    var b = points[(i + 1) % count];
                    area += a.X * b.Z - b.X * a.Z;
                }
                side = area > 0 ? 1f : -1f;
                if (EdgeInset != 0f) points = MitredOffset(points, EdgeInset * side);
            }

            // Corners: turns of OutlineCornerDegrees or more (an open line's ends too)
            float cosLimit = MathF.Cos(OutlineCornerDegrees * MathF.PI / 180f);
            var corner = new bool[count];
            int corners = 0;
            for (int i = 0; i < count; i++)
            {
                if (!closed && (i == 0 || i == count - 1)) { corner[i] = true; continue; }
                var d0 = FlatDir(points[(i - 1 + count) % count], points[i]);
                var d1 = FlatDir(points[i], points[(i + 1) % count]);
                if (Vector2.Dot(d0, d1) < cosLimit) { corner[i] = true; corners++; }
            }

            void Run(List<Vector3> run)
            {
                var cumulative = new float[run.Count];
                for (int k = 1; k < run.Count; k++) cumulative[k] = cumulative[k - 1] + Vector3.Distance(run[k - 1], run[k]);
                float length = cumulative[^1];
                if (length < 0.01f) return;
                int segments = Math.Max(1, (int)MathF.Ceiling(length / MathF.Max(Spacing, 0.1f) - 1e-3f));
                float step = length / segments;
                int seg = 0;
                for (int s = 0; s < segments; s++)
                {
                    float d = (s + 0.5f) * step;
                    while (seg < run.Count - 2 && cumulative[seg + 1] < d) seg++;
                    float span = cumulative[seg + 1] - cumulative[seg];
                    float f = span > 1e-5f ? (d - cumulative[seg]) / span : 0f;
                    var dir = run[seg + 1] - run[seg];
                    var rotation = LookRotation(new Vector3(dir.X, 0f, dir.Z), Vector3.UnitY);
                    if (OrientOutward && closed && side < 0)
                        rotation = Quaternion.Normalize(rotation * Quaternion.CreateFromAxisAngle(Vector3.UnitY, MathF.PI));
                    positions.Add(Vector3.Lerp(run[seg], run[seg + 1], f));
                    rotations.Add(rotation);
                }
            }

            if (closed && corners == 0)
            {
                // A smooth loop: one run all the way round
                var loop = new List<Vector3>(points) { points[0] };
                Run(loop);
            }
            else
            {
                int start = 0;
                if (closed) while (!corner[start]) start++;
                int i = start;
                do
                {
                    var run = new List<Vector3> { points[i] };
                    int j = i;
                    do
                    {
                        j = closed ? (j + 1) % count : j + 1;
                        run.Add(points[j]);
                    } while (!corner[j]);
                    Run(run);
                    i = j;
                } while (closed ? i != start : i < count - 1);
            }
            return BuildResult(positions, rotations);
        }

        private static Vector2 FlatDir(Vector3 a, Vector3 b)
        {
            var d = new Vector2(b.X - a.X, b.Z - a.Z);
            return d.LengthSquared() > 1e-10f ? Vector2.Normalize(d) : Vector2.UnitX;
        }

        /// <summary>
        /// A closed polygon moved <paramref name="distance"/> to the left of travel (XZ, z-up 2D), each corner on the
        /// bisector of its two edges at the mitre length (capped at three times the distance for very sharp corners).
        /// </summary>
        private static List<Vector3> MitredOffset(List<Vector3> polygon, float distance)
        {
            int n = polygon.Count;
            var result = new List<Vector3>(n);
            for (int i = 0; i < n; i++)
            {
                var d0 = FlatDir(polygon[(i - 1 + n) % n], polygon[i]);
                var d1 = FlatDir(polygon[i], polygon[(i + 1) % n]);
                var n0 = new Vector2(-d0.Y, d0.X);
                var n1 = new Vector2(-d1.Y, d1.X);
                var bisector = n0 + n1;
                bisector = bisector.LengthSquared() > 1e-8f ? Vector2.Normalize(bisector) : n0;
                float mitre = distance / MathF.Max(Vector2.Dot(bisector, n0), 1f / 3f);
                result.Add(polygon[i] + new Vector3(bisector.X, 0f, bisector.Y) * mitre);
            }
            return result;
        }

        private static SamplePointSet BuildResult(List<Vector3> positions, List<Quaternion> rotations)
        {
            int count = positions.Count;
            var density = new float[count];
            Array.Fill(density, 1f);

            var extents = new Vector3[count];
            Array.Fill(extents, Vector3.One);

            return new SamplePointSet
            {
                position = positions.ToArray(),
                extents = extents,
                rotation = rotations.ToArray(),
                density = density,
                tags = new string[count]
            };
        }

        /// <summary>
        /// Create a rotation that looks along 'forward' with 'up' as the up vector.
        /// Equivalent to Matrix4x4.CreateLookAt-style orientation.
        /// </summary>
        private static Quaternion LookRotation(Vector3 forward, Vector3 up)
        {
            forward = Vector3.Normalize(forward);
            if (forward.LengthSquared() < 0.001f) return Quaternion.Identity;

            var right = Vector3.Normalize(Vector3.Cross(up, forward));
            if (right.LengthSquared() < 0.001f)
            {
                // forward is parallel to up — pick an arbitrary right
                right = Vector3.UnitX;
            }
            var correctedUp = Vector3.Cross(forward, right);

            // Build rotation matrix → quaternion
            var m = new Matrix4x4(
                right.X, right.Y, right.Z, 0,
                correctedUp.X, correctedUp.Y, correctedUp.Z, 0,
                forward.X, forward.Y, forward.Z, 0,
                0, 0, 0, 1
            );
            return Quaternion.CreateFromRotationMatrix(m);
        }
    }
}
