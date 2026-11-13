using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Description = System.ComponentModel.DescriptionAttribute;
using Category = System.ComponentModel.CategoryAttribute;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// A lake or river: builds a water surface from the entity's Spline and draws it with the water shader
    /// (water.fx). Closed spline = still water (lake, pond), open spline = flowing water whose current
    /// follows the spline from its first point to its last.
    ///
    /// Setup: Spline + WaterBody. A river usually shares the entity with the HeightStamp that carves its
    /// bed: the spline carries the bed heights and Depth lifts the surface above them.
    ///
    /// The river surface is its own mesh rather than a RuntimeMesh strip: it is subdivided across its width
    /// and carries the current's direction and the steepness per vertex (in the normal stream), so the
    /// shader gets them smoothly instead of per triangle.
    ///
    /// Shading is shared with the ocean (water_common.fx), so all water follows time of day, sky and
    /// clouds the same way. Ripples borrow the ocean simulation's slope maps when the scene has an
    /// OceanRenderer; without one the surface is flat.
    ///
    /// In the cold at its own height it freezes (<see cref="Ice"/>): water.fx draws the ice and the snow on it where the
    /// water is shallow and slow enough (Weather.IceThickness: from the banks in, a fast deep channel stays open), and in
    /// play a static height field over the surface carries walkers on the ice that holds (Weather.WalkableIce): cooked
    /// once as frozen all over, a step of the ice (Weather.IceStep) then only opens and closes holes in it (see Update).
    /// </summary>
    [Icon("icon_ocean.png")]
    public class WaterBody : Component, IDraw, IUpdate
    {
        [Category("Shape")]
        [ValueRange(0.5f, 100f)]
        [Description(@"River: width of the surface in meters where the spline's width is 1.
         Make it reach the banks (about twice the bed stamp's Radius + Falloff); the excess hides under the terrain")]
        public float Width = 12f;

        [ValueRange(0f, 10f)]
        [Description(@"Height of the surface above the spline, i.e. the water depth when the spline carries the bed heights")]
        public float Depth = 0.9f;

        [ValueRange(0f, 1.5f)]
        [Description(@"River: how the depth grows with the spline's width (Spline.Widths): Depth x width ^ this. 0 = the same
         depth everywhere; 1 = in proportion; about 0.7 for a river that deepens as it widens. The spline's Y is then
         the surface less that depth at each point")]
        public float DepthWithWidth = 0f;

        /// <summary>
        /// The water's depth over the spline where the spline's width multiplier is <paramref name="width"/>
        /// (<see cref="Depth"/> scaled by <see cref="DepthWithWidth"/>).
        /// </summary>
        public float DepthAt(float width) =>
            DepthWithWidth > 0f ? Depth * MathF.Pow(MathF.Max(width, 1e-3f), DepthWithWidth) : Depth;

        [ValueRange(-20f, 50f)]
        [Description(@"Lake: grows the surface outward from the spline by this many meters,
         so the water reaches up the shore past the bed outline")]
        public float Expand = 0f;

        [Category("Color")]
        [Description(@"Color of deep water (light scattered back from the water itself)")]
        public Color3 WaterColor = new Color3(0.004f, 0.02f, 0.018f);

        [Description(@"Tint the bed takes on through shallow water")]
        public Color3 ShallowColor = new Color3(0.45f, 0.7f, 0.55f);

        [ValueRange(0.3f, 30f)]
        [Description(@"Underwater visibility in meters.
         At this much water the bed is tinted by ShallowColor and mostly faded out")]
        public float Visibility = 3.5f;

        [ValueRange(0f, 0.1f)]
        [Description(@"How much the ripples bend the view of the bed")]
        public float RefractionStrength = 0.02f;

        [Category("Surface")]
        [ValueRange(0f, 2f)]
        [Description(@"Ripple height. Lakes and slow rivers want far less than the open sea")]
        public float RippleStrength = 0.35f;

        [ValueRange(0.2f, 5f)]
        [Description(@"Ripple size multiplier (1 = the ocean's finest wave bands as they are)")]
        public float RippleSize = 1f;

        [ValueRange(0f, 6f)]
        [Description(@"Current speed in m/s along the spline (open splines only).
         Steep stretches run faster on their own")]
        public float FlowSpeed = 1.2f;

        [Category("Foam")]
        [ValueRange(0f, 1f)]
        [Description(@"Foam along the banks and around anything standing in the water")]
        public float EdgeFoam = 0.4f;

        [ValueRange(0f, 2f)]
        [Description(@"White water on steep stretches (open splines only)")]
        public float RapidsFoam = 1f;

        private Material _material = null!;
        private readonly MaterialBlock _params = new MaterialBlock();
        private Mesh? _mesh;
        private bool _meshDirty = true;
        private Spline? _spline;

        /// <summary>Cross-sections per spline span, and quads across the width, of a river's surface.</summary>
        private const int SegmentsPerSpan = 12;
        private const int Columns = 8;

        /// <summary>
        /// The surface is drawn as parts about this many metres across (a river's run of cross-sections, a lake's grid
        /// cell), each with its own bounds. Water reads the sun's shadow but draws after it, so the shadow receivers mark
        /// every tile under each part's sphere the camera saw (ShadowReceivers CSReceiverSpheres): as one part, a river's
        /// sphere (Greyfell Beck: 210 m) covered the near cascades 320 m away and nothing was culled there (2026-10-23).
        /// </summary>
        private const float ChunkSize = 32f;
        private OceanRenderer? _ocean;
        private DirectionalLight? _sunLight;

        // The surface as built (mesh-local), for the ice collider: its points, how steep the river falls at each, its
        // triangles (a lake's are the outline's: a few big ones)
        private Vector3[] _surfacePoints = Array.Empty<Vector3>();
        private float[] _surfaceSteep = Array.Empty<float>();
        private int[] _surfaceTriangles = Array.Empty<int>();

        // The ice collider's grid: cells of IceCell metres, a narrow river's as wide as its strip's columns, and no more
        // than IceCellsMax a side
        private const float IceCell = 2f, IceCellSmallest = 0.25f;
        private const int IceCellsMax = 2048;

        // Its own ice (Weather.StepIce), the height it takes the cold at (a lake's surface, a river's mean), the weather
        // update it last stepped at
        private float _ice, _iceHeight;
        private int _iceTick = -1;

        // How far each place of the water lies from its shore, for where the ice forms first (Weather.IceThickness): a
        // grid over the surface's bounds in world xz (ShoreCellsMax cells a side at most, none finer than a metre), each
        // cell the distance to the nearest cell the water does not cover (the bed rises over the surface there, or there
        // is no surface) over the farthest any cell lies: 0 at the shore, 1 in the middle of the open water. By the depth
        // instead, a lake whose bed is flat froze everywhere at once, in blotches where the edge's noise tipped it
        // (user, 2026-10-23: "looks roughed in"). Made on a worker when the surface is built; water.fx reads it as a
        // texture (WaterData.ShoreField), the ice collider the array.
        private const int ShoreCellsMax = 512;

        private sealed class ShoreField
        {
            public required float[] Share;
            public int Width, Height;
            public Vector2 Origin;
            public float Cell;

            // Bilinear, 0 outside
            public float At(float x, float z)
            {
                float fx = (x - Origin.X) / Cell - 0.5f, fz = (z - Origin.Y) / Cell - 0.5f;
                int x0 = (int)MathF.Floor(fx), z0 = (int)MathF.Floor(fz);
                float tx = fx - x0, tz = fz - z0;
                float Get(int i, int j) => i < 0 || j < 0 || i >= Width || j >= Height ? 0f : Share[j * Width + i];
                return (Get(x0, z0) * (1f - tx) + Get(x0 + 1, z0) * tx) * (1f - tz)
                     + (Get(x0, z0 + 1) * (1f - tx) + Get(x0 + 1, z0 + 1) * tx) * tz;
            }
        }

        private ShoreField? _shore;
        private System.Threading.Tasks.Task<ShoreField?>? _shoreTask;
        private Texture? _shoreTexture;

        // The ice that can carry a walker over this surface: a height field of the surface in world space, on a grid over
        // its bounds, with for each cell the ice level from which it holds (Weather.IceHoldsFrom). All the state ice adds
        // to it is the one level, so the collider for any level is the same field with other holes, and a client and a
        // server that agree on the level agree on the ice. Made and cooked once on a worker, per surface and shore field.
        private sealed class IceField
        {
            public required float[] HoldsFrom;              // per cell (a sample is its cell's low corner); infinity: never
            public required int[] Samples;                  // PhysicsHelper.PackedSample, as HeightField.ModifySamples takes them
            public required int Rows, Columns;              // of samples: row = x, column = z
            public required Vector3 Corner;                 // sample 0 of the field, at its lowest height
            public required float Cell, HeightScale;
            public required int CellsThatHold;              // the cells that can (all of them are as cooked)
            public System.IO.MemoryStream? Cooked;          // until the main thread has made the height field of it
        }

        // The collider: the field's height field at _iceLevel. It is in the scene only while a cell holds.
        // One water body a frame takes its finished cook (_iceTakeTick)
        private IceField? _iceField;
        private PhysX.HeightField? _iceHeightField;
        private PhysX.RigidStatic? _iceActor;
        private PhysX.Shape? _iceShape;
        private float _iceLevel;
        private int _iceHolding;
        private bool _iceInScene, _iceFieldDirty;
        private System.Threading.Tasks.Task<IceField?>? _iceCook;
        private PhysX.Cooking? _iceCooking;
        private static int _iceTakeTick = -1;

        [StructLayout(LayoutKind.Sequential)]
        public struct WaterData
        {
            public float Time;
            public float FlowSpeed;
            public float RippleStrength;
            public float RippleSize;
            public Vector3 WaterColor;
            public float Visibility;
            public Vector3 ShallowColor;
            public float RefractionStrength;
            public Vector3 SunDirection;
            public float SunIntensity;
            public Vector3 SunColor;
            public float EdgeFoam;
            public float RapidsFoam;
            public uint SlopeSRV;
            public uint NoiseSRV;
            public uint DepthGBufferSRV;
            public uint CompositeSRV;
            public float InvViewportWidth;
            public float InvViewportHeight;
            public float TileScaleA;
            public float TileScaleB;
            public uint BandA;
            public uint BandB;
            public float Flowing;       // 1 = open spline (river), 0 = closed (lake)
            public float Ice;           // WaterBody.Ice
            public float SnowCover;     // Weather.SnowCover: snow lies on the ice
            public uint ShoreField;     // Texture2D<float>: the share of the way from the shore to the middle (0 = none yet)
            public float ShoreOriginX;  // world xz of the field's corner, and the metres it spans
            public float ShoreOriginZ;
            public float ShoreSizeX;
            public float ShoreSizeZ;
            public float SnowAlongX;    // Weather.SnowLaidAlong: the way the snow's drifts on the ice run
            public float SnowAlongZ;
            public uint NearestDepthSRV;    // WaterReflections: the nearest-depth pyramid (0 = no screen-space reflections)
            public uint NearestDepthMips;
            public uint SceneColourSRV;     // WaterReflections: the composite snapshot as a pyramid, half size and down
            public uint SceneColourMips;
        }

        protected override void Awake()
        {
            _material = new Material(Assets.InternalAssets.WaterEffect);
            _spline = Entity.GetComponent<Spline>();
            _sunLight = EntityManager.FindComponent<DirectionalLight>();
            MessageDispatcher.AddListener(EngineMsg.SplineChanged, OnSplineChanged);
        }

        public override void Destroy()
        {
            MessageDispatcher.RemoveListener(EngineMsg.SplineChanged, OnSplineChanged);
            ReleaseIce();
            // A cook still running keeps its cooking object until it is done
            if (_iceCook != null && _iceCooking != null)
            {
                var cooking = _iceCooking;
                _iceCook.ContinueWith(_ => Engine.RunOnMainThreadAsync(() => PhysXRelease.Release(cooking)));
            }
            _iceCook = null;
            _iceCooking = null;
            Engine.Device.DeferDispose(_shoreTexture);
            _shoreTexture = null;

            // Deferred: a draw enqueued earlier this frame and frames still in flight reference the mesh
            Engine.Device.DeferDispose(_mesh);
            _mesh = null;
        }

        private void OnSplineChanged(Message msg)
        {
            if (msg.Data is Spline spline && spline.Entity == Entity)
                _meshDirty = true;
        }

        /// <summary>Inspector / command-server edits (Width, Depth, Expand) rebuild the surface.</summary>
        public override void OnMemberChanged() => _meshDirty = true;

        // ═══════════════════════════
        // ── Ice ──
        // ═══════════════════════════

        /// <summary>
        /// How frozen this water is (0..1): it freezes and thaws by the weather at its own surface's height
        /// (Weather.StepIce), not the camera's; Settings › OverrideWeatherSurfaces / OverrideIce set it by hand.
        /// </summary>
        public float Ice => Weather.ShownIce(_ice);

        /// <summary>
        /// In play: the ice collider follows the ice. Its height field is made and cooked once per surface and shore field,
        /// on a worker, frozen all over (IceField); a step of the ice then only opens and closes the cells it changes
        /// (FollowIce). Until 2026-10-24 every water body walked its surface and cooked its collider on the main thread at
        /// every step of the ice (every 12 s while it freezes): frames of 170-250 ms in Eldermoor. Until 2026-10-30 a
        /// worker cooked a triangle mesh of what holds at every such step.
        /// </summary>
        public void Update()
        {
            if (Engine.IsEditor) return;
            if (_meshDirty) RebuildMesh();
            TakeShoreField();
            if (_shoreTask != null) return;     // the shore field is still being made: the ice waits for it
            if (_iceFieldDirty) StartIceField();
            TakeIceField();
            if (_iceField != null) FollowIce(MathF.Round(Ice / Weather.IceStep) * Weather.IceStep);
        }

        // The ice over the surface just built, made and cooked on a worker (the shore field and the surface arrays are
        // replaced, never written). A cook still running goes first: it holds the cooking object
        private void StartIceField()
        {
            if (_iceCook != null) return;
            _iceFieldDirty = false;
            if (_surfaceTriangles.Length < 3)
            {
                ReleaseIce();
                PhysicsWorld.InvalidateControllerCaches();
                return;
            }
            var (surface, steep, triangles, shore) = (_surfacePoints, _surfaceSteep, _surfaceTriangles, _shore);
            var world = Transform.WorldMatrix;
            bool flowing = _spline != null && !_spline.Closed;
            float flowSpeed = FlowSpeed;
            float cell = flowing ? Math.Clamp(Width / Columns, IceCellSmallest, IceCell) : IceCell;
            var cooking = PhysicsWorld.Physics.CreateCooking();    // main thread: it goes into PhysX.Net's object table
            _iceCooking = cooking;
            _iceCook = System.Threading.Tasks.Task.Run(() => BuildIceField(cooking, surface, steep, triangles, world, flowing, flowSpeed, cell, shore));
        }

        // Main thread: a finished cook becomes the collider, one water body a frame, every cell of it holding
        private void TakeIceField()
        {
            if (_iceCook is not { IsCompleted: true } cook || _iceTakeTick == Engine.TickCount) return;
            _iceTakeTick = Engine.TickCount;
            _iceCook = null;
            PhysXRelease.Release(_iceCooking);
            _iceCooking = null;

            ReleaseIce();
            PhysicsWorld.InvalidateControllerCaches();
            try
            {
                if (cook.GetAwaiter().GetResult() is not { } field) return;
                _iceHeightField = PhysicsWorld.Physics.CreateHeightField(field.Cooked);
                field.Cooked = null;
                _iceActor = PhysicsWorld.Physics.CreateRigidStatic(Matrix4x4.CreateTranslation(field.Corner));
                _iceActor.UserData = Entity;
                _iceActor.Flags &= ~PhysX.ActorFlag.Visualization;
                var geometry = new PhysX.HeightFieldGeometry(_iceHeightField, 0, field.HeightScale, field.Cell, field.Cell);
                _iceShape = PhysX.RigidActorExt.CreateExclusiveShape(_iceActor, geometry, PhysicsWorld.DefaultMaterial);
                _iceShape.QueryFilterData = new PhysX.FilterData(Collider.Solid, 0, 0, 0);
                _iceField = field;
                _iceLevel = 1f;
                _iceHolding = field.CellsThatHold;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("WaterBody", $"'{Entity.Name}': cooking the ice collider failed: {ex.GetBaseException().Message}");
                ReleaseIce();
            }
        }

        /// <summary>
        /// The holes of the height field become those of this ice level. Heights stay as they are, so what PhysX needs to
        /// hear is little (checked in its source and with a capsule, 2026-10-30): scene queries read the samples as they
        /// are, the bounds cannot change, and a body in contact finds the hole by itself while it is awake, so no
        /// PxShape::setGeometry, which would also dirty the static pruner. Only a controller keeps triangles of its own.
        /// A body asleep on the ice stays where it is, with or without setGeometry, until something wakes it.
        /// </summary>
        private void FollowIce(float ice)
        {
            bool changed = ice != _iceLevel && WriteIceLevel(ice);
            bool wanted = _iceHolding > 0;
            if (wanted != _iceInScene)
            {
                _iceInScene = wanted;
                if (wanted) PhysicsWorld.Scene.AddActor(_iceActor!);
                else PhysicsWorld.Scene.RemoveActor(_iceActor!);
                changed = true;
            }
            // The walkers' controllers remember the triangles round them: ice opened or closed is not seen otherwise
            // (they walked through ice turned off and on again)
            if (changed) PhysicsWorld.InvalidateControllerCaches();
        }

        // Walks the cells for those the two levels tell apart and writes them to the height field, a run along a row at a
        // time. A level that changes nothing costs the walk. True when a cell changed.
        private bool WriteIceLevel(float ice)
        {
            var field = _iceField!;
            var (holds, samples, columns) = (field.HoldsFrom, field.Samples, field.Columns);
            float was = _iceLevel;
            _iceLevel = ice;
            bool changed = false;
            for (int row = 0; row < field.Rows - 1; row++)
            {
                int first = row * columns;
                for (int column = 0; column < columns - 1; column++)
                {
                    if (holds[first + column] <= ice == holds[first + column] <= was) continue;
                    int start = column;
                    for (; column < columns - 1 && holds[first + column] <= ice != holds[first + column] <= was; column++)
                    {
                        bool holding = holds[first + column] <= ice;
                        samples[first + column] = PhysicsHelper.PackedSample((short)samples[first + column], holding ? (byte)0 : PhysicsHelper.HoleMaterial);
                        _iceHolding += holding ? 1 : -1;
                    }
                    // The run's samples and the row above's, one further along: a sample's collision flag goes by the four
                    // cells round it, and PhysX works it out only for the samples written
                    int count = column - start + 1;
                    _iceHeightField!.ModifySamples(start, row, count, 1, samples, first + start, false);
                    _iceHeightField.ModifySamples(start, row + 1, count, 1, samples, first + columns + start, false);
                    changed = true;
                }
            }
            return changed;
        }

        /// <summary>
        /// Any thread: the height field of a surface (world space), cooked with every cell holding that ever can. A cell
        /// is ice whose middle lies in a surface triangle, as far out as the ice comes, under the banks too (the surface
        /// runs on under them), so no gap opens between the shore and the ice; a corner of one off the surface carries the
        /// triangle's plane on. Cells that would need more than full ice stay holes. A place where a river overlies itself
        /// has one height. Null when no cell can ever hold.
        /// </summary>
        private static IceField? BuildIceField(PhysX.Cooking cooking, Vector3[] surface, float[] steep, int[] triangles, Matrix4x4 world,
                                               bool flowing, float flowSpeed, float cell, ShoreField? shore)
        {
            var points = Array.ConvertAll(surface, p => Vector3.Transform(p, world));
            float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
            foreach (var p in points)
            {
                minX = MathF.Min(minX, p.X); maxX = MathF.Max(maxX, p.X);
                minY = MathF.Min(minY, p.Y); maxY = MathF.Max(maxY, p.Y);
                minZ = MathF.Min(minZ, p.Z); maxZ = MathF.Max(maxZ, p.Z);
            }
            cell = MathF.Max(cell, MathF.Max(maxX - minX, maxZ - minZ) / IceCellsMax);
            int rows = (int)MathF.Ceiling((maxX - minX) / cell) + 1, columns = (int)MathF.Ceiling((maxZ - minZ) / cell) + 1;
            if (rows < 2 || columns < 2) return null;

            var heights = new float[rows * columns];
            var holds = new float[rows * columns];
            Array.Fill(heights, float.NaN);
            Array.Fill(holds, float.PositiveInfinity);

            // The grid places a triangle can reach, in rows and columns
            (int Row0, int Row1, int Column0, int Column1) Reach(Vector3 a, Vector3 b, Vector3 c) =>
                (Math.Max(0, (int)((MathF.Min(a.X, MathF.Min(b.X, c.X)) - minX) / cell)),
                 Math.Min(rows - 1, (int)((MathF.Max(a.X, MathF.Max(b.X, c.X)) - minX) / cell) + 1),
                 Math.Max(0, (int)((MathF.Min(a.Z, MathF.Min(b.Z, c.Z)) - minZ) / cell)),
                 Math.Min(columns - 1, (int)((MathF.Max(a.Z, MathF.Max(b.Z, c.Z)) - minZ) / cell) + 1));

            // The samples on the surface, then the cells on it
            for (int t = 0; t + 2 < triangles.Length; t += 3)
            {
                var (a, b, c) = (points[triangles[t]], points[triangles[t + 1]], points[triangles[t + 2]]);
                var (row0, row1, column0, column1) = Reach(a, b, c);
                for (int r = row0; r <= row1; r++)
                for (int k = column0; k <= column1; k++)
                    if (Barycentric(new Vector2(minX + r * cell, minZ + k * cell), a, b, c, out var w))
                        heights[r * columns + k] = a.Y * w.X + b.Y * w.Y + c.Y * w.Z;
            }
            int cells = 0;
            for (int t = 0; t + 2 < triangles.Length; t += 3)
            {
                int ia = triangles[t], ib = triangles[t + 1], ic = triangles[t + 2];
                var (a, b, c) = (points[ia], points[ib], points[ic]);
                var (row0, row1, column0, column1) = Reach(a, b, c);
                for (int r = row0; r <= Math.Min(row1, rows - 2); r++)
                for (int k = column0; k <= Math.Min(column1, columns - 2); k++)
                {
                    var middle = new Vector2(minX + (r + 0.5f) * cell, minZ + (k + 0.5f) * cell);
                    if (!Barycentric(middle, a, b, c, out var w)) continue;
                    float speed = flowing ? flowSpeed * (1f + 4f * (steep[ia] * w.X + steep[ib] * w.Y + steep[ic] * w.Z)) : 0f;
                    float from = Weather.IceHoldsFrom(shore?.At(middle.X, middle.Y) ?? 1f, speed);
                    if (from > 1f) continue;
                    int index = r * columns + k;
                    if (float.IsInfinity(holds[index])) cells++;
                    holds[index] = MathF.Min(holds[index], from);
                    for (int corner = 0; corner < 4; corner++)
                    {
                        int cr = r + (corner & 1), ck = k + (corner >> 1);
                        if (!float.IsNaN(heights[cr * columns + ck])) continue;
                        Barycentric(new Vector2(minX + cr * cell, minZ + ck * cell), a, b, c, out var on);
                        heights[cr * columns + ck] = a.Y * on.X + b.Y * on.Y + c.Y * on.Z;
                    }
                }
            }
            if (cells == 0) return null;

            float heightScale = MathF.Max(maxY - minY, 1f) / short.MaxValue;
            var samples = new int[heights.Length];
            var cooked = new PhysX.HeightFieldSample[heights.Length];
            for (int i = 0; i < heights.Length; i++)
            {
                // Corners of no cell lie anywhere: at the lowest
                short height = float.IsNaN(heights[i]) ? (short)0 : (short)Math.Clamp(MathF.Round((heights[i] - minY) / heightScale), 0f, short.MaxValue);
                byte material = float.IsInfinity(holds[i]) ? PhysicsHelper.HoleMaterial : (byte)0;
                samples[i] = PhysicsHelper.PackedSample(height, material);
                cooked[i] = new PhysX.HeightFieldSample
                {
                    Height = height,
                    MaterialIndex0 = new PhysX.BitAndByte(material, true),
                    MaterialIndex1 = new PhysX.BitAndByte(material, false),
                };
            }
            var stream = new System.IO.MemoryStream();
            cooking.CookHeightField(new PhysX.HeightFieldDesc { NumberOfRows = rows, NumberOfColumns = columns, Samples = cooked }, stream);
            stream.Position = 0;
            return new IceField
            {
                HoldsFrom = holds, Samples = samples, Rows = rows, Columns = columns, Corner = new Vector3(minX, minY, minZ),
                Cell = cell, HeightScale = heightScale, CellsThatHold = cells, Cooked = stream,
            };
        }

        /// <summary>The shore field of the surface just built, made on a worker (the terrain's heights are only read).</summary>
        private void StartShoreField()
        {
            _shore = null;
            _shoreTask = null;
            if (_surfaceTriangles.Length < 3) return;
            var world = Transform.WorldMatrix;
            var points = Array.ConvertAll(_surfacePoints, p => Vector3.Transform(p, world));
            var triangles = _surfaceTriangles;
            var terrain = EntityManager.FindComponent<TerrainRenderer>();
            _shoreTask = System.Threading.Tasks.Task.Run(() => MakeShoreField(points, triangles, terrain));
        }

        /// <summary>Main thread: the finished field into its texture (water.fx), and the collider made again over it.</summary>
        private void TakeShoreField()
        {
            if (_shoreTask is not { IsCompleted: true } task) return;
            _shoreTask = null;
            _shore = task.IsCompletedSuccessfully ? task.Result : null;
            Engine.Device.DeferDispose(_shoreTexture);
            _shoreTexture = null;
            if (_shore != null)
            {
                var bytes = new byte[_shore.Share.Length * 4];
                Buffer.BlockCopy(_shore.Share, 0, bytes, 0, bytes.Length);
                _shoreTexture = Texture.CreateFromData(Engine.Device, _shore.Width, _shore.Height, bytes, Vortice.DXGI.Format.R32_Float);
            }
            _iceFieldDirty = true;
        }

        // Any thread. The grid's cells whose centre lies in a surface triangle and over the bed are water, the rest shore;
        // the distance to the shore by a two-pass chamfer (1 and √2), over its largest
        private static ShoreField? MakeShoreField(Vector3[] points, int[] triangles, TerrainRenderer? terrain)
        {
            float minX = float.MaxValue, minZ = float.MaxValue, maxX = float.MinValue, maxZ = float.MinValue;
            foreach (var p in points)
            {
                minX = MathF.Min(minX, p.X); maxX = MathF.Max(maxX, p.X);
                minZ = MathF.Min(minZ, p.Z); maxZ = MathF.Max(maxZ, p.Z);
            }
            float cell = MathF.Max(1f, MathF.Max(maxX - minX, maxZ - minZ) / (ShoreCellsMax - 2));
            var origin = new Vector2(minX - cell, minZ - cell);
            int w = (int)MathF.Ceiling((maxX - minX) / cell) + 2, h = (int)MathF.Ceiling((maxZ - minZ) / cell) + 2;
            var d = new float[w * h];

            for (int t = 0; t + 2 < triangles.Length; t += 3)
            {
                var a = points[triangles[t]];
                var b = points[triangles[t + 1]];
                var c = points[triangles[t + 2]];
                int x0 = Math.Max(0, (int)((MathF.Min(a.X, MathF.Min(b.X, c.X)) - origin.X) / cell));
                int x1 = Math.Min(w - 1, (int)((MathF.Max(a.X, MathF.Max(b.X, c.X)) - origin.X) / cell));
                int z0 = Math.Max(0, (int)((MathF.Min(a.Z, MathF.Min(b.Z, c.Z)) - origin.Y) / cell));
                int z1 = Math.Min(h - 1, (int)((MathF.Max(a.Z, MathF.Max(b.Z, c.Z)) - origin.Y) / cell));
                for (int j = z0; j <= z1; j++)
                for (int i = x0; i <= x1; i++)
                {
                    var centre = origin + new Vector2(i + 0.5f, j + 0.5f) * cell;
                    if (!Barycentric(centre, a, b, c, out var weight)) continue;
                    float surface = a.Y * weight.X + b.Y * weight.Y + c.Y * weight.Z;
                    float bed = terrain?.GetHeight(new Vector3(centre.X, surface, centre.Y)) ?? surface - 10f;
                    if (bed < surface) d[j * w + i] = float.MaxValue;
                }
            }

            const float Diagonal = 1.41421356f;
            float At(int i, int j) => i < 0 || j < 0 || i >= w || j >= h ? 0f : d[j * w + i];
            for (int j = 0; j < h; j++)
            for (int i = 0; i < w; i++)
            {
                ref float v = ref d[j * w + i];
                if (v == 0f) continue;
                v = MathF.Min(v, MathF.Min(MathF.Min(At(i - 1, j) + 1f, At(i, j - 1) + 1f),
                                           MathF.Min(At(i - 1, j - 1) + Diagonal, At(i + 1, j - 1) + Diagonal)));
            }
            float most = 0f;
            for (int j = h - 1; j >= 0; j--)
            for (int i = w - 1; i >= 0; i--)
            {
                ref float v = ref d[j * w + i];
                if (v == 0f) continue;
                v = MathF.Min(v, MathF.Min(MathF.Min(At(i + 1, j) + 1f, At(i, j + 1) + 1f),
                                           MathF.Min(At(i + 1, j + 1) + Diagonal, At(i - 1, j + 1) + Diagonal)));
                most = MathF.Max(most, v);
            }
            if (most <= 0f) return null;
            for (int k = 0; k < d.Length; k++)
                d[k] /= most;
            return new ShoreField { Share = d, Width = w, Height = h, Origin = origin, Cell = cell };
        }

        // The weights of a point (xz) in a triangle, false when it lies outside
        private static bool Barycentric(Vector2 p, Vector3 a, Vector3 b, Vector3 c, out Vector3 w)
        {
            var v0 = new Vector2(b.X - a.X, b.Z - a.Z);
            var v1 = new Vector2(c.X - a.X, c.Z - a.Z);
            var v2 = new Vector2(p.X - a.X, p.Y - a.Z);
            float d = v0.X * v1.Y - v1.X * v0.Y;
            w = default;
            if (MathF.Abs(d) < 1e-8f) return false;
            float u = (v2.X * v1.Y - v1.X * v2.Y) / d;
            float v = (v0.X * v2.Y - v2.X * v0.Y) / d;
            w = new Vector3(1f - u - v, u, v);
            return u >= 0f && v >= 0f && u + v <= 1f;
        }

        private void ReleaseIce()
        {
            if (_iceActor != null)
            {
                if (_iceInScene) PhysicsWorld.Scene.RemoveActor(_iceActor);
                _iceInScene = false;
                // The shape goes with its actor natively, but its wrapper stays in PhysX.Net's object table unless let go
                // first: the next field's shape, made at the same address, then threw ("already a managed instance")
                PhysXRelease.Release(_iceShape, releaseNative: false);
                _iceShape = null;
                PhysXRelease.Release(_iceActor);
                _iceActor = null;
            }
            if (_iceHeightField != null)
            {
                PhysXRelease.Release(_iceHeightField);
                _iceHeightField = null;
            }
            _iceField = null;
        }

        // ═══════════════════════════
        // ── Surface mesh ──
        // ═══════════════════════════

        private void RebuildMesh()
        {
            _meshDirty = false;
            _iceFieldDirty = true;   // over the new surface
            Engine.Device.DeferDispose(_mesh);
            _mesh = null;
            _surfacePoints = Array.Empty<Vector3>();
            _surfaceSteep = Array.Empty<float>();
            _surfaceTriangles = Array.Empty<int>();

            if (_spline == null || _spline.Points.Count < 2) return;
            _mesh = _spline.Closed ? BuildLake(_spline) : BuildRiver(_spline);
            var world = Transform.WorldMatrix;
            float heights = 0f;
            foreach (var p in _surfacePoints) heights += Vector3.Transform(p, world).Y;
            _iceHeight = _surfacePoints.Length > 0 ? heights / _surfacePoints.Length : Transform.WorldPosition.Y;
            StartShoreField();
        }

        /// <summary>
        /// A grid along the spline. Positions and UVs (x = meters along, y = meters across) as a strip; the
        /// normal stream holds what the shader needs instead of a normal: xz = direction of the current,
        /// y = how steeply the surface falls there (drop per meter).
        /// </summary>
        private Mesh BuildRiver(Spline spline)
        {
            var strip = SplineStrip.Sample(spline, SegmentsPerSpan, Width);
            int n = strip.Count;

            var verts = new List<Vector3>(n * (Columns + 1));
            var flow = new List<Vector3>(n * (Columns + 1));
            var uvs = new List<Vector2>(n * (Columns + 1));
            var indices = new List<uint>((n - 1) * Columns * 6);

            // The water's depth at each cross-section: the same everywhere, or growing with the spline's width there
            // (DepthWithWidth), so a river can deepen as it widens while its surface keeps falling
            var depths = new float[n];
            for (int i = 0; i < n; i++)
                depths[i] = DepthAt(strip.HalfWidths[i] * 2f / MathF.Max(Width, 1e-3f));

            for (int i = 0; i < n; i++)
            {
                // Fall of the surface's centre line across the two neighbouring cross-sections (not the bed's: where the
                // depth changes along the river they differ)
                int a = Math.Max(0, i - 1), b = Math.Min(n - 1, i + 1);
                var run = strip.Points[b] - strip.Points[a];
                run.Y += depths[b] - depths[a];
                float ground = MathF.Sqrt(run.X * run.X + run.Z * run.Z);
                float steep = ground > 1e-4f ? MathF.Abs(run.Y) / ground : 0f;

                var fwd = strip.Tangents[i];
                var dir = new Vector2(fwd.X, fwd.Z);
                dir = dir.LengthSquared() > 1e-12f ? Vector2.Normalize(dir) : Vector2.UnitX;

                var left = strip.Left(i);
                var right = strip.Right(i);
                float across = strip.HalfWidths[i] * 2f;

                for (int c = 0; c <= Columns; c++)
                {
                    float f = (float)c / Columns;
                    var p = Vector3.Lerp(left, right, f);
                    p.Y += depths[i];
                    verts.Add(p);
                    flow.Add(new Vector3(dir.X, steep, dir.Y));
                    uvs.Add(new Vector2(strip.ArcLengths[i], f * across));
                }
            }

            // A part per ChunkSize metres along the river (the parts share the cross-section where they meet)
            var parts = new List<MeshPart>();
            int partStart = 0;
            float partFrom = 0f;
            for (int i = 0; i < n - 1; i++)
            {
                if (strip.ArcLengths[i] - partFrom >= ChunkSize && indices.Count > partStart)
                {
                    parts.Add(SurfacePart(partStart, indices.Count - partStart));
                    partStart = indices.Count;
                    partFrom = strip.ArcLengths[i];
                }

                for (int c = 0; c < Columns; c++)
                {
                    uint bl = (uint)(i * (Columns + 1) + c);
                    uint br = bl + 1;
                    uint tl = bl + (uint)(Columns + 1);
                    uint tr = tl + 1;

                    indices.Add(bl); indices.Add(br); indices.Add(tl);
                    indices.Add(br); indices.Add(tr); indices.Add(tl);
                }
            }
            if (indices.Count > partStart)
                parts.Add(SurfacePart(partStart, indices.Count - partStart));

            _surfacePoints = verts.ToArray();
            _surfaceSteep = new float[flow.Count];
            for (int i = 0; i < flow.Count; i++) _surfaceSteep[i] = flow[i].Y;
            _surfaceTriangles = ToInts(indices);
            return BuildSurface(verts, flow, uvs, indices, parts);
        }

        /// <summary>The spline's outline, grown by Expand, filled and lifted by Depth.</summary>
        private Mesh BuildLake(Spline spline)
        {
            int samples = Math.Max(3, spline.SpanCount * SegmentsPerSpan);
            var polygon = new List<Vector2>(samples);
            var heights = new float[samples];
            for (int i = 0; i < samples; i++)
            {
                var p = spline.GetPoint((float)i / samples);   // no end point: the outline wraps
                polygon.Add(new Vector2(p.X, p.Z));
                heights[i] = p.Y;
            }

            // Counter-clockwise, as the triangulation expects
            if (RuntimeMesh.GetSignedArea(polygon) > 0)
            {
                polygon.Reverse();
                Array.Reverse(heights);
            }

            // Grown by Expand along the vertex normals, an outline with arms close together folds over itself where they
            // pass (a neck narrower than twice Expand), and the ear clipping stops there: the lake drew with a fan of
            // triangles missing (TestWorld's Lake 08 after it was given arms, 2026-11-09: 61 of 66). So the outline is
            // grown less until it clips whole: half, a quarter, then not at all.
            var outline = polygon;
            List<int> triangles = null!;
            foreach (float share in new[] { 1f, 0.5f, 0.25f, 0f })
            {
                float grow = Expand * share;
                if (share < 1f && MathF.Abs(Expand) <= 0.001f) break;
                polygon = MathF.Abs(grow) > 0.001f ? Grown(outline, grow) : outline;
                triangles = RuntimeMesh.EarClipTriangulate(polygon);
                if (triangles.Count == 3 * (samples - 2)) break;
                if (share > 0f)
                    Debug.LogWarning("WaterBody", $"'{Entity.Name}': its outline grown {grow:0.#} m does not triangulate whole " +
                                                  $"({triangles.Count / 3} of {samples - 2} triangles); growing it less");
            }
            if (triangles == null || triangles.Count < 3) return null!;

            var verts = new List<Vector3>(samples);
            var normals = new List<Vector3>(samples);
            var uvs = new List<Vector2>(samples);
            var indices = new List<uint>(triangles.Count);
            for (int i = 0; i < samples; i++)
            {
                verts.Add(new Vector3(polygon[i].X, heights[i] + Depth, polygon[i].Y));
                normals.Add(Vector3.UnitY);
                uvs.Add(polygon[i]);
            }
            for (int i = triangles.Count - 1; i >= 0; i--)   // reversed: faces up
                indices.Add((uint)triangles[i]);

            // The ice and the shore field take the outline's triangles as they are
            _surfacePoints = verts.ToArray();
            _surfaceSteep = new float[verts.Count];
            _surfaceTriangles = ToInts(indices);

            // A level outline (a lake's spline points all at the entity's height): the ear clipping cut by the chunk grid
            float low = float.MaxValue, high = float.MinValue;
            foreach (float h in heights) { low = MathF.Min(low, h); high = MathF.Max(high, h); }
            if (high - low <= 1e-4f)
                return BuildLevelSurface(polygon, triangles, low + Depth);

            // Else the drawn surface: the ear clipping's slivers reach across the whole lake, so split them into triangles no
            // edge of which is longer than half a chunk (the same planes, so the same surface), then one part per grid cell:
            // the cells in the order their first triangle comes, each cell's triangles in their order. Counted first and
            // then placed, in one list (a list per cell grew a second copy of the indices, ~1 GB per TestWorld load with
            // the rest of the surfaces' garbage: ram.md)
            SplitLongEdges(verts, normals, uvs, indices, ChunkSize * 0.5f);
            var cellOf = new Dictionary<(int, int), int>();
            var cellSizes = new List<int>();
            for (int t = 0; t + 2 < indices.Count; t += 3)
            {
                var cell = CellOf(verts, indices, t);
                if (!cellOf.TryGetValue(cell, out int slot))
                {
                    cellOf[cell] = slot = cellSizes.Count;
                    cellSizes.Add(0);
                }
                cellSizes[slot] += 3;
            }
            var parts = new List<MeshPart>(cellSizes.Count);
            var cellAt = new int[cellSizes.Count];
            for (int s = 0, at = 0; s < cellSizes.Count; s++)
            {
                parts.Add(SurfacePart(at, cellSizes[s]));
                cellAt[s] = at;
                at += cellSizes[s];
            }
            var grouped = new List<uint>(indices.Count);
            CollectionsMarshal.SetCount(grouped, indices.Count);
            var into = CollectionsMarshal.AsSpan(grouped);
            for (int t = 0; t + 2 < indices.Count; t += 3)
            {
                int slot = cellOf[CellOf(verts, indices, t)];
                int at = cellAt[slot];
                into[at] = indices[t]; into[at + 1] = indices[t + 1]; into[at + 2] = indices[t + 2];
                cellAt[slot] = at + 3;
            }
            return BuildSurface(verts, normals, uvs, grouped, parts);
        }

        /// <summary>
        /// A level lake's drawn surface: each triangle of the outline's ear clipping cut by the chunk grid into convex
        /// pieces, one part per cell, so no part reaches past its cell. Every vertex is an outline point, the point where an
        /// edge of the ear clipping crosses a grid line, or a grid corner, made once by what it is and shared by every
        /// piece that has it: neighbouring pieces meet without T-junctions, across cells and across triangles. A cell the
        /// outline does not touch is drawn whole instead (two triangles, or a fan from its centre over the points it shares
        /// with a shore cell's pieces). Splitting the slivers in halves until no edge was longer than half a chunk
        /// (SplitLongEdges, still used for an outline that is not level) made TestWorld's Moon Lake 359,204 triangles for
        /// 0.9 km² of flat water, cutting them by the cells 35,312 (ram.md).
        /// </summary>
        private static Mesh BuildLevelSurface(List<Vector2> polygon, List<int> triangles, float y)
        {
            var verts = new List<Vector3>();
            var uvs = new List<Vector2>();
            var made = new Dictionary<VertexOf, uint>();
            var byCell = new Dictionary<(int, int), List<uint>>();

            // Where a vertex lies, from what it is: the same arithmetic wherever it comes up
            Vector2 Position(VertexOf v)
            {
                switch (v.Kind)
                {
                    case VertexKind.Outline: return polygon[v.A];
                    case VertexKind.CutX:
                    {
                        Vector2 a = polygon[v.A], b = polygon[v.B];
                        float x = v.C * ChunkSize;
                        return new Vector2(x, a.Y + (b.Y - a.Y) * ((x - a.X) / (b.X - a.X)));
                    }
                    case VertexKind.CutZ:
                    {
                        Vector2 a = polygon[v.A], b = polygon[v.B];
                        float z = v.C * ChunkSize;
                        return new Vector2(a.X + (b.X - a.X) * ((z - a.Y) / (b.Y - a.Y)), z);
                    }
                    case VertexKind.Centre: return new Vector2((v.A + 0.5f) * ChunkSize, (v.B + 0.5f) * ChunkSize);
                    default: return new Vector2(v.A * ChunkSize, v.B * ChunkSize);
                }
            }
            uint Index(VertexOf v)
            {
                if (made.TryGetValue(v, out uint i)) return i;
                i = (uint)verts.Count;
                var p = Position(v);
                verts.Add(new Vector3(p.X, y, p.Y));
                uvs.Add(p);
                made[v] = i;
                return i;
            }

            Func<VertexOf, Vector2> position = Position;
            var piece = new List<(VertexOf At, Carrier Along)>();
            var clipped = new List<(VertexOf At, Carrier Along)>();
            var pieceAt = new List<VertexOf>();                                      // every piece's points, one after the other
            var cellPieces = new Dictionary<(int, int), List<(int Start, int Count)>>();
            for (int t = 0; t + 2 < triangles.Count; t += 3)
            {
                int a = triangles[t], b = triangles[t + 1], c = triangles[t + 2];   // counter-clockwise in (x, z)
                Vector2 pa = polygon[a], pb = polygon[b], pc = polygon[c];
                var min = Vector2.Min(pa, Vector2.Min(pb, pc));
                var max = Vector2.Max(pa, Vector2.Max(pb, pc));
                int x0 = (int)MathF.Floor(min.X / ChunkSize), x1 = (int)MathF.Floor(max.X / ChunkSize);
                int z0 = (int)MathF.Floor(min.Y / ChunkSize), z1 = (int)MathF.Floor(max.Y / ChunkSize);
                for (int cz = z0; cz <= z1; cz++)
                for (int cx = x0; cx <= x1; cx++)
                {
                    piece.Clear();
                    piece.Add((VertexOf.Outline(a), Carrier.Edge(a, b)));
                    piece.Add((VertexOf.Outline(b), Carrier.Edge(b, c)));
                    piece.Add((VertexOf.Outline(c), Carrier.Edge(c, a)));
                    Clip(piece, clipped, Carrier.LineX(cx), +1f, position);
                    Clip(clipped, piece, Carrier.LineX(cx + 1), -1f, position);
                    Clip(piece, clipped, Carrier.LineZ(cz), +1f, position);
                    Clip(clipped, piece, Carrier.LineZ(cz + 1), -1f, position);
                    if (piece.Count < 3) continue;

                    // Kept until every piece is in: only then is it known whether its cell is drawn by its pieces or whole
                    int start = pieceAt.Count;
                    foreach (var (at, _) in piece) pieceAt.Add(at);
                    if (PieceArea(pieceAt, start, piece.Count, position) <= 1e-6f)
                    {
                        pieceAt.RemoveRange(start, piece.Count);   // a sliver of nothing where a corner lies on a grid line
                        continue;
                    }
                    if (!cellPieces.TryGetValue((cx, cz), out var ranges))
                        cellPieces[(cx, cz)] = ranges = new List<(int Start, int Count)>();
                    ranges.Add((start, piece.Count));
                }
            }

            // The cells the outline touches (each segment's box, a millimetre wider, so a point on a grid line marks both
            // sides) keep their pieces. A cell with pieces the outline does not touch lies wholly inside the lake: drawn
            // whole, two triangles, where its pieces were a share of every sliver that crosses it
            var shore = new HashSet<(int, int)>();
            for (int i = 0; i < polygon.Count; i++)
            {
                Vector2 p = polygon[i], q = polygon[(i + 1) % polygon.Count];
                int sx0 = (int)MathF.Floor((MathF.Min(p.X, q.X) - 1e-3f) / ChunkSize), sx1 = (int)MathF.Floor((MathF.Max(p.X, q.X) + 1e-3f) / ChunkSize);
                int sz0 = (int)MathF.Floor((MathF.Min(p.Y, q.Y) - 1e-3f) / ChunkSize), sz1 = (int)MathF.Floor((MathF.Max(p.Y, q.Y) + 1e-3f) / ChunkSize);
                for (int z = sz0; z <= sz1; z++)
                for (int x = sx0; x <= sx1; x++)
                    shore.Add((x, z));
            }
            bool Whole((int, int) cell) => cellPieces.ContainsKey(cell) && !shore.Contains(cell);

            foreach (var (cell, ranges) in cellPieces)
            {
                var list = new List<uint>();
                if (Whole(cell))
                    WholeCell(cell, ranges, list);
                else
                {
                    // Each piece a fan, wound as the old surface was (the ear clipping's reversed: faces up)
                    foreach (var (start, count) in ranges)
                    {
                        var p0 = Position(pieceAt[start]);
                        for (int k = 1; k + 1 < count; k++)
                        {
                            var p1 = Position(pieceAt[start + k]);
                            var p2 = Position(pieceAt[start + k + 1]);
                            if ((p1.X - p0.X) * (p2.Y - p0.Y) - (p1.Y - p0.Y) * (p2.X - p0.X) <= 1e-6f) continue;
                            list.Add(Index(pieceAt[start]));
                            list.Add(Index(pieceAt[start + k + 1]));
                            list.Add(Index(pieceAt[start + k]));
                        }
                    }
                }
                if (list.Count > 0) byCell[cell] = list;
            }
            if (verts.Count == 0) return null!;

            // A whole cell: its corners, and on each side it shares with a shore cell the points that cell's pieces have
            // there (on a side shared with another whole cell neither keeps them). With none, two triangles; with some, a
            // fan from the cell's centre (a fan from a corner would leave out the points on that corner's own sides)
            void WholeCell((int X, int Z) cell, List<(int Start, int Count)> ranges, List<uint> list)
            {
                var (cx, cz) = cell;
                bool keepLeft = !Whole((cx - 1, cz)), keepRight = !Whole((cx + 1, cz));
                bool keepBottom = !Whole((cx, cz - 1)), keepTop = !Whole((cx, cz + 1));
                var left = new HashSet<VertexOf>(); var right = new HashSet<VertexOf>();
                var bottom = new HashSet<VertexOf>(); var top = new HashSet<VertexOf>();
                foreach (var (start, count) in ranges)
                    for (int k = 0; k < count; k++)
                    {
                        var v = pieceAt[start + k];
                        if (v.Kind == VertexKind.CutX)
                        {
                            if (v.C == cx && keepLeft) left.Add(v);
                            else if (v.C == cx + 1 && keepRight) right.Add(v);
                        }
                        else if (v.Kind == VertexKind.CutZ)
                        {
                            if (v.C == cz && keepBottom) bottom.Add(v);
                            else if (v.C == cz + 1 && keepTop) top.Add(v);
                        }
                    }

                // Counter-clockwise in (x, z) round the cell, as the pieces were
                var loop = new List<VertexOf> { VertexOf.Corner(cx, cz) };
                loop.AddRange(bottom.OrderBy(v => Position(v).X));
                loop.Add(VertexOf.Corner(cx + 1, cz));
                loop.AddRange(right.OrderBy(v => Position(v).Y));
                loop.Add(VertexOf.Corner(cx + 1, cz + 1));
                loop.AddRange(top.OrderByDescending(v => Position(v).X));
                loop.Add(VertexOf.Corner(cx, cz + 1));
                loop.AddRange(left.OrderByDescending(v => Position(v).Y));

                if (loop.Count == 4)
                {
                    list.Add(Index(loop[0])); list.Add(Index(loop[2])); list.Add(Index(loop[1]));
                    list.Add(Index(loop[0])); list.Add(Index(loop[3])); list.Add(Index(loop[2]));
                    return;
                }
                var centre = new VertexOf(VertexKind.Centre, cx, cz, 0);
                var pc = Position(centre);
                for (int k = 0; k < loop.Count; k++)
                {
                    var a = loop[k];
                    var b = loop[(k + 1) % loop.Count];
                    Vector2 pa = Position(a), pb = Position(b);
                    if ((pa.X - pc.X) * (pb.Y - pc.Y) - (pa.Y - pc.Y) * (pb.X - pc.X) <= 1e-6f) continue;   // two points at one place
                    list.Add(Index(centre)); list.Add(Index(b)); list.Add(Index(a));
                }
            }

            // One part per cell, row by row
            var indices = new List<uint>();
            var parts = new List<MeshPart>(byCell.Count);
            foreach (var cell in byCell.Keys.OrderBy(k => k.Item2).ThenBy(k => k.Item1))
            {
                var list = byCell[cell];
                parts.Add(SurfacePart(indices.Count, list.Count));
                indices.AddRange(list);
            }
            var normals = new List<Vector3>(verts.Count);
            for (int i = 0; i < verts.Count; i++) normals.Add(Vector3.UnitY);
            return BuildSurface(verts, normals, uvs, indices, parts);
        }

        // Twice a convex piece's area, counter-clockwise positive
        private static float PieceArea(List<VertexOf> at, int start, int count, Func<VertexOf, Vector2> position)
        {
            var p0 = position(at[start]);
            float sum = 0f;
            for (int k = 1; k + 1 < count; k++)
            {
                var p1 = position(at[start + k]);
                var p2 = position(at[start + k + 1]);
                sum += (p1.X - p0.X) * (p2.Y - p0.Y) - (p1.Y - p0.Y) * (p2.X - p0.X);
            }
            return sum;
        }

        private enum VertexKind : byte { Outline, CutX, CutZ, Corner, Centre }

        /// <summary>A vertex of a level lake's surface by what it is (BuildLevelSurface): its identity, so it is made once.</summary>
        private readonly record struct VertexOf(VertexKind Kind, int A, int B, int C)
        {
            public static VertexOf Outline(int p) => new(VertexKind.Outline, p, 0, 0);
            public static VertexOf Corner(int x, int z) => new(VertexKind.Corner, x, z, 0);
        }

        private enum CarrierKind : byte { Edge, LineX, LineZ }

        /// <summary>What an edge of a piece lies along: an edge of the ear clipping (A &lt; B) or the grid line x or z = A·ChunkSize.</summary>
        private readonly record struct Carrier(CarrierKind Kind, int A, int B)
        {
            public static Carrier Edge(int a, int b) => new(CarrierKind.Edge, Math.Min(a, b), Math.Max(a, b));
            public static Carrier LineX(int x) => new(CarrierKind.LineX, x, 0);
            public static Carrier LineZ(int z) => new(CarrierKind.LineZ, z, 0);
        }

        // Sutherland–Hodgman against one grid line, keeping what each new vertex is and what each edge lies along: a cut
        // of an ear clipping's edge is that edge's crossing, a cut of a grid line is a grid corner. sign +1 keeps the side
        // of larger x (z), -1 the side of smaller; a point on the line is kept on both sides
        private static void Clip(List<(VertexOf At, Carrier Along)> src, List<(VertexOf At, Carrier Along)> dst,
            Carrier line, float sign, Func<VertexOf, Vector2> position)
        {
            dst.Clear();
            int n = src.Count;
            if (n < 3) return;
            float at = line.A * ChunkSize;
            float Side(VertexOf v)
            {
                var p = position(v);
                return sign * ((line.Kind == CarrierKind.LineX ? p.X : p.Y) - at);
            }
            VertexOf Cut(Carrier along) => along.Kind switch
            {
                CarrierKind.Edge => new VertexOf(line.Kind == CarrierKind.LineX ? VertexKind.CutX : VertexKind.CutZ, along.A, along.B, line.A),
                CarrierKind.LineX => VertexOf.Corner(along.A, line.A),
                _ => VertexOf.Corner(line.A, along.A),
            };
            for (int i = 0; i < n; i++)
            {
                var (from, along) = src[i];
                var to = src[(i + 1) % n].At;
                float d0 = Side(from), d1 = Side(to);
                if (d0 >= 0f)
                {
                    if (d1 >= 0f) dst.Add((from, along));
                    else if (d0 == 0f) dst.Add((from, line));
                    else
                    {
                        dst.Add((from, along));
                        dst.Add((Cut(along), line));
                    }
                }
                else if (d1 > 0f) dst.Add((Cut(along), along));
            }
        }

        // The chunk-sized grid cell a triangle's centre lies in
        private static (int, int) CellOf(List<Vector3> verts, List<uint> indices, int t)
        {
            var centre = (verts[(int)indices[t]] + verts[(int)indices[t + 1]] + verts[(int)indices[t + 2]]) / 3f;
            return ((int)MathF.Floor(centre.X / ChunkSize), (int)MathF.Floor(centre.Z / ChunkSize));
        }

        private static int[] ToInts(List<uint> indices)
        {
            var ints = new int[indices.Count];
            for (int i = 0; i < ints.Length; i++) ints[i] = (int)indices[i];
            return ints;
        }

        /// <summary>A counter-clockwise outline moved <paramref name="by"/> metres outward along its vertex normals.</summary>
        private static List<Vector2> Grown(List<Vector2> polygon, float by)
        {
            int n = polygon.Count;
            var grown = new List<Vector2>(n);
            for (int i = 0; i < n; i++)
            {
                var e0 = polygon[i] - polygon[(i - 1 + n) % n];
                var e1 = polygon[(i + 1) % n] - polygon[i];
                var outward = new Vector2(e0.Y, -e0.X) + new Vector2(e1.Y, -e1.X);
                grown.Add(outward.LengthSquared() > 1e-10f ? polygon[i] + Vector2.Normalize(outward) * by : polygon[i]);
            }
            return grown;
        }

        private static MeshPart SurfacePart(int baseIndex, int count) =>
            new MeshPart { Name = "Surface", BaseIndex = baseIndex, NumIndices = count, MaterialSlot = 0 };

        private static Mesh BuildSurface(List<Vector3> verts, List<Vector3> normals, List<Vector2> uvs, List<uint> indices,
            List<MeshPart> parts)
        {
            // water.fx displaces nothing: no push normals
            return RuntimeMesh.BuildMesh(verts, normals, uvs, indices, parts, "WaterBody_", boundsPerPart: true, pushNormals: false);
        }

        // Splits every edge longer than maxEdge at its middle until none is, each edge once for both its triangles (so
        // no cracks open between neighbours): a triangle with one, two or three split edges becomes two, three or four.
        // The new points lie on the old triangles' planes, their normals and UVs the middles of the ends'.
        private static void SplitLongEdges(List<Vector3> verts, List<Vector3> normals, List<Vector2> uvs, List<uint> indices,
            float maxEdge)
        {
            float maxSq = maxEdge * maxEdge;
            // One table of middles and one list of the next pass's triangles for all passes (each pass made both anew,
            // and every split triangle an array of its new indices)
            var middles = new Dictionary<(uint, uint), uint>();
            var next = new List<uint>(indices.Count * 2);
            uint Middle(uint a, uint b)
            {
                var key = a < b ? (a, b) : (b, a);
                if (Vector3.DistanceSquared(verts[(int)a], verts[(int)b]) <= maxSq) return uint.MaxValue;
                if (!middles.TryGetValue(key, out uint m))
                {
                    m = (uint)verts.Count;
                    verts.Add((verts[(int)a] + verts[(int)b]) * 0.5f);
                    normals.Add(Vector3.Normalize(normals[(int)a] + normals[(int)b]));
                    uvs.Add((uvs[(int)a] + uvs[(int)b]) * 0.5f);
                    middles[key] = m;
                }
                return m;
            }
            void Tri(uint a, uint b, uint c) { next.Add(a); next.Add(b); next.Add(c); }

            for (int pass = 0; pass < 16; pass++)
            {
                middles.Clear();
                next.Clear();
                for (int t = 0; t + 2 < indices.Count; t += 3)
                {
                    uint a = indices[t], b = indices[t + 1], c = indices[t + 2];
                    uint ab = Middle(a, b), bc = Middle(b, c), ca = Middle(c, a);
                    int split = (ab != uint.MaxValue ? 1 : 0) + (bc != uint.MaxValue ? 1 : 0) + (ca != uint.MaxValue ? 1 : 0);
                    if (split == 0) { Tri(a, b, c); continue; }
                    if (split == 3)
                    {
                        Tri(a, ab, ca); Tri(ab, b, bc); Tri(ca, bc, c); Tri(ab, bc, ca);
                        continue;
                    }
                    // One or two split edges: turn the triangle so the first split edge is a-b, then fan from its middle
                    while (ab == uint.MaxValue)
                    {
                        (a, b, c) = (b, c, a);
                        (ab, bc, ca) = (bc, ca, ab);
                    }
                    if (split == 1)
                    {
                        Tri(a, ab, c); Tri(ab, b, c);
                    }
                    else if (bc != uint.MaxValue)
                    {
                        Tri(a, ab, c); Tri(ab, bc, c); Tri(ab, b, bc);
                    }
                    else
                    {
                        Tri(a, ab, ca); Tri(ab, b, c); Tri(ab, c, ca);
                    }
                }
                bool changed = middles.Count > 0;
                indices.Clear();
                indices.AddRange(next);
                if (!changed) break;
            }
        }

        public void Draw()
        {
            if (!Enabled) return;
            if (_material == null) return;

            // Lazy lookups: the ocean (ripple textures) may be created after this component
            _ocean ??= EntityManager.FindComponent<OceanRenderer>();
            _sunLight ??= EntityManager.FindComponent<DirectionalLight>();
            _spline ??= Entity.GetComponent<Spline>();

            if (_meshDirty) RebuildMesh();
            TakeShoreField();
            var mesh = _mesh;
            if (mesh == null) return;

            // The ice by the weather at this water's height, once per weather update
            if (_iceTick != Weather.UpdatedTick)
            {
                _iceTick = Weather.UpdatedTick;
                _ice = Weather.StepIce(_ice, _iceHeight);
            }

            var sunDir = Vector3.UnitY;
            var sunColor = Vector3.One;
            float sunIntensity = 1.0f;
            if (_sunLight != null)
            {
                sunDir = Vector3.Transform(Vector3.UnitZ, _sunLight.Entity.Transform.Rotation);
                sunColor = new Vector3(_sunLight.Color.R, _sunLight.Color.G, _sunLight.Color.B);
                sunIntensity = _sunLight.Intensity;
            }

            // Ripples: the two finest bands of the ocean simulation
            var fft = _ocean?.FFT;
            int bandCount = _ocean?.Bands.Count ?? 0;
            uint bandA = 0, bandB = 0;
            float tileA = 0f, tileB = 0f;
            if (fft != null && bandCount > 0)
            {
                bandB = (uint)(bandCount - 1);
                bandA = (uint)Math.Max(0, bandCount - 2);
                tileA = 1.0f / _ocean!.Bands[(int)bandA].LengthScale;
                tileB = 1.0f / _ocean.Bands[(int)bandB].LengthScale;
            }

            var depth = DeferredRenderer.Current?.DepthGBuffer;
            // The scene in the water (water.fx WaterScreenReflection): the renderer makes what it reads on frames a water
            // asks; none on the first (the pyramids come with the next frame) or with the setting off
            WaterReflections? reflections = null;
            if (Engine.Settings.ScreenSpaceReflections)
            {
                WaterReflections.Want();
                reflections = DeferredRenderer.Current?.WaterReflections;
            }

            _params.SetParameter("WaterData", new WaterData
            {
                Time = Time.TotalTime,
                FlowSpeed = FlowSpeed,
                RippleStrength = RippleStrength,
                RippleSize = RippleSize,
                WaterColor = new Vector3(WaterColor.R, WaterColor.G, WaterColor.B),
                Visibility = Visibility,
                ShallowColor = new Vector3(ShallowColor.R, ShallowColor.G, ShallowColor.B),
                RefractionStrength = RefractionStrength,
                SunDirection = sunDir,
                SunIntensity = sunIntensity,
                SunColor = sunColor,
                EdgeFoam = EdgeFoam,
                RapidsFoam = RapidsFoam,
                SlopeSRV = fft?.SlopeSRV ?? 0,
                NoiseSRV = fft?.NoiseSRV ?? 0,
                DepthGBufferSRV = depth?.BindlessIndex ?? 0,
                CompositeSRV = DeferredRenderer.Current?.CompositeSnapshot?.BindlessIndex ?? 0,
                InvViewportWidth = 1.0f / MathF.Max(1, depth?.Native.Description.Width ?? 1920),
                InvViewportHeight = 1.0f / MathF.Max(1, depth?.Native.Description.Height ?? 1080),
                TileScaleA = tileA,
                TileScaleB = tileB,
                BandA = bandA,
                BandB = bandB,
                Flowing = _spline != null && !_spline.Closed ? 1f : 0f,
                Ice = Ice,
                SnowCover = Weather.SnowCover,
                ShoreField = _shoreTexture?.BindlessIndex ?? 0u,
                ShoreOriginX = _shore?.Origin.X ?? 0f,
                ShoreOriginZ = _shore?.Origin.Y ?? 0f,
                ShoreSizeX = _shore != null ? _shore.Width * _shore.Cell : 1f,
                ShoreSizeZ = _shore != null ? _shore.Height * _shore.Cell : 1f,
                SnowAlongX = Weather.SnowLaidAlong.X,
                SnowAlongZ = Weather.SnowLaidAlong.Y,
                NearestDepthSRV = reflections?.NearestDepthSrv ?? 0u,
                NearestDepthMips = reflections?.NearestDepthMips ?? 0u,
                SceneColourSRV = reflections?.ColourSrv ?? 0u,
                SceneColourMips = reflections?.ColourMipCount ?? 0u,
            });

            CommandBuffer.Enqueue(mesh, _material, _params, Transform.TransformSlot);
        }
    }
}
