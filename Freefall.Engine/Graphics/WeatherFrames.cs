using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Base;
using Freefall.Components;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Which transform's frame the weather on a surface follows, by the surface's transform slot (the slot in the
    /// G-buffer's EntityId). Kept by <see cref="WeatherFrame"/>; a slot given back (<see cref="TransformBuffer.ReleaseSlot"/>)
    /// leaves the table. Two things ride with the frame:
    ///   - the snow's pattern: its drifts, wind ripples, grain and crystals are anchored to the world, so on a ship or a
    ///     cart they slid over it as it moved; a slot entered here takes them in its frame's rigid space
    ///     (decals_cs.hlsl WeatherSpaceOf, ApplyWeather);
    ///   - its shelter: the world's cover map (<see cref="ShelterMap"/>) is drawn once per camera cell and leaves what
    ///     moves out (<see cref="PersistentRenderer.InSkyCaptures"/> off), so a ship under way had no cover at all: snow
    ///     under its castles and in its hold (user, 2026-11-22). A frame gets a cover map of its own instead, drawn once
    ///     in its own frame, straight down its own up (<see cref="ShelterMap"/> draws it, one a frame, from the meshes out
    ///     of the sky captures alone, so not the quay it lay at), as soon as all of it is out of the sky captures. A place on
    ///     it is sheltered by that, read in the frame's coordinates, and by the world's map where it stands now (a bridge
    ///     over it). Still in the sky captures (a ship at its berth), the world's map holds it and its own is not read.
    ///     Read in the world's map where it stood when that was drawn (the first try, the same day), its cover changed
    ///     with every redraw: drawn straight down the world's up, a sail 10 m over the deck lands 2 m to one side at 10°
    ///     of heel, and the sails' shape and the yards' brace differ each time ("the snow pattern changes every couple
    ///     seconds", user).
    /// On the GPU an open-addressed table of (slot + 1, frame slot, frame index, 0), a power of two long and at most half
    /// full, rewritten when it changes, and per frame index its cover map: (left, top, size, map) and (lowest, height
    /// range, drawn, 0) in the frame's coordinates (weather_cover.hlsli ShelterLevel).
    /// Docs: .agent/knowledge/atmospheric_effects/weather.md "Snow on what moves".
    /// </summary>
    public static class WeatherFrames
    {
        [StructLayout(LayoutKind.Sequential)]
        private struct Entry
        {
            public uint Key;    // slot + 1, 0 = empty
            public uint Frame;  // the frame's transform slot
            public uint Root;   // the frame's index in the covers; NoRoot without one
            public uint Pad;
        }

        private const uint NoRoot = 0xFFFFFFFFu;

        /// <summary>A frame: the entity whose frame it is and its own cover map.</summary>
        internal sealed class Root
        {
            public int Index;
            public int Slot;                // its transform slot
            public Transform Transform = null!;
            public bool Check = true;       // whether all of it is out of the sky captures is to be looked at again
            public bool Due;                // its cover is to be drawn
            public bool Drawn;              // Map holds its cover, read by the decal pass
            public RenderTexture2D? Map;
            public bool MapReadable;
            public float Left, Top, Size, MinY, MaxY;   // the map's square and heights, in the frame's coordinates
        }

        private const int FrameCount = 3;
        private const int MinCapacity = 256;

        private static readonly Dictionary<int, int> _frames = new();
        private static readonly Dictionary<int, Root> _roots = new();
        private static readonly object _lock = new();
        private static Entry[] _table = Array.Empty<Entry>();
        private static Vector4[] _covers = Array.Empty<Vector4>();
        private static int _version, _builtVersion = -1;
        private static readonly GraphicsBuffer?[] _buffers = new GraphicsBuffer?[FrameCount];
        private static readonly GraphicsBuffer?[] _coverBuffers = new GraphicsBuffer?[FrameCount];
        private static readonly int[] _bufferVersion = { -1, -1, -1 };
        private static int _uploadedTick = -1;
        private static uint _srv, _mask, _coversSrv;

        /// <summary>How many slots follow a frame.</summary>
        public static int Count
        {
            get { lock (_lock) return _frames.Count; }
        }

        /// <summary>Let the surfaces of <paramref name="slot"/> take the weather in <paramref name="frameSlot"/>'s frame.</summary>
        public static void Set(int slot, int frameSlot)
        {
            if (slot < 0 || frameSlot < 0) return;
            lock (_lock)
            {
                if (_frames.TryGetValue(slot, out int was) && was == frameSlot) return;
                _frames[slot] = frameSlot;
                if (_roots.TryGetValue(frameSlot, out var root)) root.Check = true;
                _version++;
            }
        }

        /// <summary>Back to the world for <paramref name="slot"/>.</summary>
        public static void Remove(int slot)
        {
            lock (_lock)
            {
                if (_frames.Count == 0 || !_frames.Remove(slot)) return;
                _version++;
            }
        }

        /// <summary>The entity whose transform slot <paramref name="frameSlot"/> is a frame: its own cover map is kept for it.</summary>
        public static void SetRoot(int frameSlot, Transform root)
        {
            if (frameSlot < 0 || root == null) return;
            lock (_lock)
            {
                if (_roots.TryGetValue(frameSlot, out var was) && was.Transform == root) { was.Check = true; return; }
                if (was?.Map != null) Engine.Device?.DeferDispose(was.Map);
                _roots[frameSlot] = new Root { Slot = frameSlot, Transform = root };
                _version++;
            }
        }

        /// <summary>No longer a frame: its cover map goes.</summary>
        public static void RemoveRoot(int frameSlot)
        {
            lock (_lock)
            {
                if (!_roots.Remove(frameSlot, out var root)) return;
                if (root.Map != null) Engine.Device?.DeferDispose(root.Map);
                _version++;
            }
        }

        /// <summary>
        /// A renderer went into the sky captures or out of them (<see cref="PersistentRenderer.InSkyCaptures"/>): its frame,
        /// if it has one, looks again at whether all of it is out (then its own cover is drawn) or not (then the world's
        /// map holds it).
        /// </summary>
        internal static void SkyCapturesChanged(int slot)
        {
            lock (_lock)
            {
                if (_frames.Count == 0 || !_frames.TryGetValue(slot, out int frame)) return;
                if (_roots.TryGetValue(frame, out var root)) root.Check = true;
            }
        }

        /// <summary>
        /// The next frame whose own cover is to be drawn, or null (<see cref="ShelterMap"/>, main thread, once a frame).
        /// Frames told to look again do so here: all out of the sky captures and not drawn, it is due; any of it back in,
        /// its own map is no longer read.
        /// </summary>
        internal static Root? NextCover()
        {
            lock (_lock)
            {
                Root? next = null;
                foreach (var root in _roots.Values)
                {
                    if (root.Check)
                    {
                        root.Check = false;
                        bool outOfCaptures = OutOfSkyCaptures(root.Transform);
                        if (!outOfCaptures && root.Drawn)
                        {
                            root.Drawn = false;
                            _version++;
                        }
                        root.Due = outOfCaptures && !root.Drawn;
                    }
                    if (root.Due && next == null) next = root;
                }
                return next;
            }
        }

        /// <summary><paramref name="root"/>'s own cover was drawn into its map, over this square and these heights.</summary>
        internal static void CoverDrawn(Root root, float left, float top, float size, float minY, float maxY)
        {
            lock (_lock)
            {
                root.Left = left;
                root.Top = top;
                root.Size = size;
                root.MinY = minY;
                root.MaxY = maxY;
                root.Due = false;
                root.Drawn = true;
                _version++;
            }
        }

        /// <summary>Nothing to draw for <paramref name="root"/> (no meshes): not due until it is told to look again.</summary>
        internal static void CoverSkipped(Root root)
        {
            lock (_lock) root.Due = false;
        }

        // Every renderer under the frame (not under another frame) out of the sky captures, and at least one
        private static bool OutOfSkyCaptures(Transform root)
        {
            int renderers = 0;
            return OutOfSkyCaptures(root, root, ref renderers) && renderers > 0;
        }

        private static bool OutOfSkyCaptures(Transform t, Transform root, ref int renderers)
        {
            var entity = t.Entity;
            if (entity == null) return true;
            if (t != root && entity.GetComponent<WeatherFrame>() is { Enabled: true }) return true;   // its own frame
            var mesh = entity.GetComponent<MeshRenderer>();
            if (mesh != null)
            {
                renderers++;
                if (mesh.InSkyCaptures) return false;
            }
            var skinned = entity.GetComponent<SkinnedMeshRenderer>();
            if (skinned != null && skinned.InSkyCaptures) return false;
            for (int i = 0; i < t.Count; i++)
            {
                var child = t.GetChild(i);
                if (child != null && !OutOfSkyCaptures(child, root, ref renderers)) return false;
            }
            return true;
        }

        /// <summary>
        /// The frame's rigid placement (unit axes, its origin: the space the decal pass lays its weather in, scale taken
        /// out) and the box its meshes fill in it. False when it has none.
        /// </summary>
        internal static bool Placement(Root root, out Matrix4x4 rigid, out Vector3 min, out Vector3 max)
        {
            var world = root.Transform.WorldMatrix;
            var x = Unit(new Vector3(world.M11, world.M12, world.M13));
            var y = Unit(new Vector3(world.M21, world.M22, world.M23));
            var z = Unit(new Vector3(world.M31, world.M32, world.M33));
            rigid = new Matrix4x4(x.X, x.Y, x.Z, 0, y.X, y.Y, y.Z, 0, z.X, z.Y, z.Z, 0, world.M41, world.M42, world.M43, 1);
            min = new Vector3(float.MaxValue);
            max = new Vector3(float.MinValue);
            if (!Matrix4x4.Invert(rigid, out var toFrame)) return false;
            Extent(root.Transform, root.Transform, toFrame, ref min, ref max);
            return min.X <= max.X;
        }

        private static void Extent(Transform t, Transform root, Matrix4x4 toFrame, ref Vector3 min, ref Vector3 max)
        {
            var entity = t.Entity;
            if (entity == null) return;
            if (t != root && entity.GetComponent<WeatherFrame>() is { Enabled: true }) return;
            var renderer = entity.GetComponent<MeshRenderer>();
            var mesh = renderer?.Mesh;
            if (mesh != null)
            {
                var m = mesh.RootRotation * t.WorldMatrix * toFrame;
                var box = mesh.BoundingBox;
                for (int i = 0; i < 8; i++)
                {
                    var corner = new Vector3((i & 1) != 0 ? box.Max.X : box.Min.X,
                                             (i & 2) != 0 ? box.Max.Y : box.Min.Y,
                                             (i & 4) != 0 ? box.Max.Z : box.Min.Z);
                    var p = Vector3.Transform(corner, m);
                    min = Vector3.Min(min, p);
                    max = Vector3.Max(max, p);
                }
            }
            for (int i = 0; i < t.Count; i++)
            {
                var child = t.GetChild(i);
                if (child != null) Extent(child, root, toFrame, ref min, ref max);
            }
        }

        private static Vector3 Unit(Vector3 v) => v.LengthSquared() > 1e-12f ? Vector3.Normalize(v) : Vector3.Zero;

        /// <summary>The table for this frame (StructuredBuffer&lt;uint4&gt;), 0 when no slot follows a frame.</summary>
        public static uint Srv
        {
            get
            {
                Upload();
                return _srv;
            }
        }

        /// <summary>The table's length − 1 (it is a power of two).</summary>
        public static uint Mask
        {
            get
            {
                Upload();
                return _mask;
            }
        }

        /// <summary>
        /// The frames' own cover maps (StructuredBuffer&lt;float4&gt;: [root × 2] = (left, top, size, map), [root × 2 + 1] =
        /// (lowest, height range, drawn, 0), in the frame's coordinates), 0 when no slot follows a frame.
        /// </summary>
        public static uint CoversSrv
        {
            get
            {
                Upload();
                return _coversSrv;
            }
        }

        // Once a tick: the table and the covers rebuilt when they changed, and this frame's buffers brought up to them
        private static unsafe void Upload()
        {
            if (_uploadedTick == Engine.TickCount) return;
            _uploadedTick = Engine.TickCount;
            _srv = 0;
            _mask = 0;
            _coversSrv = 0;
            if (Engine.Device == null) return;
            int version;
            lock (_lock)
            {
                if (_frames.Count == 0) return;
                version = _version;
                if (_builtVersion != version)
                    Build();
            }

            int f = Engine.FrameIndex % FrameCount;
            var buffer = _buffers[f];
            if (buffer == null || buffer.ElementCount < _table.Length)
            {
                if (buffer != null) Engine.Device.DeferDispose(buffer);
                buffer = _buffers[f] = GraphicsBuffer.CreateUpload<Entry>(_table.Length, mapped: true);
                _bufferVersion[f] = -1;
            }
            var covers = _coverBuffers[f];
            if (covers == null || covers.ElementCount < _covers.Length)
            {
                if (covers != null) Engine.Device.DeferDispose(covers);
                covers = _coverBuffers[f] = GraphicsBuffer.CreateUpload<Vector4>(Math.Max(_covers.Length, 64), mapped: true);
                _bufferVersion[f] = -1;
            }
            if (_bufferVersion[f] != version)
            {
                var write = buffer.WritePtr<Entry>();
                for (int i = 0; i < _table.Length; i++) write[i] = _table[i];
                var writeCovers = covers.WritePtr<Vector4>();
                for (int i = 0; i < _covers.Length; i++) writeCovers[i] = _covers[i];
                _bufferVersion[f] = version;
            }
            _srv = buffer.SrvIndex;
            _mask = (uint)_table.Length - 1;
            _coversSrv = covers.SrvIndex;
        }

        // Under the lock: the frames numbered, the table hashed, their covers laid out by number
        private static void Build()
        {
            int index = 0;
            foreach (var root in _roots.Values) root.Index = index++;

            int capacity = MinCapacity;
            while (capacity < _frames.Count * 2) capacity *= 2;
            if (_table.Length != capacity) _table = new Entry[capacity];
            else Array.Clear(_table);
            uint mask = (uint)capacity - 1;
            foreach (var (slot, frame) in _frames)
            {
                uint at = Hash((uint)slot) & mask;
                while (_table[at].Key != 0) at = (at + 1) & mask;
                _table[at] = new Entry
                {
                    Key = (uint)slot + 1,
                    Frame = (uint)frame,
                    Root = _roots.TryGetValue(frame, out var root) ? (uint)root.Index : NoRoot,
                };
            }

            int length = _roots.Count * 2;
            if (_covers.Length != length) _covers = new Vector4[length];
            foreach (var root in _roots.Values)
            {
                bool drawn = root.Drawn && root.Map != null;
                _covers[root.Index * 2] = new Vector4(root.Left, root.Top, root.Size,
                    BitConverter.UInt32BitsToSingle(drawn ? root.Map!.BindlessIndex : 0u));
                _covers[root.Index * 2 + 1] = new Vector4(root.MinY, MathF.Max(root.MaxY - root.MinY, 1e-3f), drawn ? 1f : 0f, 0f);
            }
            _builtVersion = _version;
        }

        // decals_cs.hlsl WeatherSpaceOf: Knuth's multiplicative hash
        private static uint Hash(uint slot) => unchecked(slot * 2654435761u);
    }
}
