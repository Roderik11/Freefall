using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// Decals on the GPU (DECALS_PLAN.md Part B, the point lights' pattern): a persistent slot per Decal in a registry the
    /// GPU reads every frame. Nothing is gathered per frame: a decal's settings write its slot when they change (Decal's
    /// property setters, OnEnabledChanged), decals take and free slots as they enter and leave ComponentCache&lt;Decal&gt;,
    /// and the box comes from the entity's world matrix in the global transform buffer. Each renderer's DecalPass culls
    /// the registry and applies what is in view. Shader side: decals.hlsli.
    ///
    /// Spawned decals (DECALS_PLAN B3: footprints, blood, scorch; <see cref="Decals.Spawn"/>) live in a ring of
    /// <see cref="RingSize"/> slots at the start of the registry, each with a transform slot of its own: a spawn writes
    /// one record and one matrix, an event, never per-frame work; the cull fades it out by its age, and the oldest is
    /// overwritten when the ring comes round. Placed decals take the slots after the ring.
    /// </summary>
    public sealed class DecalBuffer : IDisposable
    {
        /// <summary>Visible decals per view (decals.hlsli MAX_VISIBLE_DECALS).</summary>
        public const int MaxVisible = 4096;

        /// <summary>Registry slots the sort's key can tell apart (decals.hlsli MAX_DECAL_SLOTS).</summary>
        public const int MaxSlots = 65536;

        /// <summary>Slots of the spawned ring: the newest this many spawned decals are kept.</summary>
        public const int RingSize = 4096;

        private const uint Enabled = 1;                 // decals.hlsli DECAL_ENABLED
        private const uint Spawned = 2;                 // DECAL_SPAWNED
        private const int PriorityShift = 8;            // DECAL_PRIORITY_SHIFT

        /// <summary>One per decal. Must match DecalRecord in decals.hlsli (48 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Record
        {
            public uint TransformSlot;
            public uint MaterialId;
            public uint Flags;
            public uint Variant;
            public Vector3 Tint;
            public float Opacity;
            public float SpawnTime, Lifetime, FadeOut, SpawnSnow;
        }

        public static DecalBuffer? Instance { get; private set; }

        private readonly StreamingBuffer<Record> _records;
        private readonly Stack<int> _freeSlots = new();
        private readonly object _lock = new();
        private int _nextSlot = RingSize;
        private bool _warnedFull;
        private int _placed;                    // placed decals registered

        private int[]? _ringTransforms;         // the ring's own transform slots, taken at the first spawn
        private int _ringNext;
        private double _ringLiveUntil = double.NegativeInfinity;   // Time.TotalTime the last spawned one has faded by

        /// <summary>Registry slots handed out so far (the ring's and the placed decals'): the cull walks this many.</summary>
        public int SlotCount => _nextSlot;

        /// <summary>Whether any decal can show: a placed one, or a spawned one not yet faded out.</summary>
        public bool Active => _placed > 0 || Base.Time.TotalTime < _ringLiveUntil;

        /// <summary>Spawned decals made since the ring last started over (a scene load).</summary>
        public int SpawnedCount => _ringNext;

        /// <summary>For engine_stats.</summary>
        public object Describe() => new
        {
            slots = SlotCount,
            placed = _placed,
            spawned = _ringNext,
            active = Active,
        };

        public uint SrvIndex => _records.SrvIndex;

        public static void Initialize(GraphicsDevice device)
        {
            if (Instance != null) return;
            Instance = new DecalBuffer(device);
            ComponentCache<Decal>.Added += Instance.Register;
            ComponentCache<Decal>.Removed += Instance.Unregister;
            foreach (var decal in ComponentCache<Decal>.All)
                Instance.Register(decal);

            // What was spawned belongs to the world it was spawned in (Play and Stop are loads too)
            SceneManager.LoadStarted += _ => Instance?.ClearSpawned();
        }

        /// <summary>Take every spawned decal away: the ring starts over.</summary>
        public void ClearSpawned()
        {
            lock (_lock)
            {
                for (int i = 0, n = Math.Min(_ringNext, RingSize); i < n; i++)
                    _records.Set(i, default); // disabled
                _ringNext = 0;
                _ringLiveUntil = double.NegativeInfinity;
            }
        }

        private DecalBuffer(GraphicsDevice device)
        {
            _records = new StreamingBuffer<Record>(device, RingSize + 256);
            _records.EnsureCapacity(RingSize);
        }

        // Called again for a decal that is already in (ComponentCache.Added may follow Initialize's walk)
        private void Register(Decal decal)
        {
            lock (_lock)
            {
                if (decal.GpuSlot < 0)
                {
                    if (_freeSlots.Count == 0 && _nextSlot >= MaxSlots)
                    {
                        if (!_warnedFull)
                            Debug.LogWarning("Decals", $"More than {MaxSlots} decals: the rest are not drawn");
                        _warnedFull = true;
                        return;
                    }
                    int slot = _freeSlots.Count > 0 ? _freeSlots.Pop() : _nextSlot++;
                    _records.EnsureCapacity(slot + 1);
                    decal.GpuSlot = slot;
                    _placed++;
                }
            }

            // A transform slot taken on first access holds identity until the transform next changes
            var transform = decal.Entity?.Transform;
            if (transform != null)
                TransformBuffer.Instance.SetTransform(transform.TransformSlot, transform.Matrix);
            Changed(decal);
        }

        private void Unregister(Decal decal)
        {
            lock (_lock)
            {
                int slot = decal.GpuSlot;
                if (slot < 0) return;
                _records.Set(slot, default); // disabled
                _freeSlots.Push(slot);
                decal.GpuSlot = -1;
                _placed--;
            }
        }

        /// <summary>
        /// A decal in the spawned ring (<see cref="Decals.Spawn"/>): its box as a world matrix (X and Z span it, Y is the
        /// projection axis, as a Decal's entity), lasting <paramref name="lifetime"/> seconds and fading out over the last
        /// <paramref name="fadeOut"/>; it overwrites the oldest when the ring comes round.
        /// </summary>
        internal void Spawn(Material material, in Matrix4x4 box, int variant, Vector3 tint, float opacity, int priority,
            float lifetime, float fadeOut)
        {
            if (material == null || !IsDecalMaterial(material) || TransformBuffer.Instance == null) return;
            lock (_lock)
            {
                if (_ringTransforms == null)
                {
                    _ringTransforms = new int[RingSize];
                    for (int i = 0; i < RingSize; i++)
                        _ringTransforms[i] = TransformBuffer.Instance.AllocateSlot();
                }
                int slot = _ringNext++ % RingSize;
                TransformBuffer.Instance.SetTransform(_ringTransforms[slot], box);
                double now = Base.Time.TotalTime;
                lifetime = MathF.Max(lifetime, 0.01f);
                _records.Set(slot, new Record
                {
                    TransformSlot = (uint)_ringTransforms[slot],
                    MaterialId = (uint)Math.Max(material.MaterialID, 0),
                    Flags = Enabled | Spawned | ((uint)Math.Clamp(priority, 0, 15) << PriorityShift),
                    Variant = (uint)Math.Max(variant, 0),
                    Tint = tint,
                    Opacity = Math.Clamp(opacity, 0f, 1f),
                    SpawnTime = (float)now,
                    Lifetime = lifetime,
                    FadeOut = Math.Clamp(fadeOut, 0.01f, lifetime),
                    SpawnSnow = Weather.SnowFallen,
                });
                _ringLiveUntil = Math.Max(_ringLiveUntil, now + lifetime);
            }
        }

        /// <summary>Writes a decal's slot from its settings. Decal calls it when one of them changes.</summary>
        internal void Changed(Decal decal)
        {
            int slot = decal.GpuSlot;
            if (slot < 0) return;

            int transformSlot = decal.Entity?.Transform?.TransformSlot ?? -1;
            // Only a material of the decal effect has the settings block the pass reads
            var material = decal.Material;
            bool drawn = decal.Enabled && transformSlot >= 0 && material != null && IsDecalMaterial(material);
            _records.Set(slot, new Record
            {
                TransformSlot = (uint)Math.Max(transformSlot, 0),
                MaterialId = (uint)Math.Max(material?.MaterialID ?? 0, 0),
                Flags = (drawn ? Enabled : 0) | ((uint)Math.Clamp(decal.Priority, 0, 15) << PriorityShift),
                Variant = (uint)Math.Max(decal.Variant, 0),
                Tint = new Vector3(decal.Tint.R, decal.Tint.G, decal.Tint.B),
                Opacity = Math.Clamp(decal.Opacity, 0f, 1f),
            });
        }

        /// <summary>Whether a material is a decal's (its effect is the engine's "decal").</summary>
        public static bool IsDecalMaterial(Material material) => material.Effect?.Key == "decal";

        /// <summary>Copies the slots that changed into this frame's GPU buffer. Once per frame, before the cull.</summary>
        public void Upload()
        {
            _records.Upload();
            StreamingBuffer<Record>.FlushDeferredDisposals();
        }

        public void Dispose() => _records.Dispose();
    }
}
