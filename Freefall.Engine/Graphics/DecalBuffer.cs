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
    /// </summary>
    public sealed class DecalBuffer : IDisposable
    {
        /// <summary>Visible decals per view (decals.hlsli MAX_VISIBLE_DECALS).</summary>
        public const int MaxVisible = 4096;

        /// <summary>Registry slots the sort's key can tell apart (decals.hlsli MAX_DECAL_SLOTS).</summary>
        public const int MaxSlots = 65536;

        private const uint Enabled = 1;                 // decals.hlsli DECAL_ENABLED
        private const int PriorityShift = 8;            // DECAL_PRIORITY_SHIFT

        /// <summary>One per decal. Must match DecalRecord in decals.hlsli (32 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Record
        {
            public uint TransformSlot;
            public uint MaterialId;
            public uint Flags;
            public uint Variant;
            public Vector3 Tint;
            public float Opacity;
        }

        public static DecalBuffer? Instance { get; private set; }

        private readonly StreamingBuffer<Record> _records;
        private readonly Stack<int> _freeSlots = new();
        private readonly object _lock = new();
        private int _nextSlot;
        private bool _warnedFull;

        /// <summary>Registry slots handed out so far: the cull walks this many.</summary>
        public int SlotCount => _nextSlot;

        public uint SrvIndex => _records.SrvIndex;

        public static void Initialize(GraphicsDevice device)
        {
            if (Instance != null) return;
            Instance = new DecalBuffer(device);
            ComponentCache<Decal>.Added += Instance.Register;
            ComponentCache<Decal>.Removed += Instance.Unregister;
            foreach (var decal in ComponentCache<Decal>.All)
                Instance.Register(decal);
        }

        private DecalBuffer(GraphicsDevice device)
        {
            _records = new StreamingBuffer<Record>(device, 256);
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
