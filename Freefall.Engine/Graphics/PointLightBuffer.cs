using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// Point lights on the GPU: a persistent slot per PointLight in a registry the GPU reads every frame. Nothing
    /// is gathered per frame: a light's settings write its slot when they change (PointLight's property setters,
    /// OnEnabledChanged), lights take and free slots as they enter and leave ComponentCache&lt;PointLight&gt;, the
    /// position comes from the light entity's world matrix in the global transform buffer, and the flicker is
    /// computed in the shader. The GPU culls the registry into each renderer's visible list
    /// (point_light_cull.hlsl), which the tiled lighting and the shadow selection read.
    /// Shader side: point_lights.hlsli.
    /// </summary>
    public sealed class PointLightBuffer : IDisposable
    {
        /// <summary>Visible lights per renderer and frame (point_lights.hlsli MAX_VISIBLE_LIGHTS).</summary>
        public const int MaxVisible = 4096;

        private const uint Enabled = 1;

        /// <summary>One per light. Must match PointLightRecord in point_lights.hlsli (48 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Record
        {
            public Vector3 Color;
            public float Intensity;
            public uint TransformSlot;
            public float Range;
            public float Flicker;
            public float FlickerSpeed;
            public float FlickerPhase;
            public uint Flags;
            private uint _pad0, _pad1;
        }

        /// <summary>A light in view, written by the GPU. Must match PointLightData in point_lights.hlsli (48 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct VisibleLight
        {
            public Vector3 Color;
            public float Intensity;
            public Vector3 Position;
            public float Range;
            public uint ShadowLight;
            public uint Slot;
            public uint Flags;
            public float Score;
        }

        /// <summary>The camera and frame for the cull and the shadow selection. Must match PointLightFrame (192 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct FrameParams
        {
            public Matrix4x4 CameraRelativeVP;
            public Vector4 Plane0, Plane1, Plane2, Plane3, Plane4;  // camera-relative left, right, bottom, top, near
            public Vector3 CameraPosition;
            public float Time;
            public float PixelsPerMetre;
            public float FadeStep;
            public float SunLight;          // what the sun adds to a lit surface (× albedo), clouds included
            public uint ShadowBudget;
            public uint Frame;
            public float Ambient;           // what the sky's ambient adds to an up-facing surface (× albedo)
            private uint _pad0, _pad1;
        }

        public static PointLightBuffer? Instance { get; private set; }

        private readonly StreamingBuffer<Record> _records;
        private readonly Stack<int> _freeSlots = new();
        private readonly object _lock = new();
        private PointLight?[] _lights;
        private int _nextSlot;

        /// <summary>Registry slots handed out so far: the cull walks this many.</summary>
        public int SlotCount => _nextSlot;

        /// <summary>Slots the GPU buffer holds (per-slot buffers elsewhere size by this).</summary>
        public int Capacity => _records.Capacity;

        public uint SrvIndex => _records.SrvIndex;

        public static void Initialize(GraphicsDevice device)
        {
            if (Instance != null) return;
            Instance = new PointLightBuffer(device);
            ComponentCache<PointLight>.Added += Instance.Register;
            ComponentCache<PointLight>.Removed += Instance.Unregister;
            foreach (var light in ComponentCache<PointLight>.All)
                Instance.Register(light);
        }

        private PointLightBuffer(GraphicsDevice device)
        {
            _records = new StreamingBuffer<Record>(device, 1024);
            _lights = new PointLight?[_records.Capacity];
        }

        /// <summary>The light at a registry slot (editor stats), or null.</summary>
        public PointLight? LightAt(int slot)
        {
            lock (_lock)
                return slot >= 0 && slot < _lights.Length ? _lights[slot] : null;
        }

        // Called again for a light that is already in (ComponentCache.Added may follow Initialize's walk)
        private void Register(PointLight light)
        {
            lock (_lock)
            {
                if (light.GpuSlot < 0)
                {
                    int slot = _freeSlots.Count > 0 ? _freeSlots.Pop() : _nextSlot++;
                    _records.EnsureCapacity(slot + 1);
                    if (_lights.Length < _records.Capacity)
                        Array.Resize(ref _lights, _records.Capacity);
                    _lights[slot] = light;
                    light.GpuSlot = slot;
                }
            }

            // A transform slot taken on first access holds identity until the transform next changes
            var transform = light.Entity?.Transform;
            if (transform != null)
                TransformBuffer.Instance.SetTransform(transform.TransformSlot, transform.Matrix);
            Changed(light);
        }

        private void Unregister(PointLight light)
        {
            lock (_lock)
            {
                int slot = light.GpuSlot;
                if (slot < 0) return;
                _records.Set(slot, default); // disabled
                _lights[slot] = null;
                _freeSlots.Push(slot);
                light.GpuSlot = -1;
            }
        }

        /// <summary>Writes a light's slot from its settings. PointLight calls it when one of them changes.</summary>
        internal void Changed(PointLight light)
        {
            int slot = light.GpuSlot;
            if (slot < 0) return;

            int transformSlot = light.Entity?.Transform?.TransformSlot ?? -1;
            var c = light.Color;
            _records.Set(slot, new Record
            {
                Color = new Vector3(c.R, c.G, c.B),
                Intensity = light.Intensity,
                TransformSlot = (uint)Math.Max(transformSlot, 0),
                Range = light.Range,
                Flicker = light.Flicker,
                FlickerSpeed = light.FlickerSpeed,
                FlickerPhase = (light.Id & 1023) * 0.37f, // neighbouring lights do not waver in step
                Flags = light.Enabled && transformSlot >= 0 ? Enabled : 0,
            });
        }

        /// <summary>Copies the slots that changed into this frame's GPU buffer. Once per frame, before the cull.</summary>
        public void Upload()
        {
            _records.Upload();
            StreamingBuffer<Record>.FlushDeferredDisposals();
        }

        public void Dispose() => _records.Dispose();
    }
}
