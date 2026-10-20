using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// Per-instance effect data: the values one object gives an effect's <c>struct Instance</c> (its "//=" members,
    /// FXParser.ParseInstanceParams), where the material's <c>struct Settings</c> hold what all its objects share.
    ///
    /// Every slot is <see cref="Stride"/> bytes of one raw buffer. A draw names its slot in
    /// InstanceDescriptor.CustomDataIdx, and the shader loads it from the buffer SceneConstants.InstanceDataIdx
    /// names: <c>ByteAddressBuffer all = ResourceDescriptorHeap[InstanceDataIdx]; all.Load&lt;Instance&gt;(slot * 128)</c>.
    /// Draws stay batched by effect: nothing here is a constant buffer, a push constant or a material.
    ///
    /// A renderer (PersistentRenderer) holds one slot per effect it draws with; changing a value rewrites those
    /// bytes and nothing else (no draw is registered again), so a script can fade an object every frame. Each
    /// effect with instance data also has a slot holding its defaults, for draws that bring none
    /// (Effect.DefaultInstanceSlot). Slot 0 is all zeros.
    ///
    /// The buffer is an upload heap rewritten in place when something changed, like the material settings
    /// (Material.SettingsStride): a frame still in flight may see a value one frame early, never a torn draw
    /// list. A freed slot is handed out again only after the frames in flight that may still read it are done.
    /// </summary>
    public static class InstanceData
    {
        /// <summary>Bytes per slot: the largest <c>struct Instance</c> an effect may declare.</summary>
        public const int Stride = 128;

        private const int ReuseAfterTicks = 3 + 1;   // frames in flight + 1, as GraphicsDevice.DeferDispose

        private static readonly Lock _lock = new();
        private static byte[] _data = new byte[Stride * 256];
        private static int _next = 1;                                   // slot 0: zeros
        private static readonly Stack<int> _free = new();
        private static readonly List<(int slot, int tick)> _released = new();
        // What changed since the last upload: only those bytes are copied (a town's 4000 windows hold a slot each, and
        // one lit window must not copy them all)
        private static bool _dirty = true;
        private static int _dirtyFrom, _dirtyTo = int.MaxValue;

        private static ID3D12Resource? _buffer;
        private static IntPtr _bufferPtr;
        private static int _bufferSlots;
        private static uint _bindlessIndex;

        /// <summary>
        /// The buffer's bindless index (SceneConstants.InstanceDataIdx). Made on first use, so a shader never
        /// reads descriptor 0 for it; never changes afterwards.
        /// </summary>
        public static uint BindlessIndex
        {
            get
            {
                if (_bindlessIndex == 0) Upload(Freefall.Engine.Device);
                return _bindlessIndex;
            }
        }

        /// <summary>A slot of its own (zeros until written). Give it back with <see cref="Release"/>.</summary>
        public static int Allocate()
        {
            lock (_lock)
            {
                ReclaimReleased();
                int slot = _free.Count > 0 ? _free.Pop() : _next++;
                int end = (slot + 1) * Stride;
                if (end > _data.Length)
                    Array.Resize(ref _data, Math.Max(_data.Length * 2, end));
                Array.Clear(_data, slot * Stride, Stride);
                MarkDirty(slot * Stride, end);
                return slot;
            }
        }

        /// <summary>Gives a slot back; it is handed out again once no frame in flight can still read it.</summary>
        public static void Release(int slot)
        {
            if (slot <= 0) return;
            lock (_lock)
                _released.Add((slot, Freefall.Engine.TickCount + ReuseAfterTicks));
        }

        /// <summary>Writes a value at a byte offset in a slot (an Instance member's MaterialParamDesc.Offset).</summary>
        public static void Write<T>(int slot, int offset, T value) where T : unmanaged
        {
            int size = Marshal.SizeOf<T>();
            if (slot <= 0 || offset < 0 || offset + size > Stride) return;
            lock (_lock)
            {
                int at = slot * Stride + offset;
                if (at + size > _data.Length) return;
                MemoryMarshal.Write(_data.AsSpan(at), in value);
                MarkDirty(at, at + size);
            }
        }

        // (under _lock)
        private static void MarkDirty(int from, int to)
        {
            if (!_dirty)
            {
                _dirty = true;
                _dirtyFrom = from;
                _dirtyTo = to;
                return;
            }
            _dirtyFrom = Math.Min(_dirtyFrom, from);
            _dirtyTo = Math.Max(_dirtyTo, to);
        }

        /// <summary>
        /// Writes one member of an effect's instance layout from a value given as up to four floats (a float uses
        /// X, a float2 XY, a float3 or colour XYZ; int, uint and bool take X).
        /// </summary>
        public static void Write(int slot, MaterialParamDesc member, Vector4 value)
        {
            switch (member.Type)
            {
                case "float": Write(slot, member.Offset, value.X); break;
                case "float2": Write(slot, member.Offset, new Vector2(value.X, value.Y)); break;
                case "float3": Write(slot, member.Offset, new Vector3(value.X, value.Y, value.Z)); break;
                case "float4": Write(slot, member.Offset, value); break;
                case "int": Write(slot, member.Offset, (int)value.X); break;
                case "uint": Write(slot, member.Offset, (uint)Math.Max(value.X, 0f)); break;
                case "bool": Write(slot, member.Offset, value.X != 0f ? 1u : 0u); break;
                case "TextureRef": Write(slot, member.Offset, (uint)Math.Max(value.X, 0f)); break;   // a bindless index
            }
        }

        /// <summary>Fills a slot with an effect's instance defaults.</summary>
        public static void WriteDefaults(int slot, IReadOnlyList<MaterialParamDesc> layout)
        {
            foreach (var member in layout)
                Write(slot, member, DefaultOf(member));
        }

        /// <summary>A member's default as up to four floats (colours default to white when not given).</summary>
        public static Vector4 DefaultOf(MaterialParamDesc member)
        {
            if (member.IsTexture)
                return new Vector4(member.DefaultTextureAsset?.BindlessIndex ?? 0u, 0f, 0f, 0f);
            float fallback = member.IsColor ? 1f : 0f;
            return new Vector4(member.At(0, fallback), member.At(1, fallback), member.At(2, fallback), member.At(3));
        }

        /// <summary>
        /// Creates or grows the buffer and copies what changed. Once a frame before drawing (DeferredRenderer,
        /// next to the materials buffer).
        /// </summary>
        public static void Upload(GraphicsDevice device)
        {
            if (device == null) return;
            lock (_lock)
            {
                int slots = Math.Max(_next, 256);
                if (_buffer == null || slots > _bufferSlots)
                {
                    int capacity = Math.Max(slots, _bufferSlots * 2);
                    int bytes = capacity * Stride;
                    if (_buffer != null) device.DeferDispose(_buffer);
                    _buffer = device.NativeDevice.CreateCommittedResource(
                        new HeapProperties(HeapType.Upload),
                        HeapFlags.None,
                        ResourceDescription.Buffer((ulong)bytes),
                        ResourceStates.GenericRead);
                    unsafe
                    {
                        void* ptr;
                        _buffer.Map(0, null, &ptr);
                        _bufferPtr = (IntPtr)ptr;
                    }
                    if (_bindlessIndex == 0) _bindlessIndex = device.AllocateBindlessIndex();
                    device.CreateRawBufferSRV(_buffer, (uint)(bytes / 4), _bindlessIndex);
                    _bufferSlots = capacity;
                    _dirty = true;                  // a new buffer takes everything
                    _dirtyFrom = 0;
                    _dirtyTo = int.MaxValue;
                }

                if (!_dirty) return;
                int from = Math.Max(_dirtyFrom, 0);
                int to = Math.Min(_dirtyTo, Math.Min(_data.Length, _bufferSlots * Stride));
                if (to > from)
                    unsafe
                    {
                        fixed (byte* src = _data)
                            Buffer.MemoryCopy(src + from, (byte*)_bufferPtr + from, (long)_bufferSlots * Stride - from, to - from);
                    }
                _dirty = false;
            }
        }

        // Released slots whose frames are done go back to the free list (under _lock)
        private static void ReclaimReleased()
        {
            int now = Freefall.Engine.TickCount;
            for (int i = _released.Count - 1; i >= 0; i--)
            {
                if (_released[i].tick > now) continue;
                _free.Push(_released[i].slot);
                _released.RemoveAt(i);
            }
        }
    }
}
