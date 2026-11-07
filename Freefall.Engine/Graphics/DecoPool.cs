using System;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// A terrain's decoration control (the top 8 decorators per texel, one texel a metre) as pages, the decoration pool
    /// (TERRAIN_PAGES_PLAN.md 5c). Until 5c one Texture2DArray held the whole map (2 slices of RGBA16_UINT, Y-flipped).
    ///
    /// A page is 256 texels a side inside a border of 1, so a bilinear gather of four texels (the terrain's sward tint)
    /// stays in it: <see cref="SlotTexels"/> (258). A level L page covers 2^L × 2^L level 0 pages, each texel each
    /// decorator's coverage averaged over its ground and the top 8 kept. Pages lie in terrain UV with +Z up. Four slots to a
    /// slice pair, 2 × 2; a slot's two slices are array slices (slot / 4) · 2 and + 1 of <see cref="Control"/>, each channel
    /// (decorator slot &lt;&lt; 8) | coverage, 255 &lt;&lt; 8 empty (decoration_prepass.hlsl). Which page is in which slot,
    /// and the page table, are <see cref="PageSlots"/>'s. Shaders read it only through terrain_deco.hlsli, from
    /// <c>TerrainDecoParams</c>, whose index the readers get by push constant (the terrain's DecoControlMapIdx, the spawn's
    /// DecoControlIdx).
    /// </summary>
    public sealed class DecoPool : IDisposable, IPagedMap
    {
        public const int PageTexels = TerrainMapResidency.PageTexels;
        public const int Border = 1;
        public const int SlotTexels = PageTexels + 2 * Border;
        public const int SliceTexels = 2 * SlotTexels;
        /// <summary>Array slices a slot takes: two RGBA16_UINT texels hold the 8 entries.</summary>
        public const int SlicesPerSlot = 2;

        private const int FrameCount = 3;

        object? IPagedMap.Resource => Control;
        int IPagedMap.Border => Border;

        /// <summary>terrain_deco.hlsli TerrainDecoParams. 32 bytes.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Params
        {
            public uint Control, PageTable, PagesAcross, Texels;
            public uint Pad0, Pad1, Pad2, Pad3;
        }

        public PageSlots Slots { get; } = new();

        /// <summary>Texels a side of the whole map at level 0 (Terrain.EffectiveDecorationMapResolution).</summary>
        public int Texels { get; private set; }

        public int Capacity => Control != null ? Slots.Capacity : 0;

        /// <summary>The pages (Texture2DArray RGBA16_UINT). Rests in Common.</summary>
        public ID3D12Resource? Control { get; private set; }
        public uint ControlSrv { get; private set; }
        public uint ControlUav { get; private set; }

        private readonly GraphicsBuffer?[] _params = new GraphicsBuffer?[FrameCount];

        /// <summary>Array slices this many slots take.</summary>
        public static int SlicesFor(int slots) => (slots + 3) / 4 * SlicesPerSlot;

        public bool Matches(int texels, int capacity)
        {
            bool all = capacity <= 0;
            int across = TerrainMapResidency.PagesAcrossFor(texels);
            return Control != null && Texels == texels && Slots.AllResident == all
                   && Slots.Capacity == (all ? across * across : capacity);
        }

        /// <summary>
        /// The pool for a map of this many texels a side and page slots (0: every level 0 page at its own index), made when
        /// there is none like it; the table goes up on the next <c>Slots.RecordTable</c>. True when it was made just now.
        /// </summary>
        public bool Ensure(int texels, int capacity)
        {
            if (Matches(texels, capacity)) return false;
            bool all = capacity <= 0;
            int across = TerrainMapResidency.PagesAcrossFor(texels);
            int slots = all ? across * across : capacity;
            int slices = SlicesFor(slots);
            if (slices > HeightPool.MaxSlices)
                throw new InvalidOperationException($"A decoration pool of {slots} pages needs {slices} array slices: D3D12 allows {HeightPool.MaxSlices}");
            Release();
            Texels = texels;

            var device = Engine.Device;
            Control = device.CreateTexture2D(Format.R16G16B16A16_UInt, SliceTexels, SliceTexels, slices, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
            Control.Name = "Terrain decoration pool";
            ControlSrv = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(Control, new ShaderResourceViewDescription
            {
                Format = Format.R16G16B16A16_UInt,
                ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2DArray = new Texture2DArrayShaderResourceView { MostDetailedMip = 0, MipLevels = 1, FirstArraySlice = 0, ArraySize = (uint)slices },
            }, device.GetCpuHandle(ControlSrv));
            ControlUav = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(Control, null, new UnorderedAccessViewDescription
            {
                Format = Format.R16G16B16A16_UInt,
                ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                Texture2DArray = new Texture2DArrayUnorderedAccessView { MipSlice = 0, FirstArraySlice = 0, ArraySize = (uint)slices },
            }, device.GetCpuHandle(ControlUav));

            Slots.Reset(across, all ? 0 : capacity, 0, all);
            return true;
        }

        /// <summary>A slot's quarter corner (texels) and the array slice of its first slice.</summary>
        public (int x, int y, int slice) SlotPlace(int slot)
        {
            int q = slot & 3;
            return ((q & 1) * SlotTexels, (q >> 1) * SlotTexels, (slot >> 2) * SlicesPerSlot);
        }

        /// <summary>This frame's <c>TerrainDecoParams</c>: its bindless index, 0 before the pool is made.</summary>
        public uint WriteParams()
        {
            if (Control == null) return 0;
            int slot = Engine.FrameIndex % FrameCount;   // the buffer of this frame in flight
            var buffer = _params[slot] ??= GraphicsBuffer.CreateUpload<Params>(1, mapped: true);
            unsafe
            {
                *buffer.WritePtr<Params>() = new Params
                {
                    Control = ControlSrv,
                    PageTable = Slots.TableSrv,
                    PagesAcross = (uint)Slots.PagesAcross,
                    Texels = (uint)Texels,
                };
            }
            return buffer.SrvIndex;
        }

        private void Release()
        {
            var device = Engine.Device;
            if (Control != null)
            {
                device.DeferDispose(Control);
                device.DeferReleaseBindlessIndex(ControlSrv);
                device.DeferReleaseBindlessIndex(ControlUav);
            }
            Slots.Release();
            Control = null;
            ControlSrv = ControlUav = 0;
            Texels = 0;
        }

        public void Dispose()
        {
            Release();
            for (int i = 0; i < _params.Length; i++)
            {
                Engine.Device.DeferDispose(_params[i]);
                _params[i] = null;
            }
        }
    }
}
