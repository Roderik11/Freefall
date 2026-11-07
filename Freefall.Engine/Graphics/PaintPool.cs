using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// A terrain's paint (the layers' blend weights) as pages, the paint pool (TERRAIN_PAGES_PLAN.md 5b). Until 5b one
    /// Texture2DArray held the whole map at TerrainSize × PaintPixelsPerMetre a side.
    ///
    /// A page is <see cref="PageTexels"/> (256) paint texels a side inside a border of <see cref="Border"/> (4), so an
    /// anisotropic footprint stays in it: <see cref="SlotTexels"/> (264). A level L page covers 2^L × 2^L level 0 pages,
    /// each texel the average of 2^L × 2^L level 0 texels' ground. Pages lie in terrain UV, +Z up (the whole-map array was
    /// Y-flipped). Four slots to a slice, 2 × 2, like the heights; a slot's layer slice g (four layers each) is array
    /// slice (slot / 4) · LayerSlices + g of <see cref="Weights"/> (RGBA8), and its top layer (the channel that shows most,
    /// R8_UINT, 255 = none) array slice slot / 4 of <see cref="TopLayer"/>. Which page is in which slot, and the page table,
    /// are <see cref="PageSlots"/>'s. Shaders read it only through terrain_paint.hlsli, from <c>TerrainPaintParams</c>,
    /// whose index the terrain's height params carry (<c>TerrainPagesParams.PaintIdx</c>).
    /// </summary>
    public sealed class PaintPool : IDisposable, IPagedMap
    {
        object? IPagedMap.Resource => Weights;
        int IPagedMap.Border => Border;

        /// <summary>Paint texels a page holds a side.</summary>
        public const int PageTexels = 256;
        /// <summary>Texels round a page, so filtering near its edge (anisotropic in the terrain's shader) stays in it.</summary>
        public const int Border = 4;
        /// <summary>Texels a page takes a side in its slice, border included.</summary>
        public const int SlotTexels = PageTexels + 2 * Border;
        /// <summary>Texels a slice has a side: 2 × 2 slots.</summary>
        public const int SliceTexels = 2 * SlotTexels;

        private const int FrameCount = 3;

        /// <summary>terrain_paint.hlsli TerrainPaintParams. 64 bytes.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Params
        {
            public uint Weights, TopLayer, PageTable, PagesAcross;
            public uint Texels, LayerSlices, LayerCount, Pad0;
            public Vector4 Pad1, Pad2;
        }

        /// <summary>Which page is in which slot, and the page table.</summary>
        public PageSlots Slots { get; } = new();

        /// <summary>Paint texels a side of the whole map at level 0 (Terrain.EffectiveSplatmapResolution).</summary>
        public int Texels { get; private set; }

        /// <summary>Array slices a slot's weights take: four layers each.</summary>
        public int LayerSlices { get; private set; }

        public int Capacity => Weights != null ? Slots.Capacity : 0;

        /// <summary>The weights (Texture2DArray RGBA8_UNorm) and the top layer (Texture2DArray R8_UInt). Rest in Common.</summary>
        public ID3D12Resource? Weights { get; private set; }
        public ID3D12Resource? TopLayer { get; private set; }

        public uint WeightsSrv { get; private set; }
        public uint WeightsUav { get; private set; }
        public uint TopSrv { get; private set; }
        public uint TopUav { get; private set; }

        private readonly GraphicsBuffer?[] _params = new GraphicsBuffer?[FrameCount];

        /// <summary>Level 0 pages a side for a map of this many paint texels.</summary>
        public static int PagesAcrossFor(int texels) => Math.Max(1, (texels + PageTexels - 1) / PageTexels);

        /// <summary>Array slices the weights of this many slots take.</summary>
        public static int WeightSlicesFor(int slots, int layerSlices) => (slots + 3) / 4 * layerSlices;

        /// <summary>The pool is made for a map of this many texels, layer slices and page slots (0: every level 0 page).</summary>
        public bool Matches(int texels, int layerSlices, int capacity)
        {
            bool all = capacity <= 0;
            int across = PagesAcrossFor(texels);
            return Weights != null && Texels == texels && LayerSlices == layerSlices && Slots.AllResident == all
                   && Slots.Capacity == (all ? across * across : capacity);
        }

        /// <summary>
        /// The pool for a map of this many paint texels a side, this many layer slices and page slots (0: every level 0 page
        /// at its own index): made, its page table written on this list (or on the next <c>Slots.RecordTable</c> without
        /// one), when there is none like it. True when it was made just now, so nothing is in it yet.
        /// </summary>
        public bool Ensure(int texels, int layerSlices, int capacity, ID3D12GraphicsCommandList? cmd)
        {
            if (Matches(texels, layerSlices, capacity)) return false;
            bool all = capacity <= 0;
            int across = PagesAcrossFor(texels);
            int slots = all ? across * across : capacity;
            int weightSlices = WeightSlicesFor(slots, layerSlices);
            if (weightSlices > HeightPool.MaxSlices)
                throw new InvalidOperationException($"A paint pool of {slots} pages and {layerSlices} layer slices needs {weightSlices} " +
                                                    $"array slices: D3D12 allows {HeightPool.MaxSlices}");
            Release();
            Texels = texels;
            LayerSlices = layerSlices;

            var device = Engine.Device;
            Weights = device.CreateTexture2D(Format.R8G8B8A8_UNorm, SliceTexels, SliceTexels, weightSlices, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
            Weights.Name = "Terrain paint pool";
            int topSlices = (slots + 3) / 4;
            TopLayer = device.CreateTexture2D(Format.R8_UInt, SliceTexels, SliceTexels, topSlices, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
            TopLayer.Name = "Terrain paint top layer";

            (WeightsSrv, WeightsUav) = Views(Weights, Format.R8G8B8A8_UNorm, weightSlices);
            (TopSrv, TopUav) = Views(TopLayer, Format.R8_UInt, topSlices);

            Slots.Reset(across, all ? 0 : capacity, 0, all);
            if (cmd != null) Slots.RecordTable(cmd);
            return true;
        }

        private static (uint srv, uint uav) Views(ID3D12Resource texture, Format format, int slices)
        {
            var device = Engine.Device;
            uint srv = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(texture, new ShaderResourceViewDescription
            {
                Format = format,
                ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2DArray = new Texture2DArrayShaderResourceView { MostDetailedMip = 0, MipLevels = 1, FirstArraySlice = 0, ArraySize = (uint)slices },
            }, device.GetCpuHandle(srv));
            uint uav = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(texture, null, new UnorderedAccessViewDescription
            {
                Format = format,
                ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                Texture2DArray = new Texture2DArrayUnorderedAccessView { MipSlice = 0, FirstArraySlice = 0, ArraySize = (uint)slices },
            }, device.GetCpuHandle(uav));
            return (srv, uav);
        }

        /// <summary>A slot's quarter corner (texels) and the array slices of its first layer slice and of its top layer.</summary>
        public (int x, int y, int weightSlice, int topSlice) SlotPlace(int slot)
        {
            int q = slot & 3;
            return ((q & 1) * SlotTexels, (q >> 1) * SlotTexels, (slot >> 2) * LayerSlices, slot >> 2);
        }

        /// <summary>This frame's <c>TerrainPaintParams</c> (terrain_paint.hlsli): its bindless index, 0 before the pool is
        /// made. One buffer per frame in flight.</summary>
        public uint WriteParams(int layerCount)
        {
            if (Weights == null) return 0;
            int slot = Engine.FrameIndex % FrameCount;
            var buffer = _params[slot] ??= GraphicsBuffer.CreateUpload<Params>(1, mapped: true);
            unsafe
            {
                *buffer.WritePtr<Params>() = new Params
                {
                    Weights = WeightsSrv,
                    TopLayer = TopSrv,
                    PageTable = Slots.TableSrv,
                    PagesAcross = (uint)Slots.PagesAcross,
                    Texels = (uint)Texels,
                    LayerSlices = (uint)LayerSlices,
                    LayerCount = (uint)layerCount,
                };
            }
            return buffer.SrvIndex;
        }

        private void Release()
        {
            var device = Engine.Device;
            if (Weights != null)
            {
                device.DeferDispose(Weights);
                device.DeferReleaseBindlessIndex(WeightsSrv);
                device.DeferReleaseBindlessIndex(WeightsUav);
            }
            if (TopLayer != null)
            {
                device.DeferDispose(TopLayer);
                device.DeferReleaseBindlessIndex(TopSrv);
                device.DeferReleaseBindlessIndex(TopUav);
            }
            Slots.Release();
            Weights = TopLayer = null;
            WeightsSrv = WeightsUav = TopSrv = TopUav = 0;
            Texels = LayerSlices = 0;
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
