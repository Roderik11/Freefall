using System;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// What the water's screen-space reflections read (water_reflections_cs.hlsl, water.fx WaterScreenReflection): a
    /// pyramid of the nearest depth on the screen (1 / view depth, half size and down), which the water's trace walks
    /// cell by cell without stepping over thin things, and a pyramid of the composite snapshot (half size and down) for
    /// the blur of a rippled reflection. Made by <see cref="DeferredRenderer"/> after the composite is snapshot and before
    /// the forward pass, on frames when a <see cref="Components.WaterBody"/> draws and
    /// <c>EngineSettings.ScreenSpaceReflections</c> is on. Deterministic: no dither, no history.
    /// </summary>
    public sealed class WaterReflections : IDisposable
    {
        /// <summary>Mips of the colour pyramid: half size down to 1/128, a footprint of up to ~256 pixels.</summary>
        private const int ColourMips = 7;

        private const uint AllSubresources = 0xffffffff;   // D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES
        private static int _wantedTick = -1;

        private readonly ComputeShader _shader;
        private readonly int _kNearestFirst, _kNearest, _kAverage;
        private MipTexture? _nearest, _colour;
        private int _width, _height;
        private bool _firstFrame = true;
        private readonly ResourceBarrier[] _barriers = new ResourceBarrier[1];

        /// <summary>The nearest-depth pyramid (R32F, all mips), 0 before the first frame.</summary>
        public uint NearestDepthSrv => _nearest?.FullSrv ?? 0;
        /// <summary>Its mips (level 0 of the trace is the full-size linear depth, these are levels 1 and up).</summary>
        public uint NearestDepthMips => (uint)(_nearest?.Mips ?? 0);
        /// <summary>The colour pyramid (RGBA16F, all mips), 0 before the first frame.</summary>
        public uint ColourSrv => _colour?.FullSrv ?? 0;
        public uint ColourMipCount => (uint)(_colour?.Mips ?? 0);

        /// <summary>Asked for by a water that draws this frame (WaterBody.Draw).</summary>
        public static void Want() => _wantedTick = Engine.TickCount;

        /// <summary>A water drew this frame or the last (a frame's draws are queued before it renders).</summary>
        public static bool Wanted => Engine.TickCount - _wantedTick <= 1;

        public WaterReflections()
        {
            _shader = new ComputeShader("water_reflections_cs.hlsl");
            _kNearestFirst = _shader.FindKernel("CSNearestFirst");
            _kNearest = _shader.FindKernel("CSNearest");
            _kAverage = _shader.FindKernel("CSAverage");
        }

        /// <summary>Both pyramids, from the linear depth (shader-readable) and the composite snapshot (pixel-shader
        /// resource, as SnapshotComposite leaves it: put back so).</summary>
        public void Execute(ID3D12GraphicsCommandList list, DeferredRenderer renderer)
        {
            var snapshot = renderer.CompositeSnapshot;
            if (snapshot == null)
                return;
            var desc = renderer.DepthGBuffer.Native.Description;
            int width = (int)desc.Width, height = (int)desc.Height;
            Ensure(width, height);

            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);

            PixMarker.Begin(list, "Water reflections");
            var from = _firstFrame ? ResourceStates.Common : ResourceStates.AllShaderResource;
            _firstFrame = false;
            Transition(list, _nearest!.Texture, from, ResourceStates.UnorderedAccess, AllSubresources);
            Transition(list, _colour!.Texture, from, ResourceStates.UnorderedAccess, AllSubresources);
            Transition(list, snapshot.Native, ResourceStates.PixelShaderResource, ResourceStates.AllShaderResource, AllSubresources);

            Build(list, _nearest, _kNearestFirst, _kNearest, renderer.DepthGBuffer.BindlessIndex, width, height);
            Build(list, _colour, _kAverage, _kAverage, snapshot.BindlessIndex, width, height);

            Transition(list, snapshot.Native, ResourceStates.AllShaderResource, ResourceStates.PixelShaderResource, AllSubresources);
            PixMarker.End(list);
        }

        // Mip 0 from the source, each next from the one above; every mip goes to shader-read once written
        private void Build(ID3D12GraphicsCommandList list, MipTexture target, int firstKernel, int kernel, uint source, int width, int height)
        {
            int srcW = width, srcH = height;
            for (int mip = 0; mip < target.Mips; mip++)
            {
                int outW = Math.Max(1, srcW / 2), outH = Math.Max(1, srcH / 2);   // as D3D12 sizes the mips
                int k = mip == 0 ? firstKernel : kernel;
                _shader.SetPushConstant(k, "Source", mip == 0 ? source : target.MipSrvs[mip - 1]);
                _shader.SetPushConstant(k, "Output", target.MipUavs[mip]);
                _shader.SetPushConstant(k, "SourceWidth", (uint)srcW);
                _shader.SetPushConstant(k, "SourceHeight", (uint)srcH);
                _shader.SetPushConstant(k, "OutWidth", (uint)outW);
                _shader.SetPushConstant(k, "OutHeight", (uint)outH);
                _shader.Dispatch(k, list, ((uint)outW + 7) / 8, ((uint)outH + 7) / 8);
                Transition(list, target.Texture, ResourceStates.UnorderedAccess, ResourceStates.AllShaderResource, (uint)mip);
                srcW = outW;
                srcH = outH;
            }
        }

        private void Transition(ID3D12GraphicsCommandList list, ID3D12Resource resource, ResourceStates before, ResourceStates after, uint subresource)
        {
            _barriers[0] = new ResourceBarrier(new ResourceTransitionBarrier(resource, before, after, subresource));
            list.ResourceBarrier(_barriers);
        }

        private void Ensure(int width, int height)
        {
            if (_nearest != null && _width == width && _height == height)
                return;
            _nearest?.DeferDispose();
            _colour?.DeferDispose();
            int halfW = Math.Max(1, width / 2), halfH = Math.Max(1, height / 2);
            int fullChain = 1 + (int)Math.Floor(Math.Log2(Math.Max(halfW, halfH)));
            _nearest = new MipTexture(halfW, halfH, fullChain, Format.R32_Float);
            _colour = new MipTexture(halfW, halfH, Math.Min(ColourMips, fullChain), Format.R16G16B16A16_Float);
            _width = width;
            _height = height;
            _firstFrame = true;
        }

        public void Dispose()
        {
            _nearest?.DeferDispose();
            _colour?.DeferDispose();
            _nearest = _colour = null;
            _shader.Dispose();
        }

        /// <summary>A texture with a mip chain (D3D12's sizes: half the one above, rounded down; the last texel of a row or
        /// column also takes the odd one left over), a UAV and an SRV per mip and one SRV of them all.</summary>
        private sealed class MipTexture
        {
            public readonly ID3D12Resource Texture;
            public readonly int Mips;
            public readonly uint[] MipUavs, MipSrvs;
            public readonly uint FullSrv;

            public MipTexture(int width, int height, int mips, Format format)
            {
                var device = Engine.Device;
                Mips = mips;
                Texture = device.CreateTexture2D(format, width, height, 1, mips, ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
                MipUavs = new uint[mips];
                MipSrvs = new uint[mips];
                for (int i = 0; i < mips; i++)
                {
                    MipUavs[i] = device.AllocateBindlessIndex();
                    device.NativeDevice.CreateUnorderedAccessView(Texture, null, new UnorderedAccessViewDescription
                    {
                        Format = format,
                        ViewDimension = UnorderedAccessViewDimension.Texture2D,
                        Texture2D = new Texture2DUnorderedAccessView { MipSlice = (uint)i }
                    }, device.GetCpuHandle(MipUavs[i]));
                    MipSrvs[i] = device.AllocateBindlessIndex();
                    device.NativeDevice.CreateShaderResourceView(Texture, new ShaderResourceViewDescription
                    {
                        Format = format,
                        ViewDimension = ShaderResourceViewDimension.Texture2D,
                        Shader4ComponentMapping = ShaderComponentMapping.Default,
                        Texture2D = new Texture2DShaderResourceView { MostDetailedMip = (uint)i, MipLevels = 1 }
                    }, device.GetCpuHandle(MipSrvs[i]));
                }
                FullSrv = device.AllocateBindlessIndex();
                device.NativeDevice.CreateShaderResourceView(Texture, new ShaderResourceViewDescription
                {
                    Format = format,
                    ViewDimension = ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = (uint)mips }
                }, device.GetCpuHandle(FullSrv));
            }

            public void DeferDispose()
            {
                var device = Engine.Device;
                device.DeferDispose(Texture);
                foreach (var i in MipUavs) device.DeferReleaseBindlessIndex(i);
                foreach (var i in MipSrvs) device.DeferReleaseBindlessIndex(i);
                device.DeferReleaseBindlessIndex(FullSrv);
            }
        }
    }
}
