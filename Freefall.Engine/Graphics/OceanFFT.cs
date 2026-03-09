using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// GPU-driven FFT ocean simulation (Tessendorf).
    /// Uses ComputeShader + GraphicsBuffer abstractions for compute dispatch.
    /// Produces displacement + slope Tex2DArrays sampled by ocean.fx.
    /// </summary>
    public class OceanFFT : IDisposable
    {
        public const int N = 512;
        private int _numBands;
        public int NumBands => _numBands;

        // ── Compute shaders (multi-kernel, reflection-based binding) ──
        private ComputeShader? _spectrumShader;   // ocean_spectrum.hlsl
        private ComputeShader? _fftShader;        // ocean_fft.hlsl
        private ComputeShader? _mipShader;        // ocean_mipgen.hlsl
        private ComputeShader? _noiseShader;      // ocean_noise.hlsl

        // Cached kernel indices
        private int _kInitSpectrum, _kPackConjugate, _kEvolveSpectrum;
        private int _kHorizontalFFT, _kVerticalFFT, _kAssembleMaps;
        private int _kDownsample;
        private int _kGenerateNoise;

        // ── GPU Textures (raw — no Texture factory for Tex2DArray with per-mip UAVs) ──
        private ID3D12Resource? _initialSpectrumTex;  // Tex2DArray N×N×bands, RGBA16F
        private ID3D12Resource? _spectrumTex;          // Tex2DArray N×N×bands*2, RGBA16F
        private ID3D12Resource? _displacementTex;      // Tex2DArray N×N×bands, RGBA16F (xyz + foam)
        private ID3D12Resource? _slopeTex;             // Tex2DArray N×N×bands, RG16F

        // ── Bindless descriptor indices ──
        private uint _initialSpectrumUAV;
        private uint _spectrumUAV;
        private uint _displacementUAV;
        private uint _slopeUAV;

        /// <summary>Bindless SRV for displacement Tex2DArray (xyz + foam in alpha).</summary>
        public uint DisplacementSRV { get; private set; }
        /// <summary>Bindless SRV for slope Tex2DArray (dh/dx, dh/dz).</summary>
        public uint SlopeSRV { get; private set; }
        /// <summary>Bindless SRV for noise texture (Perlin+Worley, RGBA).</summary>
        public uint NoiseSRV { get; private set; }

        private ID3D12Resource? _noiseTex;
        private uint _noiseUAV;
        private const int NoiseSize = 256;

        // Per-mip UAVs and SRVs for mipmap generation
        private uint[] _displacementMipUAVs = null!;
        private uint[] _slopeMipUAVs = null!;
        private int _mipCount;

        // ── Buffers (GraphicsBuffer abstractions) ──
        private GraphicsBuffer? _spectrumParamsBuffer;  // Upload, mapped
        private GraphicsBuffer? _lengthScalesBuffer;    // Upload, mapped
        private GraphicsBuffer? _constantsBuffer;       // Upload, mapped (SRV for HLSL StructuredBuffer)

        private GraphicsDevice _device = null!;
        private bool _initialized;
        private bool _spectrumInitialized;

        // Cached descriptor heap array for compute dispatches
        private ID3D12DescriptorHeap[]? _cachedSrvHeapArray;

        [StructLayout(LayoutKind.Sequential)]
        public struct SpectrumParameters
        {
            public float Scale;
            public float Angle;
            public float SpreadBlend;
            public float Swell;
            public float Alpha;
            public float PeakOmega;
            public float Gamma;
            public float ShortWavesFade;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct OceanConstants
        {
            public float FrameTime;
            public float DeltaTime;
            public float Gravity;
            public float Depth;
            public float RepeatTime;
            public float LowCutoff;
            public float HighCutoff;
            public uint N;
            public uint Seed;
            public Vector2 Lambda;
            public float FoamBias;
            public float FoamDecayRate;
            public float FoamThreshold;
            public float FoamAdd;
            public uint NumBands;
            public uint LengthScalesSRV;  // bindless SRV to uint[] buffer
            public uint _pad0, _pad1;
        }

        /// <summary>
        /// User-facing parameters for ocean configuration.
        /// </summary>
        public struct OceanParams
        {
            public float Gravity;
            public float Depth;
            public float RepeatTime;
            public float LowCutoff;
            public float HighCutoff;
            public int Seed;
            public Vector2 Lambda;      // Choppiness (horizontal displacement scale)
            public float FoamBias;
            public float FoamDecayRate;
            public float FoamThreshold;
            public float FoamAdd;
            public uint[] LengthScales; // one per band
            public SpectrumParameters[] Spectrums; // 2 per band
        }

        private OceanParams _params;

        /// <summary>
        /// Update mutable runtime parameters — called each frame before dispatch.
        /// Constants-buffer values (depth, foam, choppiness) take effect immediately.
        /// </summary>
        public void UpdateParams(float depth, float foamBias, float foamDecayRate,
            float foamThreshold, float foamAdd, Vector2 lambda)
        {
            _params.Depth = depth;
            _params.FoamBias = foamBias;
            _params.FoamDecayRate = foamDecayRate;
            _params.FoamThreshold = foamThreshold;
            _params.FoamAdd = foamAdd;
            _params.Lambda = lambda;
        }

        /// <summary>
        /// Force spectrum re-initialization on the next frame.
        /// Re-uploads spectrum params and length scales to GPU.
        /// </summary>
        public unsafe void RequestSpectrumReinit(SpectrumParameters[] newSpectrums, uint[] newLengthScales)
        {
            if (!_initialized) return;

            int specCount = _numBands * 2;
            for (int i = 0; i < specCount; i++)
                _params.Spectrums[i] = newSpectrums[i];
            for (int i = 0; i < _numBands; i++)
                _params.LengthScales[i] = newLengthScales[i];

            // Re-upload spectrum params
            _spectrumParamsBuffer!.Upload<SpectrumParameters>(newSpectrums.AsSpan(0, specCount));

            // Re-upload length scales
            _lengthScalesBuffer!.Upload<uint>(newLengthScales.AsSpan(0, _numBands));

            _spectrumInitialized = false;
        }

        public void Create(GraphicsDevice device, OceanParams oceanParams)
        {
            _device = device;
            _params = oceanParams;
            _numBands = oceanParams.LengthScales.Length;

            CompileShaders();
            CreateTextures();
            CreateBuffers();

            _initialized = true;
            _spectrumInitialized = false;

            Debug.Log("OceanFFT", $"Created: {N}x{N}, {NumBands} bands, DisplacementSRV={DisplacementSRV}, SlopeSRV={SlopeSRV}");
        }

        private void CompileShaders()
        {
            // Spectrum shader: CSInitSpectrum, CSPackConjugate, CSEvolveSpectrum
            _spectrumShader = new ComputeShader("ocean_spectrum.hlsl");
            _kInitSpectrum = _spectrumShader.FindKernel("CSInitSpectrum");
            _kPackConjugate = _spectrumShader.FindKernel("CSPackConjugate");
            _kEvolveSpectrum = _spectrumShader.FindKernel("CSEvolveSpectrum");

            // FFT shader: CSHorizontalFFT, CSVerticalFFT, CSAssembleMaps
            _fftShader = new ComputeShader("ocean_fft.hlsl");
            _kHorizontalFFT = _fftShader.FindKernel("CSHorizontalFFT");
            _kVerticalFFT = _fftShader.FindKernel("CSVerticalFFT");
            _kAssembleMaps = _fftShader.FindKernel("CSAssembleMaps");

            // Mip downsample shader
            _mipShader = new ComputeShader("ocean_mipgen.hlsl");
            _kDownsample = _mipShader.FindKernel("CSDownsample");

            // Noise generation shader
            _noiseShader = new ComputeShader("ocean_noise.hlsl");
            _kGenerateNoise = _noiseShader.FindKernel("CSGenerateNoise");
        }

        private void CreateTextures()
        {
            int spectrumSlices = _numBands * 2;

            // Initial spectrum: RGBA16F, N×N, numBands slices (packed H0+H0conj)
            _initialSpectrumTex = _device.CreateTexture2D(
                Format.R16G16B16A16_Float, N, N, _numBands, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _initialSpectrumUAV = _device.AllocateBindlessIndex();
            CreateTex2DArrayUAV(_initialSpectrumTex, Format.R16G16B16A16_Float, _numBands, _initialSpectrumUAV);

            // Evolved spectrum: RGBA16F, N×N, numBands*2 slices (2 per band)
            _spectrumTex = _device.CreateTexture2D(
                Format.R16G16B16A16_Float, N, N, spectrumSlices, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _spectrumUAV = _device.AllocateBindlessIndex();
            CreateTex2DArrayUAV(_spectrumTex, Format.R16G16B16A16_Float, spectrumSlices, _spectrumUAV);

            // Displacement output: RGBA16F, N×N, numBands slices (xyz + foam), with mips
            int mipCount = 1 + (int)Math.Floor(Math.Log2(N));
            _displacementTex = _device.CreateTexture2D(
                Format.R16G16B16A16_Float, N, N, _numBands, mipCount,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _displacementUAV = _device.AllocateBindlessIndex();
            CreateTex2DArrayUAV(_displacementTex, Format.R16G16B16A16_Float, _numBands, _displacementUAV);

            DisplacementSRV = _device.AllocateBindlessIndex();
            CreateTex2DArraySRV(_displacementTex, Format.R16G16B16A16_Float, _numBands, mipCount, DisplacementSRV);

            // Slope output: RG16F, N×N, numBands slices, with mips
            _slopeTex = _device.CreateTexture2D(
                Format.R16G16_Float, N, N, _numBands, mipCount,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _slopeUAV = _device.AllocateBindlessIndex();
            CreateTex2DArrayUAV(_slopeTex, Format.R16G16_Float, _numBands, _slopeUAV);

            SlopeSRV = _device.AllocateBindlessIndex();
            CreateTex2DArraySRV(_slopeTex, Format.R16G16_Float, _numBands, mipCount, SlopeSRV);

            // Create per-mip UAVs for downsample chain
            _mipCount = mipCount;
            _displacementMipUAVs = new uint[mipCount];
            _slopeMipUAVs = new uint[mipCount];

            // Mip 0 UAVs already exist
            _displacementMipUAVs[0] = _displacementUAV;
            _slopeMipUAVs[0] = _slopeUAV;

            for (int m = 1; m < mipCount; m++)
            {
                _displacementMipUAVs[m] = _device.AllocateBindlessIndex();
                CreateTex2DArrayMipUAV(_displacementTex, Format.R16G16B16A16_Float, _numBands, m, _displacementMipUAVs[m]);
                _slopeMipUAVs[m] = _device.AllocateBindlessIndex();
                CreateTex2DArrayMipUAV(_slopeTex, Format.R16G16_Float, _numBands, m, _slopeMipUAVs[m]);
            }
        }

        private void CreateTex2DArrayUAV(ID3D12Resource tex, Format format, int arraySize, uint bindlessIdx)
        {
            var uavDesc = new UnorderedAccessViewDescription
            {
                Format = format,
                ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                Texture2DArray = new Texture2DArrayUnorderedAccessView
                {
                    MipSlice = 0,
                    FirstArraySlice = 0,
                    ArraySize = (uint)arraySize
                }
            };
            _device.NativeDevice.CreateUnorderedAccessView(tex, null, uavDesc, _device.GetCpuHandle(bindlessIdx));
        }

        private void CreateTex2DArraySRV(ID3D12Resource tex, Format format, int arraySize, int mipLevels, uint bindlessIdx)
        {
            var srvDesc = new ShaderResourceViewDescription
            {
                Format = format,
                ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2DArray = new Texture2DArrayShaderResourceView
                {
                    MostDetailedMip = 0,
                    MipLevels = (uint)mipLevels,
                    FirstArraySlice = 0,
                    ArraySize = (uint)arraySize
                }
            };
            _device.NativeDevice.CreateShaderResourceView(tex, srvDesc, _device.GetCpuHandle(bindlessIdx));
        }

        private void CreateTex2DArrayMipUAV(ID3D12Resource tex, Format format, int arraySize, int mipLevel, uint bindlessIdx)
        {
            var uavDesc = new UnorderedAccessViewDescription
            {
                Format = format,
                ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                Texture2DArray = new Texture2DArrayUnorderedAccessView
                {
                    MipSlice = (uint)mipLevel,
                    FirstArraySlice = 0,
                    ArraySize = (uint)arraySize
                }
            };
            _device.NativeDevice.CreateUnorderedAccessView(tex, null, uavDesc, _device.GetCpuHandle(bindlessIdx));
        }

        private unsafe void CreateBuffers()
        {
            int specCount = _numBands * 2;

            // Spectrum parameters: upload buffer with SRV, mapped for fast re-upload
            _spectrumParamsBuffer = GraphicsBuffer.CreateUpload<SpectrumParameters>(specCount, mapped: true);
            var pSpec = _spectrumParamsBuffer.WritePtr<SpectrumParameters>();
            for (int i = 0; i < specCount; i++)
                pSpec[i] = _params.Spectrums[i];

            // Length scales: upload buffer with SRV, mapped
            int lsCount = Math.Max(_numBands, 4); // D3D12 minimum buffer size guard
            _lengthScalesBuffer = GraphicsBuffer.CreateUpload<uint>(lsCount, mapped: true);
            var pLS = _lengthScalesBuffer.WritePtr<uint>();
            for (int i = 0; i < _numBands; i++)
                pLS[i] = _params.LengthScales[i];

            // Constants: upload buffer with SRV (HLSL reads as StructuredBuffer<OceanConstants>)
            _constantsBuffer = GraphicsBuffer.CreateUpload<OceanConstants>(1, mapped: true);
        }

        private unsafe void UpdateConstants(float frameTime, float deltaTime)
        {
            var constants = new OceanConstants
            {
                FrameTime = frameTime,
                DeltaTime = deltaTime,
                Gravity = _params.Gravity,
                Depth = _params.Depth,
                RepeatTime = _params.RepeatTime,
                LowCutoff = _params.LowCutoff,
                HighCutoff = _params.HighCutoff,
                N = N,
                Seed = (uint)_params.Seed,
                Lambda = _params.Lambda,
                FoamBias = _params.FoamBias,
                FoamDecayRate = _params.FoamDecayRate,
                FoamThreshold = _params.FoamThreshold,
                FoamAdd = _params.FoamAdd,
                NumBands = (uint)_numBands,
                LengthScalesSRV = _lengthScalesBuffer!.SrvIndex,
            };
            *_constantsBuffer!.WritePtr<OceanConstants>() = constants;
        }

        /// <summary>
        /// Bind shared push constants across spectrum and FFT shaders.
        /// Call once before a series of dispatches.
        /// </summary>
        private void BindSharedPushConstants()
        {
            // Spectrum shader: all 3 kernels share the same 4 push constants
            _spectrumShader!.SetPushConstant("InitialSpectrum", _initialSpectrumUAV);
            _spectrumShader.SetPushConstant("Spectrum", _spectrumUAV);
            _spectrumShader.SetPushConstant("SpectrumParams", _spectrumParamsBuffer!.SrvIndex);
            _spectrumShader.SetPushConstant("OceanConstants", _constantsBuffer!.SrvIndex);

            // FFT shader: 6 push constants (4 shared + displacement + slope)
            _fftShader!.SetPushConstant("InitialSpectrum", _initialSpectrumUAV);
            _fftShader.SetPushConstant("Spectrum", _spectrumUAV);
            _fftShader.SetPushConstant("SpectrumParams", _spectrumParamsBuffer.SrvIndex);
            _fftShader.SetPushConstant("OceanConstants", _constantsBuffer.SrvIndex);
            _fftShader.SetPushConstant("Displacement", _displacementUAV);
            _fftShader.SetPushConstant("Slope", _slopeUAV);
        }

        /// <summary>
        /// Initialize the spectrum (one-time). Call after Create().
        /// </summary>
        public void InitSpectrum(ID3D12GraphicsCommandList cmd)
        {
            if (!_initialized) return;
            if (_spectrumInitialized) return;

            UpdateConstants(0, 0);
            BindSharedPushConstants();

            uint groups = (uint)((N + 7) / 8);

            cmd.SetComputeRootSignature(_device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            cmd.SetDescriptorHeaps(1, _cachedSrvHeapArray);

            // CSInitSpectrum
            _spectrumShader!.Dispatch(_kInitSpectrum, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_initialSpectrumTex!);

            // CSPackConjugate
            _spectrumShader.Dispatch(_kPackConjugate, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_initialSpectrumTex!);

            _spectrumInitialized = true;
        }

        /// <summary>
        /// Generate noise texture (one-time). Call once after Create().
        /// </summary>
        public void GenerateNoise(ID3D12GraphicsCommandList cmd)
        {
            if (_noiseTex != null) return; // already generated

            // Create RGBA8 noise texture
            _noiseTex = _device.CreateTexture2D(
                Format.R8G8B8A8_UNorm, NoiseSize, NoiseSize, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _noiseUAV = _device.AllocateBindlessIndex();
            var uavDesc = new UnorderedAccessViewDescription
            {
                Format = Format.R8G8B8A8_UNorm,
                ViewDimension = UnorderedAccessViewDimension.Texture2D,
                Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
            };
            _device.NativeDevice.CreateUnorderedAccessView(_noiseTex, null, uavDesc, _device.GetCpuHandle(_noiseUAV));

            NoiseSRV = _device.AllocateBindlessIndex();
            var srvDesc = new ShaderResourceViewDescription
            {
                Format = Format.R8G8B8A8_UNorm,
                ViewDimension = ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
            };
            _device.NativeDevice.CreateShaderResourceView(_noiseTex, srvDesc, _device.GetCpuHandle(NoiseSRV));

            // Dispatch noise generation via ComputeShader
            cmd.SetComputeRootSignature(_device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            cmd.SetDescriptorHeaps(1, _cachedSrvHeapArray);

            _noiseShader!.SetPushConstant("Output", _noiseUAV);
            _noiseShader.SetPushConstant("TexSize", (uint)NoiseSize);

            uint groups = (uint)((NoiseSize + 7) / 8);
            _noiseShader.Dispatch(_kGenerateNoise, cmd, groups, groups);

            cmd.ResourceBarrierUnorderedAccessView(_noiseTex);
            Debug.Log("OceanFFT", "Noise texture generated");
        }

        /// <summary>
        /// Per-frame update: evolve spectrum → FFT → assemble maps.
        /// </summary>
        public void Update(ID3D12GraphicsCommandList cmd, float frameTime, float deltaTime)
        {
            if (!_initialized || !_spectrumInitialized) return;

            UpdateConstants(frameTime, deltaTime);
            BindSharedPushConstants();

            cmd.SetComputeRootSignature(_device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            cmd.SetDescriptorHeaps(1, _cachedSrvHeapArray);

            uint groups8 = (uint)((N + 7) / 8);

            // 1. Evolve spectrum
            _spectrumShader!.Dispatch(_kEvolveSpectrum, cmd, groups8, groups8);
            cmd.ResourceBarrierUnorderedAccessView(_spectrumTex!);

            // 2. Horizontal FFT (N rows, SIZE threads each)
            _fftShader!.Dispatch(_kHorizontalFFT, cmd, 1, (uint)N);
            cmd.ResourceBarrierUnorderedAccessView(_spectrumTex!);

            // 3. Vertical FFT (N columns, SIZE threads each)
            _fftShader.Dispatch(_kVerticalFFT, cmd, 1, (uint)N);
            cmd.ResourceBarrierUnorderedAccessView(_spectrumTex!);

            // 4. Assemble displacement + slope maps
            _fftShader.Dispatch(_kAssembleMaps, cmd, groups8, groups8);
            cmd.ResourceBarrierUnorderedAccessView(_displacementTex!);
            cmd.ResourceBarrierUnorderedAccessView(_slopeTex!);

            // 5. Generate mip chain for displacement and slope textures
            GenerateMips(cmd);
        }

        private void GenerateMips(ID3D12GraphicsCommandList cmd)
        {
            int srcSize = N;
            for (int mip = 1; mip < _mipCount; mip++)
            {
                int dstSize = srcSize / 2;
                if (dstSize < 1) break;

                uint groups = (uint)((dstSize + 7) / 8);

                // Downsample displacement (RGBA16F)
                _mipShader!.SetPushConstant(_kDownsample, "SrcMip", _displacementMipUAVs[mip - 1]);
                _mipShader.SetPushConstant(_kDownsample, "DstMip", _displacementMipUAVs[mip]);
                _mipShader.SetPushConstant(_kDownsample, "MipTexelSize", (uint)dstSize);
                _mipShader.SetPushConstant(_kDownsample, "NumSlices", (uint)_numBands);
                _mipShader.SetPushConstant(_kDownsample, "IsRG16F", 0u);
                _mipShader.Dispatch(_kDownsample, cmd, groups, groups);
                cmd.ResourceBarrierUnorderedAccessView(_displacementTex!);

                // Downsample slope (RG16F)
                _mipShader.SetPushConstant(_kDownsample, "SrcMip", _slopeMipUAVs[mip - 1]);
                _mipShader.SetPushConstant(_kDownsample, "DstMip", _slopeMipUAVs[mip]);
                _mipShader.SetPushConstant(_kDownsample, "MipTexelSize", (uint)dstSize);
                _mipShader.SetPushConstant(_kDownsample, "NumSlices", (uint)_numBands);
                _mipShader.SetPushConstant(_kDownsample, "IsRG16F", 1u);
                _mipShader.Dispatch(_kDownsample, cmd, groups, groups);
                cmd.ResourceBarrierUnorderedAccessView(_slopeTex!);

                srcSize = dstSize;
            }
        }

        public void Dispose()
        {
            _initialSpectrumTex?.Dispose();
            _spectrumTex?.Dispose();
            _displacementTex?.Dispose();
            _slopeTex?.Dispose();
            _noiseTex?.Dispose();

            _spectrumParamsBuffer?.Dispose();
            _lengthScalesBuffer?.Dispose();
            _constantsBuffer?.Dispose();

            _spectrumShader?.Dispose();
            _fftShader?.Dispose();
            _mipShader?.Dispose();
            _noiseShader?.Dispose();
        }
    }
}
