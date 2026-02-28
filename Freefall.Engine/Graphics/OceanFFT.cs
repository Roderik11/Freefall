using System;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// GPU-driven FFT ocean simulation (Tessendorf).
    /// Manages spectrum textures, compute PSOs, and per-frame dispatches.
    /// Produces displacement + slope Tex2DArrays sampled by ocean.fx.
    /// </summary>
    public class OceanFFT : IDisposable
    {
        public const int N = 512;
        private const int LOG_N = 9;
        private int _numBands;
        public int NumBands => _numBands;

        // ── Compute PSOs ──
        private ID3D12PipelineState? _initSpectrumPSO;
        private ID3D12PipelineState? _packConjugatePSO;
        private ID3D12PipelineState? _evolveSpectrumPSO;
        private ID3D12PipelineState? _horizontalFFTPSO;
        private ID3D12PipelineState? _verticalFFTPSO;
        private ID3D12PipelineState? _assembleMapsPSO;

        // ── GPU Textures ──
        private ID3D12Resource? _initialSpectrumTex;  // Tex2DArray N×N×4, RGBA16F
        private ID3D12Resource? _spectrumTex;          // Tex2DArray N×N×8, RGBA16F
        private ID3D12Resource? _displacementTex;      // Tex2DArray N×N×4, RGBA16F (xyz + foam)
        private ID3D12Resource? _slopeTex;             // Tex2DArray N×N×4, RG16F

        // ── Bindless descriptor indices ──
        private uint _initialSpectrumUAV;
        private uint _spectrumUAV;
        private uint _displacementUAV;
        private uint _slopeUAV;

        /// <summary>Bindless SRV for displacement Tex2DArray (xyz + foam in alpha).</summary>
        public uint DisplacementSRV { get; private set; }
        /// <summary>Bindless SRV for slope Tex2DArray (dh/dx, dh/dz).</summary>
        public uint SlopeSRV { get; private set; }

        // ── Spectrum parameters buffer ──
        private ID3D12Resource? _spectrumParamsBuffer;
        private uint _spectrumParamsSRV;

        // ── Length scales buffer (variable-length, one uint per band) ──
        private ID3D12Resource? _lengthScalesBuffer;
        private uint _lengthScalesSRV;

        // ── Constants buffer ──
        private ID3D12Resource? _constantsBuffer;
        private uint _constantsSRV;
        private IntPtr _constantsPtr;

        private GraphicsDevice _device;
        private bool _initialized;
        private bool _spectrumInitialized;

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
            int specCount = _numBands * 2;
            for (int i = 0; i < specCount; i++)
                _params.Spectrums[i] = newSpectrums[i];
            for (int i = 0; i < _numBands; i++)
                _params.LengthScales[i] = newLengthScales[i];

            // Re-upload spectrum params
            void* pSpec;
            _spectrumParamsBuffer.Map(0, null, &pSpec);
            var specSpan = new Span<SpectrumParameters>(pSpec, specCount);
            for (int i = 0; i < specCount; i++)
                specSpan[i] = newSpectrums[i];
            _spectrumParamsBuffer.Unmap(0);

            // Re-upload length scales
            void* pLS;
            _lengthScalesBuffer.Map(0, null, &pLS);
            var lsSpan = new Span<uint>(pLS, _numBands);
            for (int i = 0; i < _numBands; i++)
                lsSpan[i] = newLengthScales[i];
            _lengthScalesBuffer.Unmap(0);

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
            string basePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Resources", "Shaders");

            // Spectrum shaders
            string spectrumSource = File.ReadAllText(Path.Combine(basePath, "ocean_spectrum.hlsl"));
            var initShader = new Shader(spectrumSource, "CSInitSpectrum", "cs_6_6");
            _initSpectrumPSO = _device.CreateComputePipelineState(initShader.Bytecode);
            initShader.Dispose();

            var conjugateShader = new Shader(spectrumSource, "CSPackConjugate", "cs_6_6");
            _packConjugatePSO = _device.CreateComputePipelineState(conjugateShader.Bytecode);
            conjugateShader.Dispose();

            var evolveShader = new Shader(spectrumSource, "CSEvolveSpectrum", "cs_6_6");
            _evolveSpectrumPSO = _device.CreateComputePipelineState(evolveShader.Bytecode);
            evolveShader.Dispose();

            // FFT shaders
            string fftSource = File.ReadAllText(Path.Combine(basePath, "ocean_fft.hlsl"));
            var hfftShader = new Shader(fftSource, "CSHorizontalFFT", "cs_6_6");
            _horizontalFFTPSO = _device.CreateComputePipelineState(hfftShader.Bytecode);
            hfftShader.Dispose();

            var vfftShader = new Shader(fftSource, "CSVerticalFFT", "cs_6_6");
            _verticalFFTPSO = _device.CreateComputePipelineState(vfftShader.Bytecode);
            vfftShader.Dispose();

            var assembleShader = new Shader(fftSource, "CSAssembleMaps", "cs_6_6");
            _assembleMapsPSO = _device.CreateComputePipelineState(assembleShader.Bytecode);
            assembleShader.Dispose();

            Debug.Log("OceanFFT", "All compute shaders compiled");
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

        private unsafe void CreateBuffers()
        {
            int specCount = _numBands * 2;

            // Spectrum parameters: numBands*2 × SpectrumParameters
            int specSize = Marshal.SizeOf<SpectrumParameters>() * specCount;
            _spectrumParamsBuffer = _device.CreateUploadBuffer(specSize);
            _spectrumParamsSRV = _device.AllocateBindlessIndex();
            _device.CreateStructuredBufferSRV(_spectrumParamsBuffer, (uint)specCount, (uint)Marshal.SizeOf<SpectrumParameters>(), _spectrumParamsSRV);

            // Upload spectrum params
            void* pSpec;
            _spectrumParamsBuffer.Map(0, null, &pSpec);
            var specSpan = new Span<SpectrumParameters>(pSpec, specCount);
            for (int i = 0; i < specCount; i++)
                specSpan[i] = _params.Spectrums[i];
            _spectrumParamsBuffer.Unmap(0);

            // Length scales buffer: one uint per band
            int lsSize = sizeof(uint) * _numBands;
            lsSize = Math.Max(lsSize, 16); // D3D12 minimum buffer size
            _lengthScalesBuffer = _device.CreateUploadBuffer(lsSize);
            _lengthScalesSRV = _device.AllocateBindlessIndex();
            _device.CreateStructuredBufferSRV(_lengthScalesBuffer, (uint)_numBands, (uint)sizeof(uint), _lengthScalesSRV);

            void* pLS;
            _lengthScalesBuffer.Map(0, null, &pLS);
            var lsSpan = new Span<uint>(pLS, _numBands);
            for (int i = 0; i < _numBands; i++)
                lsSpan[i] = _params.LengthScales[i];
            _lengthScalesBuffer.Unmap(0);

            // Constants buffer: single OceanConstants struct
            int constSize = (Marshal.SizeOf<OceanConstants>() + 255) & ~255; // 256-byte aligned
            _constantsBuffer = _device.CreateUploadBuffer(constSize);
            _constantsSRV = _device.AllocateBindlessIndex();
            _device.CreateStructuredBufferSRV(_constantsBuffer, 1, (uint)Marshal.SizeOf<OceanConstants>(), _constantsSRV);

            // Persistently map
            void* pConst;
            _constantsBuffer.Map(0, null, &pConst);
            _constantsPtr = (IntPtr)pConst;
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
                LengthScalesSRV = _lengthScalesSRV,
            };
            *(OceanConstants*)_constantsPtr = constants;
        }

        /// <summary>
        /// Set compute root signature + push constants for ocean compute dispatches.
        /// Must be called before every compute dispatch.
        /// </summary>
        private void SetPushConstants(ID3D12GraphicsCommandList cmd)
        {
            cmd.SetComputeRootSignature(_device.GlobalRootSignature);
            cmd.SetComputeRoot32BitConstant(0, _initialSpectrumUAV, 0);
            cmd.SetComputeRoot32BitConstant(0, _spectrumUAV, 1);
            cmd.SetComputeRoot32BitConstant(0, _spectrumParamsSRV, 2);
            cmd.SetComputeRoot32BitConstant(0, _constantsSRV, 3);
            cmd.SetComputeRoot32BitConstant(0, _displacementUAV, 4);
            cmd.SetComputeRoot32BitConstant(0, _slopeUAV, 5);
        }

        /// <summary>
        /// Initialize the spectrum (one-time). Call after Create().
        /// </summary>
        public void InitSpectrum(ID3D12GraphicsCommandList cmd)
        {
            if (!_initialized) return;

            UpdateConstants(0, 0);

            uint groups = (uint)((N + 7) / 8);

            // CSInitSpectrum
            cmd.SetPipelineState(_initSpectrumPSO!);
            SetPushConstants(cmd);
            cmd.Dispatch(groups, groups, 1);

            // UAV barrier
            cmd.ResourceBarrierUnorderedAccessView(_initialSpectrumTex!);

            // CSPackConjugate
            cmd.SetPipelineState(_packConjugatePSO!);
            SetPushConstants(cmd);
            cmd.Dispatch(groups, groups, 1);

            // UAV barrier
            cmd.ResourceBarrierUnorderedAccessView(_initialSpectrumTex!);

            _spectrumInitialized = true;
            Debug.Log("OceanFFT", "Spectrum initialized");
        }

        /// <summary>
        /// Per-frame update: evolve spectrum → FFT → assemble maps.
        /// </summary>
        public void Update(ID3D12GraphicsCommandList cmd, float frameTime, float deltaTime)
        {
            if (!_initialized || !_spectrumInitialized) return;

            UpdateConstants(frameTime, deltaTime);

            uint groups8 = (uint)((N + 7) / 8);

            // 1. Evolve spectrum
            cmd.SetPipelineState(_evolveSpectrumPSO!);
            SetPushConstants(cmd);
            cmd.Dispatch(groups8, groups8, 1);

            cmd.ResourceBarrierUnorderedAccessView(_spectrumTex!);

            // 2. Horizontal FFT (N rows, SIZE threads each)
            cmd.SetPipelineState(_horizontalFFTPSO!);
            SetPushConstants(cmd);
            cmd.Dispatch(1, (uint)N, 1);

            cmd.ResourceBarrierUnorderedAccessView(_spectrumTex!);

            // 3. Vertical FFT (N columns, SIZE threads each)
            cmd.SetPipelineState(_verticalFFTPSO!);
            SetPushConstants(cmd);
            cmd.Dispatch(1, (uint)N, 1);

            cmd.ResourceBarrierUnorderedAccessView(_spectrumTex!);

            // 4. Assemble displacement + slope maps
            cmd.SetPipelineState(_assembleMapsPSO!);
            SetPushConstants(cmd);
            cmd.Dispatch(groups8, groups8, 1);

            // TODO: GenerateMips for displacement and slope textures
            // D3D12 doesn't have auto mip generation — need a downsample compute pass
            // For now, sample mip 0 only
        }

        public void Dispose()
        {
            _initialSpectrumTex?.Dispose();
            _spectrumTex?.Dispose();
            _displacementTex?.Dispose();
            _slopeTex?.Dispose();
            _spectrumParamsBuffer?.Dispose();
            _lengthScalesBuffer?.Dispose();
            _constantsBuffer?.Dispose();

            _initSpectrumPSO?.Dispose();
            _packConjugatePSO?.Dispose();
            _evolveSpectrumPSO?.Dispose();
            _horizontalFFTPSO?.Dispose();
            _verticalFFTPSO?.Dispose();
            _assembleMapsPSO?.Dispose();
        }
    }
}
