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
        private int _kHorizontalFFT, _kVerticalFFT, _kAssembleMaps, _kClearTallies;
        private int _kDownsample;
        private int _kGenerateNoise;

        // ── GPU Textures (raw — no Texture factory for Tex2DArray with per-mip UAVs) ──
        private ID3D12Resource? _initialSpectrumTex;  // Tex2DArray N×N×bands, RGBA16F
        private ID3D12Resource? _spectrumTex;          // Tex2DArray N×N×bands*2, RGBA16F
        private ID3D12Resource? _displacementTex;      // Tex2DArray N×N×bands, RGBA16F (xyz + foam)
        private ID3D12Resource? _slopeTex;             // Tex2DArray N×N×bands, RGBA16F slope moments (sx, sz, sx², sz²)

        // ── Bindless descriptor indices ──
        private uint _initialSpectrumUAV;
        private uint _spectrumUAV;
        private uint _displacementUAV;
        private uint _slopeUAV;

        /// <summary>Bindless SRV for displacement Tex2DArray (xyz + foam in alpha).</summary>
        public uint DisplacementSRV { get; private set; }
        /// <summary>Bindless SRV for the slope moments Tex2DArray (dh/dx, dh/dz and their squares; with mips, so a
        /// sample gives the mean slope and the slope variance of what it covers).</summary>
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

        // Texture states (a texture is never promoted to UnorderedAccess implicitly): the two spectra stay in
        // UnorderedAccess once out of Common (only the kernels touch them, as UAVs); the displacement and slope maps
        // are UnorderedAccess while the FFT and the mips write them and AllShaderResource for the water between.
        private bool _spectraInUav, _mapsReadable;

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
            public float LowCutoff;     // wavenumbers (rad/m) this layer covers: the wind bands share them out
            public float HighCutoff;
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
            public uint NoWhitecapBands;  // bit per band whose crests make no foam (a swell); was padding: 0 = all foam
            public uint WhitecapBand;     // the band whose whitecaps follow the wind (+1), tallied; 0 = none
            public float CapSqueeze;      // its crests whiten past this squeeze (1 − J)
            public float CapSpread;       // the spread of its squeeze; 0 = not tallied yet (FoamBias then)
            public float _pad2;
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
            public uint NoWhitecapBands; // bit per band whose crests make no foam
            public uint[] LengthScales; // one per band
            public SpectrumParameters[] Spectrums; // 2 per band
        }

        private OceanParams _params;

        // ── The wind's whitecaps (since 2026-11-23) ──
        // CSAssembleMaps tallies the whitecap band every frame: how squeezed its crests are (1 − J) and how much foam lies
        // on it. Read back FrameCount frames later, the tallies say what share of the band the foam covers; the share of
        // crests that whiten is steered so that it covers what the wind wants (Monahan), and their level is the squeeze that
        // share of the band's crests reach. So the whitecaps follow the wind whatever the waves' steepness (a storm's sea
        // grown long is less steep than one only grown tall), the choppiness or the foam's lifetime.
        // ocean_fft.hlsl's tallies: SQUEEZE_BINS of SQUEEZE_STEP from SQUEEZE_FROM, then FOAM_BINS over 0..1
        private const int SqueezeBins = 128, FoamBins = 16, Tallies = SqueezeBins + FoamBins;
        private const float SqueezeFrom = -1.2f, SqueezeStep = 0.025f;
        private const int TallyFrames = 3;
        // Foam this thick counts as covering the sea (bins of 1/16: from 0.5)
        private const int CoveringBin = FoamBins / 2;
        // How fast the share of crests that whiten follows the cover wanted (in e-folds a second, at most one an
        // update): slower than the foam's life and the crests' passing, or it chases the wave groups
        private const float ShareGain = 0.3f;
        private GraphicsBuffer? _tallies;
        private readonly GraphicsBuffer?[] _tallyReadback = new GraphicsBuffer?[TallyFrames];
        private readonly int[] _talliedBand = { -1, -1, -1 };   // the band each slot holds tallies of; −1 = none in flight
        private int _capBand = -1;
        private float _wantedCover;
        private float _crestShare = -1f;                         // < 0: not steered yet
        private float _capSqueeze, _capSpread;
        private float _measuredCover = -1f;

        /// <summary>The share of the wind sea the whitecaps cover now (foam of half thickness and more on the whitecap
        /// band), as last read back; −1 until the first reading.</summary>
        public float WhitecapCover => _measuredCover;

        /// <summary>The share of the whitecap band's crests that whiten, as steered toward the cover wanted.</summary>
        public float CrestShare => MathF.Max(_crestShare, 0f);

        /// <summary>The squeeze (1 − J) past which the whitecap band's crests whiten, and the spread of its squeeze.</summary>
        public float CapSqueeze => _capSqueeze;
        public float CapSpread => _capSpread;

        /// <summary>
        /// Update mutable runtime parameters — called each frame before dispatch.
        /// Constants-buffer values (depth, foam, choppiness) take effect immediately.
        ///   whitecapBand: the band whose whitecaps follow the wind (−1: none, every band foams by FoamBias);
        ///   whitecapCover: the share of the sea its whitecaps should cover (0..1)
        /// </summary>
        public void UpdateParams(float depth, float foamBias, float foamDecayRate,
            float foamThreshold, float foamAdd, Vector2 lambda, uint noWhitecapBands, int whitecapBand = -1, float whitecapCover = 0f)
        {
            _params.Depth = depth;
            _params.FoamBias = foamBias;
            _params.FoamDecayRate = foamDecayRate;
            _params.FoamThreshold = foamThreshold;
            _params.FoamAdd = foamAdd;
            _params.Lambda = lambda;
            _params.NoWhitecapBands = noWhitecapBands;
            if (whitecapBand != _capBand)
            {
                // Another band's tallies say nothing of this one's: its level and spread are measured anew
                _capBand = whitecapBand;
                _capSpread = 0f;
                _measuredCover = -1f;
            }
            _wantedCover = Math.Clamp(whitecapCover, 0f, 1f);
        }

        // The tallies of a frame (slot), read back: the whitecap band's squeeze and foam
        private unsafe void ReadTallies(uint* t, float dt)
        {
            long total = 0;
            double sum = 0, sum2 = 0;
            for (int b = 0; b < SqueezeBins; b++)
            {
                double s = SqueezeFrom + (b + 0.5) * SqueezeStep;
                total += t[b];
                sum += t[b] * s;
                sum2 += t[b] * s * s;
            }
            if (total == 0) return;
            double mean = sum / total;
            float spread = (float)Math.Sqrt(Math.Max(sum2 / total - mean * mean, 1e-8));

            long foamTotal = 0, covered = 0;
            for (int b = 0; b < FoamBins; b++)
            {
                foamTotal += t[SqueezeBins + b];
                if (b >= CoveringBin) covered += t[SqueezeBins + b];
            }
            float cover = foamTotal > 0 ? (float)covered / foamTotal : 0f;

            // The share of crests that whiten, steered toward the cover wanted (in e-folds of the ratio, so a sea a tenth
            // as white as wanted catches up as fast as one ten times too white). It starts at a third of the cover: the
            // foam lies behind the crests that make it for longer than they stay squeezed
            bool first = _capSpread <= 0f;
            float wanted = _wantedCover;
            if (wanted <= 1e-6f)
                _crestShare = 0f;
            else
            {
                if (_crestShare <= 0f) _crestShare = wanted / 3f;
                float ratio = wanted / MathF.Max(cover, wanted * 0.05f);
                float step = Math.Clamp(MathF.Log(ratio) * ShareGain * dt, -1f, 1f);
                _crestShare = Math.Clamp(_crestShare * MathF.Exp(step), 1e-5f, 0.5f);
            }

            // The squeeze that share of the crests reach: counted down from the most squeezed, spread evenly in a bin
            float level = 10f;
            if (_crestShare > 0f)
            {
                double target = _crestShare * (double)total;
                long above = 0;
                level = SqueezeFrom;
                for (int b = SqueezeBins - 1; b >= 0; b--)
                {
                    if (above + t[b] >= target && t[b] > 0)
                    {
                        float inside = (float)((target - above) / t[b]);
                        level = SqueezeFrom + (b + 1 - inside) * SqueezeStep;
                        break;
                    }
                    above += t[b];
                }
            }

            // Smoothed over about half a second: one frame's tallies of a tile vary with the wave groups in it
            float ease = first || _crestShare <= 0f ? 1f : 1f - MathF.Exp(-dt / 0.5f);
            _capSqueeze += (level - _capSqueeze) * ease;
            _capSpread += (spread - _capSpread) * ease;
            _measuredCover = _measuredCover < 0f ? cover : _measuredCover + (cover - _measuredCover) * (1f - MathF.Exp(-dt / 2f));
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
            _kClearTallies = _fftShader.FindKernel("CSClearTallies");

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
            _spectraInUav = _mapsReadable = false;

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

            // Slope moments output: RGBA16F (sx, sz, sx², sz²), N×N, numBands slices, with mips. The mips average all
            // four, so the water shaders get the mean slope and the variance of whatever a sample covers.
            _slopeTex = _device.CreateTexture2D(
                Format.R16G16B16A16_Float, N, N, _numBands, mipCount,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _slopeUAV = _device.AllocateBindlessIndex();
            CreateTex2DArrayUAV(_slopeTex, Format.R16G16B16A16_Float, _numBands, _slopeUAV);

            SlopeSRV = _device.AllocateBindlessIndex();
            CreateTex2DArraySRV(_slopeTex, Format.R16G16B16A16_Float, _numBands, mipCount, SlopeSRV);

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
                CreateTex2DArrayMipUAV(_slopeTex, Format.R16G16B16A16_Float, _numBands, m, _slopeMipUAVs[m]);
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

            // The whitecap band's tallies and their readbacks
            _tallies = GraphicsBuffer.CreateStructured<uint>(Tallies, uav: true);
            for (int i = 0; i < TallyFrames; i++)
                _tallyReadback[i] = GraphicsBuffer.CreateReadback<uint>(Tallies);
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
                NoWhitecapBands = _params.NoWhitecapBands,
                WhitecapBand = _capBand >= 0 && _capBand < _numBands ? (uint)(_capBand + 1) : 0u,
                CapSqueeze = _capSqueeze,
                CapSpread = _capSpread,
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
            _fftShader.SetPushConstant("WhitecapTallies", _tallies?.UavIndex ?? 0u);
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

            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            cmd.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            cmd.SetComputeRootSignature(_device.GlobalRootSignature);

            if (!_spectraInUav)
            {
                cmd.ResourceBarrierTransition(_initialSpectrumTex!, ResourceStates.Common, ResourceStates.UnorderedAccess);
                cmd.ResourceBarrierTransition(_spectrumTex!, ResourceStates.Common, ResourceStates.UnorderedAccess);
                _spectraInUav = true;
            }

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
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            cmd.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            cmd.SetComputeRootSignature(_device.GlobalRootSignature);

            _noiseShader!.SetPushConstant("Output", _noiseUAV);
            _noiseShader.SetPushConstant("TexSize", (uint)NoiseSize);

            uint groups = (uint)((NoiseSize + 7) / 8);
            cmd.ResourceBarrierTransition(_noiseTex, ResourceStates.Common, ResourceStates.UnorderedAccess);
            _noiseShader.Dispatch(_kGenerateNoise, cmd, groups, groups);
            cmd.ResourceBarrierTransition(_noiseTex, ResourceStates.UnorderedAccess, ResourceStates.AllShaderResource);
            Debug.Log("OceanFFT", "Noise texture generated");
        }

        /// <summary>
        /// Per-frame update: evolve spectrum → FFT → assemble maps.
        /// </summary>
        public unsafe void Update(ID3D12GraphicsCommandList cmd, float frameTime, float deltaTime)
        {
            if (!_initialized || !_spectrumInitialized) return;

            // The tallies this slot took TallyFrames frames ago, if they are of the band that whitens now
            int slot = Engine.FrameIndex % TallyFrames;
            if (_talliedBand[slot] >= 0 && _talliedBand[slot] == _capBand && _tallyReadback[slot] is { } readback)
                ReadTallies(readback.ReadPtr<uint>(), deltaTime);
            _talliedBand[slot] = -1;

            UpdateConstants(frameTime, deltaTime);
            BindSharedPushConstants();

            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            cmd.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            cmd.SetComputeRootSignature(_device.GlobalRootSignature);

            uint groups8 = (uint)((N + 7) / 8);

            var mapsFrom = _mapsReadable ? ResourceStates.AllShaderResource : ResourceStates.Common;
            cmd.ResourceBarrierTransition(_displacementTex!, mapsFrom, ResourceStates.UnorderedAccess);
            cmd.ResourceBarrierTransition(_slopeTex!, mapsFrom, ResourceStates.UnorderedAccess);

            // 1. Evolve spectrum
            _spectrumShader!.Dispatch(_kEvolveSpectrum, cmd, groups8, groups8);
            cmd.ResourceBarrierUnorderedAccessView(_spectrumTex!);

            // 2. Horizontal FFT (N rows, SIZE threads each)
            _fftShader!.Dispatch(_kHorizontalFFT, cmd, 1, (uint)N);
            cmd.ResourceBarrierUnorderedAccessView(_spectrumTex!);

            // 3. Vertical FFT (N columns, SIZE threads each)
            _fftShader.Dispatch(_kVerticalFFT, cmd, 1, (uint)N);
            cmd.ResourceBarrierUnorderedAccessView(_spectrumTex!);

            // 4. Assemble displacement + slope maps (and tally the whitecap band)
            bool tally = _capBand >= 0 && _capBand < _numBands && _tallies != null;
            if (tally)
            {
                _tallies!.Transition(cmd, ResourceStates.UnorderedAccess);
                _fftShader.Dispatch(_kClearTallies, cmd, (uint)(Tallies + 63) / 64);
                _tallies.UAVBarrier(cmd);
            }
            _fftShader.Dispatch(_kAssembleMaps, cmd, groups8, groups8);
            cmd.ResourceBarrierUnorderedAccessView(_displacementTex!);
            cmd.ResourceBarrierUnorderedAccessView(_slopeTex!);
            if (tally)
            {
                _tallies!.Transition(cmd, ResourceStates.CopySource);
                cmd.CopyBufferRegion(_tallyReadback[slot]!.Native, 0, _tallies.Native, 0, (ulong)(Tallies * sizeof(uint)));
                _talliedBand[slot] = _capBand;
            }

            // 5. Generate mip chain for displacement and slope textures
            GenerateMips(cmd);

            // The water reads them through their SRVs
            cmd.ResourceBarrierTransition(_displacementTex!, ResourceStates.UnorderedAccess, ResourceStates.AllShaderResource);
            cmd.ResourceBarrierTransition(_slopeTex!, ResourceStates.UnorderedAccess, ResourceStates.AllShaderResource);
            _mapsReadable = true;
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

                // Downsample the slope moments (RGBA16F: averaging s and s² keeps the variance of what a texel covers)
                _mipShader.SetPushConstant(_kDownsample, "SrcMip", _slopeMipUAVs[mip - 1]);
                _mipShader.SetPushConstant(_kDownsample, "DstMip", _slopeMipUAVs[mip]);
                _mipShader.SetPushConstant(_kDownsample, "MipTexelSize", (uint)dstSize);
                _mipShader.SetPushConstant(_kDownsample, "NumSlices", (uint)_numBands);
                _mipShader.SetPushConstant(_kDownsample, "IsRG16F", 0u);
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
            _tallies?.Dispose();
            foreach (var r in _tallyReadback) r?.Dispose();

            _spectrumShader?.Dispose();
            _fftShader?.Dispose();
            _mipShader?.Dispose();
            _noiseShader?.Dispose();
        }
    }
}
