using System;
using System.Numerics;
using System.Collections.Generic;
using Freefall.Assets;
using Vortice.Direct3D12;
using Vortice.Direct3D12.Shader;
using Vortice.DXGI;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// Material texture indices + scalar properties for bindless lookup — matches HLSL MaterialData (80 bytes)
    /// </summary>
    public struct MaterialData
    {
        public uint AlbedoIdx;
        public uint NormalIdx;
        public uint RoughnessIdx;
        public uint MetallicIdx;
        public uint EmissiveIdx;
        public uint AOIdx;
        public uint DetailNormalIdx;
        public uint DetailMaskIdx;
        public Vector2 DetailTiling;
        public Vector2 DetailOffset;
        public Vector3 EmissiveColor;
        public float EmissiveIntensity;
        public Vector3 Tint;            // multiplies the albedo map (Material "Tint", default white)
        public uint SettingsBufferIdx;  // the raw buffer of every material's effect settings (Material.SettingsStride bytes at its MaterialID); was padding

        /// <summary>
        /// Texture slot names derived from this struct's fields (strips "Idx" suffix).
        /// E.g. "AlbedoIdx" → "Albedo", "NormalIdx" → "Normal", etc.
        /// </summary>
        public static readonly string[] TextureSlotNames =
            typeof(MaterialData).GetFields()
                .Where(f => f.Name.EndsWith("Idx"))
                .Select(f => f.Name[..^3])
                .ToArray();
    }

    /// <summary>
    /// Base class for material scalar properties exposed in the inspector.
    /// Same pattern as TextureEffectParameter — the inspector reads the Value property.
    /// </summary>
    public abstract class MaterialProperty
    {
        public string Name { get; set; }
        public Material Material { get; set; }
        public abstract Type ValueType { get; }

        /// <summary>What it does (an effect's "-- text" after its default); null for none.</summary>
        public string? Description { get; set; }

        /// <summary>The range of a number the inspector shows as a slider (an effect's "[min, max]"); null for a plain field.</summary>
        public System.Numerics.Vector2? Range { get; set; }

        /// <summary>Take the value of a property of the same type (a clone, or the same setting before a rebuild).</summary>
        public abstract void CopyFrom(MaterialProperty other);

        /// <summary>
        /// Parse a string value and set this property. Used by MaterialLoader.
        /// </summary>
        public abstract bool TryParseAndSet(string text);

        /// <summary>
        /// Format this property's current value as a YAML-compatible string.
        /// Returns null if the value is at its default.
        /// </summary>
        public abstract string FormatValue();

        /// <summary>
        /// Restore the default value. Used when a hot-reloaded definition no longer sets this property.
        /// </summary>
        public abstract void ResetToDefault();
    }

    /// <summary>
    /// Typed material property. Setter calls back to Material to update MaterialData on GPU.
    /// </summary>
    public class MaterialProperty<T> : MaterialProperty where T : struct
    {
        private T _value;
        private T _defaultValue;
        private Action<Material, T> _apply;

        public override Type ValueType => typeof(T);

        public T Value
        {
            get => _value;
            set
            {
                _value = value;
                _apply?.Invoke(Material, value);
            }
        }

        public MaterialProperty(string name, Material mat, T defaultValue, Action<Material, T> apply)
        {
            Name = name;
            Material = mat;
            _value = defaultValue;
            _defaultValue = defaultValue;
            _apply = apply;
        }

        public override bool TryParseAndSet(string text)
        {
            if (!Serialization.YAMLConverterHelper.TryParse<T>(text, out var result))
                return false;
            Value = result;
            return true;
        }

        public override string FormatValue()
        {
            if (EqualityComparer<T>.Default.Equals(_value, _defaultValue))
                return null;
            return Serialization.YAMLConverterHelper.Format(_value);
        }

        public override void ResetToDefault() => Value = _defaultValue;

        public override void CopyFrom(MaterialProperty other)
        {
            if (other is MaterialProperty<T> same) Value = same.Value;
        }
    }

    
    /// <summary>
    /// Maps a shader pass to a RenderPass enum (Apex pattern)
    /// </summary>
    public class ShaderPass
    {
        /// <summary>A pass whose name is no RenderPass (e.g. the terrain's "DepthPrepass"): selected by name only.</summary>
        public const RenderPass NamedOnly = (RenderPass)(-1);

        public RenderPass RenderPass;
        public EffectPass EffectPass = null!;
        public int Index;
    }

    /// <summary>
    /// Parses technique passes and maps them to RenderPass enum by name (Apex pattern)
    /// </summary>
    public class SubShader
    {
        public List<ShaderPass> ShaderPasses = new();

        public SubShader(EffectTechnique technique)
        {
            for (int i = 0; i < technique.Passes.Count; i++)
            {
                var pass = technique.Passes[i];
                // An unknown name used to parse to default(RenderPass) = Shadow, and got the shadow PSO overrides.
                // "<Pass>_<variant>" (Shadow_Holes) is a variant of that pass: it gets the pass' PSO state, and only a
                // renderer that asks for it by name draws with it (SetPass(RenderPass) finds the plain one first).
                var passName = pass.Name;
                int variant = passName.IndexOf('_');
                if (variant > 0) passName = passName[..variant];
                if (!Enum.TryParse<RenderPass>(passName, out var type))
                    type = ShaderPass.NamedOnly;

                ShaderPasses.Add(new ShaderPass
                {
                    EffectPass = pass,
                    RenderPass = type,
                    Index = i
                });
            }
        }
    }

    [CreateAsset("Material")]
    public class Material : Asset, IDisposable
    {
        // Per-pass PSOs (Apex pattern for multi-pass rendering like shadow maps)
        private Dictionary<int, PipelineState> _passPipelineStates = new();
        private Dictionary<int, PipelineState> _passWireframePSOs = new();
        private bool _isFullscreenPass = false; // Skip wireframe for post-process passes
        private Dictionary<string, ConstantBuffer> _constantBuffers = new();
        private Dictionary<string, Texture> _textures = new();
        private List<TextureEffectParameter> _textureParameters = new();
        private List<MaterialProperty> _materialProperties = new();
        // The effect's own settings (Effect.MaterialParams), writing into this material's block of the settings buffer
        private readonly List<MaterialProperty> _effectProperties = new();
        // Its texture settings (TextureRef): slots like the MaterialData ones, writing the texture's bindless index
        private readonly List<TextureEffectParameter> _effectTextureParameters = new();
        private readonly Dictionary<string, MaterialParamDesc> _effectTextureSettings = new();

        // An effect that declares settings ("//=" in its struct Settings) reads those, not MaterialData
        private bool HasOwnSettings => _effect?.MaterialParams.Count > 0;
        
        /// <summary>
        /// Read-only access to texture bindings (slot name → Texture).
        /// Used by MaterialSerializer for YAML serialization.
        /// </summary>
        public IReadOnlyDictionary<string, Texture> Textures => _textures;
        
        /// <summary>
        /// Texture parameters for inspector binding, writing back through SetTexture(). Normally the MaterialData
        /// slots (Albedo, Normal, ...); for an effect that declares its own settings, its texture settings
        /// (<c>TextureRef Name; //= Gray</c>), which write the texture's bindless index into the settings block.
        /// </summary>
        public IReadOnlyList<TextureEffectParameter> TextureParameters => HasOwnSettings ? _effectTextureParameters : _textureParameters;
        
        /// <summary>
        /// Scalar material properties for inspector binding and the .mat "Parameters". Normally the MaterialData
        /// ones (Tint, emissive, detail), each writing to MaterialData on change. An effect that declares its own
        /// settings ("//=" in its struct Settings, Effect.MaterialParams) gets those instead: it does not read
        /// MaterialData; its settings are written into the material's block of the settings buffer (WriteSetting).
        /// </summary>
        public IReadOnlyList<MaterialProperty> MaterialProperties => HasOwnSettings ? _effectProperties : _materialProperties;

        /// <summary>A material property by name, or null.</summary>
        public MaterialProperty? GetProperty(string name)
        {
            foreach (var p in MaterialProperties)
                if (p.Name == name) return p;
            return null;
        }

        /// <summary>
        /// Set a material property by name (an effect setting or a MaterialData one). False if there is none of
        /// that name and type. This changes the material for everything drawn with it; a value for one object
        /// is per-object data (the effect's struct Instance, PersistentRenderer.SetInstance).
        /// </summary>
        public bool SetProperty<T>(string name, T value) where T : struct
        {
            if (GetProperty(name) is not MaterialProperty<T> p) return false;
            p.Value = value;
            return true;
        }
        
        // Material ID indirection system
        public int MaterialID { get; private set; }
        private static int _nextMaterialID = 0;
        private static List<MaterialData> _allMaterials = new();
        private static readonly Lock _materialIdLock = new();
        private static ID3D12Resource? _materialsBuffer;
        private static IntPtr _materialsBufferPtr;
        private static bool _materialsBufferDirty = true;
        
        // SubShaders parsed from Effect techniques (Apex pattern)
        private readonly List<SubShader> _subShaders = new();
        
        public PipelineState PipelineState => _passPipelineStates.TryGetValue(Pass, out var pso) ? pso : _passPipelineStates.GetValueOrDefault(0)!;
        public IEnumerable<ConstantBuffer> ConstantBuffers => _constantBuffers.Values;
        private Effect _effect;
        public Effect Effect
        {
            get => _effect;
            set
            {
                // During construction or plain assignment: just set reference
                if (!_built)
                {
                    _effect = value;
                    return;
                }
                // Live swap: rebuild PSOs
                if (value != null && value != _effect)
                    SetEffect(value);
            }
        }

        // Set once constructed: assigning Effect then swaps it (SetEffect). Not "has pipeline states": a material
        // of a project effect that does not compile has none, and must still be able to change to another effect.
        private bool _built;
        
        // Pass/Technique selection (Apex pattern)
        public int Pass { get; set; } = 0;
        public int Technique { get; set; } = 0;
        
        public List<ShaderPass> GetPasses() => _subShaders.Count > Technique ? _subShaders[Technique].ShaderPasses : new();

        /// <summary>True if the active pass has Hull+Domain shaders (requires patch topology).</summary>
        public bool HasTessellation
        {
            get
            {
                var passes = GetPasses();
                if (Pass < passes.Count)
                    return passes[Pass].EffectPass.HullShader != null;
                return false;
            }
        }
        
        /// <summary>
        /// Set active pass by RenderPass enum (Apex pattern)
        /// </summary>
        public void SetPass(RenderPass pass)
        {
            int i = 0;
            foreach (var p in GetPasses())
            {
                if (p.RenderPass == pass)
                {
                    Pass = i;
                    return;
                }
                i++;
            }
        }

        /// <summary>
        /// Set active pass by name (for non-enum passes like "ShadowSinglePass")
        /// </summary>
        public void SetPass(string name)
        {
            int i = 0;
            foreach (var p in GetPasses())
            {
                if (p.EffectPass.Name == name)
                {
                    Pass = i;
                    return;
                }
                i++;
            }
        }
        
        public RenderPass GetPass(int index) => GetPasses()[index].RenderPass;
        
        /// <summary>
        /// Check if material has a specific render pass defined
        /// </summary>
        public bool HasPass(RenderPass pass)
        {
            foreach (var p in GetPasses())
            {
                if (p.RenderPass == pass) return true;
            }
            return false;
        }
        
        private Dictionary<string, int> _textureSlots = new();

        /// <summary>
        /// Parameterless constructor for deferred asset stub creation.
        /// Stubs only carry a GUID for later resolution — they are never used for rendering.
        /// </summary>
        public Material() { }

        public Material(Effect effect) : this(effect, Freefall.Engine.Device) { }

        public Material(Effect effect, GraphicsDevice device)
        {
            // Assign unique MaterialID (only once per Material lifetime)
            lock (_materialIdLock)
            {
                MaterialID = _nextMaterialID++;
                _allMaterials.Add(new MaterialData
                {
                    EmissiveColor = Vector3.One,
                    EmissiveIntensity = 1.0f,
                    Tint = Vector3.One,
                    SettingsBufferIdx = _settingsBufferBindlessIndex,
                });
                _materialsBufferDirty = true;
            }
            
            InitMaterialProperties();
            InitFromEffect(effect, device);
            InitEffectProperties();

            lock (_liveMaterials) _liveMaterials.Add(new WeakReference<Material>(this));
            _built = true;
        }

        /// <summary>
        /// The effect's own settings as material properties. Settings that keep their name and type keep
        /// their value when the effect changes or is recompiled; new ones start at their default.
        /// </summary>
        private void InitEffectProperties()
        {
            var before = new Dictionary<string, MaterialProperty>();
            foreach (var p in _effectProperties) before[p.Name] = p;
            _effectProperties.Clear();
            _effectTextureParameters.Clear();
            _effectTextureSettings.Clear();
            if (_effect == null) return;

            foreach (var desc in _effect.MaterialParams)
            {
                string name = desc.Name;
                int at = desc.Offset;

                // A texture: a slot (saved under Textures:, like the MaterialData ones) whose bindless index goes
                // into the block; the default texture until one is set
                if (desc.IsTexture)
                {
                    _effectTextureSettings[name] = desc;
                    _textures.TryGetValue(name, out var assigned);
                    WriteSetting(at, (assigned ?? desc.DefaultTextureAsset)?.BindlessIndex ?? 0u);
                    var slot = new TextureEffectParameter { Name = name, Material = this };
                    slot.Value = assigned;    // its setter calls SetTexture, which is fine: the setting is known
                    _effectTextureParameters.Add(slot);
                    continue;
                }

                MaterialProperty? prop = desc.Type switch
                {
                    "float" => new MaterialProperty<float>(name, this, desc.At(0),
                        (m, v) => m.WriteSetting(at, v)),
                    "float2" => new MaterialProperty<Vector2>(name, this, new Vector2(desc.At(0), desc.At(1)),
                        (m, v) => m.WriteSetting(at, v)),
                    "float4" => new MaterialProperty<Vector4>(name, this, new Vector4(desc.At(0), desc.At(1), desc.At(2), desc.At(3)),
                        (m, v) => m.WriteSetting(at, v)),
                    "float3" when desc.IsColor => new MaterialProperty<Color3>(name, this, new Color3(desc.At(0, 1f), desc.At(1, 1f), desc.At(2, 1f)),
                        (m, v) => m.WriteSetting(at, v.ToVector3())),
                    "float3" => new MaterialProperty<Vector3>(name, this, new Vector3(desc.At(0), desc.At(1), desc.At(2)),
                        (m, v) => m.WriteSetting(at, v)),
                    "int" => new MaterialProperty<int>(name, this, (int)desc.At(0),
                        (m, v) => m.WriteSetting(at, v)),
                    "uint" => new MaterialProperty<int>(name, this, (int)desc.At(0),
                        (m, v) => m.WriteSetting(at, (uint)Math.Max(v, 0))),
                    "bool" => new MaterialProperty<bool>(name, this, desc.At(0) != 0f,
                        (m, v) => m.WriteSetting(at, v ? 1u : 0u)),   // an HLSL bool is 4 bytes
                    _ => null,
                };
                if (prop == null) continue;

                prop.Description = desc.Description;
                if (desc.Min is float lo && desc.Max is float hi) prop.Range = new Vector2(lo, hi);
                prop.ResetToDefault();                                    // writes the default into the cbuffer
                if (before.TryGetValue(name, out var old)) prop.CopyFrom(old);
                _effectProperties.Add(prop);
            }
        }

        // ── Effect settings on the GPU ──
        // Every material's effect settings (Effect.MaterialParams) sit in one raw buffer, SettingsStride bytes at
        // MaterialID * SettingsStride, which shaders load by the instance's MaterialID (MaterialData.SettingsBufferIdx
        // names the buffer). Not a per-material constant buffer: draws are batched by effect (BatchKey), and a batch
        // binds the constant buffers of the first material it met, so every other material would draw with its
        // settings (found 2026-10-16: every bindstone tube drew with one material's noise and pulse).
        public const int SettingsStride = 256;
        private static byte[] _settingsData = new byte[SettingsStride * 64];
        private static bool _settingsDirty = true;
        private static ID3D12Resource? _settingsBuffer;
        private static IntPtr _settingsBufferPtr;
        private static uint _settingsBufferBindlessIndex;
        private static int _settingsBufferCapacity;

        // An effect setting into this material's block of the settings buffer
        private unsafe void WriteSetting<T>(int offset, T value) where T : unmanaged
        {
            if (offset < 0 || offset + sizeof(T) > SettingsStride) return;
            lock (_materialIdLock)
            {
                int at = MaterialID * SettingsStride + offset;
                if (at + sizeof(T) > _settingsData.Length)
                    Array.Resize(ref _settingsData, Math.Max(_settingsData.Length * 2, (MaterialID + 1) * SettingsStride));
                System.Runtime.InteropServices.MemoryMarshal.Write(_settingsData.AsSpan(at), in value);
                _settingsDirty = true;
            }
        }

        // The settings buffer: grown with the materials buffer, uploaded when a setting changed. Its bindless index
        // never changes once made; every MaterialData carries it.
        private static void BindSettingsBuffer(GraphicsDevice device, int capacity)
        {
            bool fixIndices = false;
            if (_settingsBuffer == null || capacity > _settingsBufferCapacity)
            {
                int bytes = capacity * SettingsStride;
                if (_settingsBuffer != null) device.DeferDispose(_settingsBuffer);
                _settingsBuffer = device.NativeDevice.CreateCommittedResource(
                    new HeapProperties(HeapType.Upload),
                    HeapFlags.None,
                    ResourceDescription.Buffer((ulong)bytes),
                    ResourceStates.GenericRead);
                unsafe
                {
                    void* ptr;
                    _settingsBuffer.Map(0, null, &ptr);
                    _settingsBufferPtr = (IntPtr)ptr;
                }
                if (_settingsBufferBindlessIndex == 0)
                {
                    _settingsBufferBindlessIndex = device.AllocateBindlessIndex();
                    fixIndices = true;
                }
                device.CreateRawBufferSRV(_settingsBuffer, (uint)(bytes / 4), _settingsBufferBindlessIndex);
                _settingsBufferCapacity = capacity;
                _settingsDirty = true;
            }

            lock (_materialIdLock)
            {
                if (fixIndices)
                {
                    var all = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_allMaterials);
                    for (int i = 0; i < all.Length; i++) all[i].SettingsBufferIdx = _settingsBufferBindlessIndex;
                    _materialsBufferDirty = true;
                }
                if (_settingsDirty)
                {
                    int count = Math.Min(_settingsData.Length, _settingsBufferCapacity * SettingsStride);
                    unsafe
                    {
                        fixed (byte* src = _settingsData)
                            Buffer.MemoryCopy(src, (void*)_settingsBufferPtr, _settingsBufferCapacity * SettingsStride, count);
                    }
                    _settingsDirty = false;
                }
            }
        }
        
        private void InitMaterialProperties()
        {
            _materialProperties.Clear();
            // Colour-matches a texture to the scene without editing the file: a pack's grey-teal leaves pulled to
            // green, a too-bright bark darkened (like TerrainLayer.Tint). Unity packs often got this from _Color.
            _materialProperties.Add(new MaterialProperty<Color3>("Tint", this, new Color3(1f, 1f, 1f),
                (m, v) => m.UpdateMaterialScalar((ref MaterialData d) => d.Tint = v.ToVector3())));
            _materialProperties.Add(new MaterialProperty<Color3>("EmissiveColor", this, new Color3(),
                (m, v) => m.UpdateMaterialScalar((ref MaterialData d) => d.EmissiveColor = v.ToVector3())));
            _materialProperties.Add(new MaterialProperty<float>("EmissiveIntensity", this, 1.0f,
                (m, v) => m.SetEmissiveIntensity(v)));
            // Lamp glass: glows only while lamps are lit (Daylight.Lamps), like a PointLight.OnlyWhenDark
            _materialProperties.Add(new MaterialProperty<bool>("EmissiveWhenDark", this, false,
                (m, v) => m.SetEmissiveWhenDark(v)));
            _materialProperties.Add(new MaterialProperty<Vector2>("DetailTiling", this, Vector2.Zero,
                (m, v) => m.UpdateMaterialScalar((ref MaterialData d) => d.DetailTiling = v)));
            _materialProperties.Add(new MaterialProperty<Vector2>("DetailOffset", this, Vector2.Zero,
                (m, v) => m.UpdateMaterialScalar((ref MaterialData d) => d.DetailOffset = v)));
        }

        /// <summary>
        /// Raised after a material's effect was swapped. Persistent (GPU-resident) draws live in
        /// batches keyed by effect and pass, so renderers using the material re-register.
        /// </summary>
        public static event Action<Material>? EffectChanged;

        /// <summary>
        /// Change the Effect on an existing material at runtime.
        /// Disposes old PSOs/CBs and rebuilds with the new Effect.
        /// Preserves existing texture bindings.
        /// </summary>
        public void SetEffect(Effect effect)
        {
            if (effect == null || effect == Effect) return;

            // Dispose old GPU state
            foreach (var cb in _constantBuffers.Values)
                cb.Dispose();
            _constantBuffers.Clear();
            _passPipelineStates.Clear();
            _passWireframePSOs.Clear();
            _subShaders.Clear();
            _textureParameters.Clear();
            _textureSlots.Clear();
            _bindlessIndices.Clear();

            // Preserve existing textures — they'll be re-bound after init
            var savedTextures = new Dictionary<string, Texture>(_textures);
            _textures.Clear();

            InitFromEffect(effect, Freefall.Engine.Device);
            InitEffectProperties();

            // Re-apply saved textures
            foreach (var (slot, tex) in savedTextures)
                SetTexture(slot, tex);

            EffectChanged?.Invoke(this);

            //Debug.Log($"[Material] Switched '{Name}' to effect '{effect.Name}'");
        }

        // Every effect-based material, so shader hot reload can rebuild their pipeline states
        private static readonly List<WeakReference<Material>> _liveMaterials = new();

        /// <summary>
        /// Rebuild every live material using <paramref name="effect"/> (any instance of its Key: an engine effect's
        /// name, a project effect's GUID) after it was recompiled (ShaderHotReload, EffectLoader). Main thread,
        /// between frames. Returns the count.
        /// </summary>
        internal static int RebuildForEffect(Effect effect)
        {
            var targets = new List<Material>();
            lock (_liveMaterials)
            {
                for (int i = _liveMaterials.Count - 1; i >= 0; i--)
                {
                    if (!_liveMaterials[i].TryGetTarget(out var m)) { _liveMaterials.RemoveAt(i); continue; }
                    if (m._effect != null && m._effect.Key == effect.Key) targets.Add(m);
                }
            }

            foreach (var m in targets)
                m.RebuildFromEffect();
            return targets.Count;
        }

        /// <summary>
        /// Same as SetEffect for an effect whose shaders changed underneath it: new PSOs and constant
        /// buffers, keeping textures, bindless indices and the values already written to the buffers.
        /// Old GPU objects may still be referenced by frames in flight, so they are released deferred.
        /// </summary>
        private void RebuildFromEffect()
        {
            var device = Freefall.Engine.Device;
            var effect = _effect;

            var oldBuffers = new Dictionary<string, ConstantBuffer>(_constantBuffers);
            // Only the native PSO: PipelineState.Dispose() also releases its root signature, which here
            // is the device's shared GlobalRootSignature (disposing it crashes the next draw).
            foreach (var pso in _passPipelineStates.Values) device.DeferDispose(pso.Native);
            foreach (var pso in _passWireframePSOs.Values) device.DeferDispose(pso.Native);

            _constantBuffers.Clear();
            _passPipelineStates.Clear();
            _passWireframePSOs.Clear();
            _subShaders.Clear();
            _textureParameters.Clear();
            _textureSlots.Clear();

            InitFromEffect(effect, device);

            foreach (var (name, cb) in _constantBuffers)
            {
                if (oldBuffers.TryGetValue(name, out var oldCb))
                    cb.CopyValuesFrom(oldCb);
            }
            foreach (var cb in oldBuffers.Values) device.DeferDispose(cb);

            // The settings may have changed with the source; known ones keep their values (written again
            // into the new buffer, whose layout may differ from the old one's)
            InitEffectProperties();

            EffectChanged?.Invoke(this);
        }

        /// <summary>
        /// Core initialization: build PSOs, SubShaders, CBs from an Effect.
        /// </summary>
        /// <summary>
        /// Blend "NoColor": every render target bound but none written, for a depth-only pass drawn with the G-buffer
        /// targets still bound (the terrain's depth prepass; the targets' formats must still match the bound ones).
        /// </summary>
        private static BlendDescription NoColorWrites()
        {
            var desc = new BlendDescription { AlphaToCoverageEnable = false, IndependentBlendEnable = false };
            desc.RenderTarget[0] = new RenderTargetBlendDescription
            {
                BlendEnable = false,
                SourceBlend = Blend.One,
                DestinationBlend = Blend.Zero,
                BlendOperation = BlendOperation.Add,
                SourceBlendAlpha = Blend.One,
                DestinationBlendAlpha = Blend.Zero,
                BlendOperationAlpha = BlendOperation.Add,
                RenderTargetWriteMask = ColorWriteEnable.None
            };
            return desc;
        }

        // The deferred renderer's targets, in the order it binds them (DeferredRenderer.CreateRenderTextures):
        // 3-6 = the G-buffer (Albedo, Normals, Data, DepthGBuffer, EntityId, Height; always all six bound, a shader with
        // fewer outputs gets no writes to the rest), 2 = the forward pass (Composite +
        // EntityId), 1 = Composite (forward and post-process passes) or Albedo (transparent G-buffer pass): both
        // R16G16B16A16_Float. A pipeline's formats must match the bound targets'. Until 2026-10-17 the count of 1 and
        // 2 declared R8G8B8A8_UNorm and EntityId R32_Float while the targets are float16 and R32_UInt (undefined in
        // D3D12, reported only by the debug layer). A shader drawing into another target names its format
        // (RenderTargetFormat, e.g. the editor previews' 8-bit thumbnails).
        private static readonly Format[] GBufferFormats =
        {
            Format.R16G16B16A16_Float, Format.R16G16B16A16_SNorm, Format.R8G8B8A8_UNorm,
            Format.R32_Float, Format.R32_UInt, Format.R16G16_Float,
        };

        /// <summary>
        /// The blend state fitted to the bound targets:
        /// - blending off on integer targets (EntityId, R32_UInt): D3D12 rejects a pipeline that blends into one
        ///   (E_INVALIDARG), and Vortice's BlendDescription constructors enable blending on every target, Opaque's
        ///   One/Zero included, so an id is written as it is;
        /// - no writes to the targets past the shader's own count (<paramref name="written"/>): a G-buffer shader with 4
        ///   outputs (grass, the sky) is drawn with all six G-buffer targets bound, so its pipeline names all six.
        /// The other targets keep the blend as it was (each its own copy of target 0's when it was shared).
        /// </summary>
        private static BlendDescription FitBlendToTargets(BlendDescription blend, Format[] formats, int written)
        {
            bool integer = false;
            foreach (var f in formats) integer |= IsIntegerFormat(f);
            if (!integer && written >= formats.Length) return blend;

            if (!blend.IndependentBlendEnable)
            {
                var shared = blend.RenderTarget[0];
                for (int i = 1; i < 8; i++)
                    blend.RenderTarget[i] = shared;
                blend.IndependentBlendEnable = true;
            }
            for (int i = 0; i < formats.Length && i < 8; i++)
            {
                var rt = blend.RenderTarget[i];
                if (IsIntegerFormat(formats[i]))
                {
                    rt.BlendEnable = false;
                    rt.LogicOpEnable = false;
                }
                if (i >= written)
                    rt.RenderTargetWriteMask = ColorWriteEnable.None;
                blend.RenderTarget[i] = rt;
            }
            return blend;
        }

        private static bool IsIntegerFormat(Format f) => f is Format.R32_UInt or Format.R32_SInt or Format.R16_UInt
            or Format.R16_SInt or Format.R8_UInt or Format.R8_SInt or Format.R32G32_UInt or Format.R16G16_UInt
            or Format.R8G8B8A8_UInt or Format.R32G32B32A32_UInt or Format.R16G16B16A16_UInt;

        private static Format[] TargetFormats(int count, string? overrideFormat)
        {
            if (overrideFormat != null && count <= 1 && Enum.TryParse<Format>(overrideFormat, true, out var single))
                return new[] { single };
            return count switch
            {
                >= 3 => GBufferFormats,     // the G-buffer pass always binds all six (FitBlendToTargets masks the rest)
                2 => new[] { Format.R16G16B16A16_Float, Format.R32_UInt },
                _ => new[] { Format.R16G16B16A16_Float },
            };
        }

        private void InitFromEffect(Effect effect, GraphicsDevice device)
        {
            //Debug.Log($"[Material] Creating Material for Effect: {effect.Name}");
            _effect = effect;   // not the property: during SetEffect it would swap again
            
            // Populate SubShaders from Effect techniques (Apex pattern)
            foreach (var technique in effect.Techniques)
                _subShaders.Add(new SubShader(technique));
            
            // Create texture parameters for each MaterialData slot
            foreach (var slotName in MaterialData.TextureSlotNames)
            {
                _textures.TryGetValue(slotName, out var tex);
                _textureParameters.Add(new TextureEffectParameter
                {
                    Name = slotName,
                    Material = this,
                    Value = tex
                });
            }

            // A project effect whose first compile failed has no techniques: no pipeline states, no passes, so no
            // draws. Its settings still load and save; it is built when the effect compiles (RebuildForEffect).
            if (effect.Techniques.Count == 0 || effect.Techniques[0].Passes.Count == 0)
                return;

            var renderState = effect.RenderState;

            // Render target formats from shader metadata: the deferred renderer's targets for that count
            Format[] rtvFormats = TargetFormats(renderState.RenderTargetCount, renderState.RenderTargetFormat);


            // In bindless, we use the Global Root Signature from the device
            var rootSignature = device.GlobalRootSignature;

            // Input Layout (Bindless = Empty)
            // We fetch vertices manually in VS using SV_VertexID

            // Rasterizer State — use CullMode from shader metadata (FXParser defaults to "Back")
            var cullMode = renderState.CullMode switch
            {
                "None" => CullMode.None,
                "Front" => CullMode.Front,
                _ => CullMode.Back
            };
            var rasterizerDesc = new RasterizerDescription(cullMode, FillMode.Solid);

            // Blend State - use shader metadata, fall back to Effect's property
            BlendDescription blendDesc = renderState.BlendMode switch
            {
                "Additive" => new BlendDescription(Blend.One, Blend.One),
                "AlphaBlend" => BlendDescription.AlphaBlend,
                _ => effect.BlendState == BlendState.Additive 
                    ? new BlendDescription(Blend.One, Blend.One)
                    : effect.BlendState == BlendState.AlphaBlend 
                        ? BlendDescription.AlphaBlend 
                        : BlendDescription.Opaque
            };

            DepthStencilDescription depthDesc;
            Format depthFormat;
            
            // Parse DepthFunc from shader metadata (default: GreaterEqual for reverse depth)
            var parsedDepthFunc = renderState.DepthFunc switch
            {
                "Less" => ComparisonFunction.Less,
                "LessEqual" => ComparisonFunction.LessEqual,
                "Greater" => ComparisonFunction.Greater,
                "GreaterEqual" => ComparisonFunction.GreaterEqual,
                "Equal" => ComparisonFunction.Equal,
                "Always" => ComparisonFunction.Always,
                "Never" => ComparisonFunction.Never,
                _ => ComparisonFunction.GreaterEqual // Reverse depth default
            };
            
            if (!renderState.DepthTest && !renderState.DepthWrite)
            {
                // No depth (fullscreen quads, post-process)
                depthDesc = DepthStencilDescription.None;
                depthFormat = Format.Unknown;
            }
            else if (!renderState.DepthWrite)
            {
                // Read only (skybox, transparent)
                depthDesc = new DepthStencilDescription
                {
                    DepthEnable = true,
                    DepthWriteMask = DepthWriteMask.Zero,
                    DepthFunc = parsedDepthFunc,
                    StencilEnable = false
                };
                depthFormat = Format.D32_Float;
            }
            else
            {
                // Full depth (standard geometry) — reverse depth: GreaterEqual
                depthDesc = new DepthStencilDescription
                {
                    DepthEnable = true,
                    DepthWriteMask = DepthWriteMask.All,
                    DepthFunc = parsedDepthFunc,
                    StencilEnable = false
                };
                depthFormat = Format.D32_Float;
            }


            var psoDescBase = new GraphicsPipelineStateDescription()
            {
                RootSignature = rootSignature,
                InputLayout = default, 
                RasterizerState = rasterizerDesc,
                BlendState = blendDesc,
                DepthStencilState = depthDesc,
                PrimitiveTopologyType = PrimitiveTopologyType.Triangle,
                RenderTargetFormats = rtvFormats,
                DepthStencilFormat = depthFormat,
                SampleDescription = new SampleDescription(1, 0),
                SampleMask = uint.MaxValue,
                Flags = PipelineStateFlags.None
            };

            // Create PSO for each pass (Apex multi-pass pattern for shadow maps, etc.)
            var firstTechnique = effect.Techniques[0];
            for (int passIndex = 0; passIndex < firstTechnique.Passes.Count; passIndex++)
            {
                var currentPass = firstTechnique.Passes[passIndex];
                
                // Shadow pass overrides (shared by both traditional and mesh shader paths)
                var passType = _subShaders[0].ShaderPasses[passIndex].RenderPass;
                var passRtvFormats = rtvFormats;
                int passWritten = currentPass.RenderState?.RenderTargetCount ?? renderState.RenderTargetCount;   // outputs the shader writes
                var passDepthFormat = depthFormat;
                var passDepthDesc = depthDesc;
                var passRasterDesc = rasterizerDesc;
                
                // Per-pass render state override (from SetRenderState() in pass block)
                var passRS = currentPass.RenderState;
                if (passRS != null)
                {
                    // Resolve RT formats for this pass
                    passRtvFormats = TargetFormats(passRS.RenderTargetCount, passRS.RenderTargetFormat);
                    
                    // Per-pass DepthFunc override (e.g. preview shaders use LessEqual vs reverse depth)
                    var passDepthFunc = parsedDepthFunc;
                    if (passRS.DepthFunc != null)
                    {
                        passDepthFunc = passRS.DepthFunc switch
                        {
                            "Less" => ComparisonFunction.Less,
                            "LessEqual" => ComparisonFunction.LessEqual,
                            "Greater" => ComparisonFunction.Greater,
                            "GreaterEqual" => ComparisonFunction.GreaterEqual,
                            "Equal" => ComparisonFunction.Equal,
                            "Always" => ComparisonFunction.Always,
                            "Never" => ComparisonFunction.Never,
                            _ => parsedDepthFunc
                        };
                    }

                    // Resolve depth stencil for this pass
                    if (!passRS.DepthTest && !passRS.DepthWrite)
                    {
                        passDepthDesc = DepthStencilDescription.None;
                        passDepthFormat = Format.Unknown;
                    }
                    else if (!passRS.DepthWrite)
                    {
                        passDepthDesc = new DepthStencilDescription
                        {
                            DepthEnable = true,
                            DepthWriteMask = DepthWriteMask.Zero,
                            DepthFunc = passDepthFunc,
                            StencilEnable = false
                        };
                        passDepthFormat = Format.D32_Float;
                    }
                    else
                    {
                        passDepthDesc = new DepthStencilDescription
                        {
                            DepthEnable = true,
                            DepthWriteMask = DepthWriteMask.All,
                            DepthFunc = passDepthFunc,
                            StencilEnable = false
                        };
                        passDepthFormat = Format.D32_Float;
                    }
                }

                // Per-pass blend state override
                var passBlendDesc = blendDesc;
                if (passRS != null && passRS.BlendMode != "Opaque")
                {
                    passBlendDesc = passRS.BlendMode switch
                    {
                        "Additive" => new BlendDescription(Blend.One, Blend.One),
                        "AlphaBlend" => BlendDescription.AlphaBlend,
                        "NoColor" => NoColorWrites(),
                        _ => blendDesc
                    };
                }

                if (passType == RenderPass.Shadow || passType == RenderPass.LocalShadow)
                {
                    passRtvFormats = Array.Empty<Format>();
                    passDepthFormat = Format.D32_Float;
                    passDepthDesc = new DepthStencilDescription
                    {
                        DepthEnable = true,
                        DepthWriteMask = DepthWriteMask.All,
                        DepthFunc = ComparisonFunction.LessEqual,
                        StencilEnable = false
                    };
                    var shadowRaster = new RasterizerDescription(CullMode.None, FillMode.Solid);
                    shadowRaster.DepthBias = 100;
                    shadowRaster.SlopeScaledDepthBias = 3.0f;
                    shadowRaster.DepthBiasClamp = 0.0f;
                    passRasterDesc = shadowRaster;
                }
                
                if (currentPass.IsMeshShaderPass)
                {
                    // Mesh shader pipeline: AS → MS → PS (no vertex/geometry/tessellation shaders)
                    var meshPsoDesc = new PipelineState.MeshShaderPSODescription
                    {
                        AmplificationShader = currentPass.AmplificationShader?.Bytecode ?? default,
                        MeshShader = currentPass.MeshShader!.Bytecode,
                        PixelShader = currentPass.PixelShader?.Bytecode ?? default,
                        RasterizerState = passRasterDesc,
                        BlendState = FitBlendToTargets(passBlendDesc, passRtvFormats, passWritten),
                        DepthStencilState = passDepthDesc,
                        RenderTargetFormats = passRtvFormats,
                        DepthStencilFormat = passDepthFormat,
                        SampleDescription = new SampleDescription(1, 0),
                        SampleMask = uint.MaxValue,
                    };
                    
                    _passPipelineStates[passIndex] = new PipelineState(device, rootSignature, meshPsoDesc);
                    
                    // Wireframe variant for mesh shaders
                    meshPsoDesc.RasterizerState = new RasterizerDescription(CullMode.None, FillMode.Wireframe);
                    _passWireframePSOs[passIndex] = new PipelineState(device, rootSignature, meshPsoDesc);
                }
                else
                {
                    // Traditional pipeline: VS → [HS → DS →] [GS →] PS
                    var psoDesc = psoDescBase;
                    psoDesc.VertexShader = currentPass.VertexShader?.Bytecode ?? default;
                    psoDesc.PixelShader = currentPass.PixelShader?.Bytecode ?? default;
                    psoDesc.HullShader = currentPass.HullShader?.Bytecode ?? default;
                    psoDesc.DomainShader = currentPass.DomainShader?.Bytecode ?? default;
                    
                    if (currentPass.HullShader != null)
                        psoDesc.PrimitiveTopologyType = PrimitiveTopologyType.Patch;
                    
                    // Apply per-pass render state overrides (from SetRenderState() or Shadow defaults)
                    psoDesc.RenderTargetFormats = passRtvFormats;
                    psoDesc.DepthStencilFormat = passDepthFormat;
                    psoDesc.DepthStencilState = passDepthDesc;
                    psoDesc.RasterizerState = passRasterDesc;
                    psoDesc.BlendState = FitBlendToTargets(passBlendDesc, passRtvFormats, passWritten);
                    
                    _passPipelineStates[passIndex] = new PipelineState(device, rootSignature, psoDesc);
                    
                    psoDesc.RasterizerState = new RasterizerDescription(CullMode.None, FillMode.Wireframe);
                    _passWireframePSOs[passIndex] = new PipelineState(device, rootSignature, psoDesc);
                }
                
                // Reflection for constant buffers (from first pass for simplicity)
                if (passIndex == 0)
                {
                    if (currentPass.VertexShader?.Reflection != null) CreateConstantBuffers(device, currentPass.VertexShader.Reflection);
                    if (currentPass.PixelShader?.Reflection != null) CreateConstantBuffers(device, currentPass.PixelShader.Reflection);
                    if (currentPass.HullShader?.Reflection != null) CreateConstantBuffers(device, currentPass.HullShader.Reflection);
                    if (currentPass.DomainShader?.Reflection != null) CreateConstantBuffers(device, currentPass.DomainShader.Reflection);
                    if (currentPass.MeshShader?.Reflection != null) CreateConstantBuffers(device, currentPass.MeshShader.Reflection);
                    if (currentPass.AmplificationShader?.Reflection != null) CreateConstantBuffers(device, currentPass.AmplificationShader.Reflection);
                }
            }
            
            // Detect fullscreen passes (no depth = post-process, don't wireframe)
            _isFullscreenPass = !renderState.DepthTest && !renderState.DepthWrite;
        }

        public Material(PipelineState pipelineState, Shader vertexShader, Shader pixelShader, GraphicsDevice device)
        {
            _passPipelineStates[0] = pipelineState; // Custom PSO provided for pass 0

            // Reflect and Create Constant Buffers
            if (vertexShader.Reflection != null)
                CreateConstantBuffers(device, vertexShader.Reflection);
            
            if (pixelShader.Reflection != null)
                CreateConstantBuffers(device, pixelShader.Reflection);
            _built = true;
        }
        
        private void CreateConstantBuffers(GraphicsDevice device, ID3D12ShaderReflection reflection)
        {
            for (uint i = 0; i < reflection.Description.ConstantBuffers; i++)
            {
                var buffer = reflection.GetConstantBufferByIndex(i);
                var desc = buffer.Description;
                
                if (!_constantBuffers.ContainsKey(desc.Name))
                {
                    var cb = new ConstantBuffer(device, buffer);
                    
                    // Assign slot based on name (matches root signature layout)
                    cb.Slot = desc.Name switch
                    {
                        "SceneConstants" => 1,
                        "ObjectConstants" => 2,
                        "PreviewConstants" => 2,
                        "terrain" => 2,
                        "landscape" => 2,
                        "tiling" => 3,
                        _ => -1 // Not bound automatically (e.g. PushConstants)
                    };
                    
                    _constantBuffers.Add(desc.Name, cb);
                }
                else
                {
                    // Merge variables from additional shader stages.
                    // DXC optimizes out unused variables per-stage, so the DS may
                    // reference cbuffer members (e.g., MaxHeight) that the VS doesn't.
                    _constantBuffers[desc.Name].MergeVariables(buffer);
                }
            }

            for (uint i = 0; i < reflection.Description.BoundResources; i++)
            {
                var desc = reflection.GetResourceBindingDescription(i);
                if (desc.Type == Vortice.Direct3D.ShaderInputType.Texture || 
                    desc.Type == Vortice.Direct3D.ShaderInputType.Structured ||
                    desc.Type == Vortice.Direct3D.ShaderInputType.ByteAddress)
                {
                    _textureSlots[desc.Name] = (int)desc.BindPoint;
                }
            }
        }


        public void SetParameter<T>(string name, T value) where T : unmanaged
        {
            foreach (var cb in _constantBuffers.Values)
                cb.SetParameter(name, value);
        }

        public void SetParameter<T>(string name, T[] values) where T : unmanaged
        {
            foreach (var cb in _constantBuffers.Values)
            {
                cb.SetParameterArray(name, values);
            }
        }

        public void SetTexture(string name, Texture texture)
        {
            if (texture == null) return;
            _textures[name] = texture;
            UpdateMaterialData(name, texture.BindlessIndex);
            if (_effectTextureSettings.TryGetValue(name, out var setting))
                WriteSetting(setting.Offset, texture.BindlessIndex);

            // Keep TextureEffectParameter in sync (for inspector display)
            foreach (var tp in _textureParameters)
            {
                if (tp.Name == name) { tp.Value = texture; break; }
            }
            foreach (var tp in _effectTextureParameters)
            {
                if (tp.Name == name) { tp.Value = texture; break; }
            }
        }

        /// <summary>
        /// Remove a texture binding (hot reload of a definition that dropped the slot).
        /// </summary>
        public void ClearTexture(string name)
        {
            if (!_textures.Remove(name)) return;
            UpdateMaterialData(name, 0);
            if (_effectTextureSettings.TryGetValue(name, out var setting))
                WriteSetting(setting.Offset, setting.DefaultTextureAsset?.BindlessIndex ?? 0u);

            foreach (var tp in _textureParameters)
            {
                if (tp.Name == name) { tp.Value = null; break; }
            }
            foreach (var tp in _effectTextureParameters)
            {
                if (tp.Name == name) { tp.Value = null; break; }
            }
        }

        private Dictionary<string, uint> _bindlessIndices = new();

        public void SetTextureIndex(string name, uint index)
        {
            _bindlessIndices[name] = index;
        }
        
        /// <summary>
        /// Set detail map tiling scale (uniform).
        /// </summary>
        public void SetDetailTiling(float tiling)
        {
            var v = new Vector2(tiling, tiling);
            UpdateMaterialScalar((ref MaterialData d) => d.DetailTiling = v);
        }
        
        /// <summary>
        /// Set emissive color tint.
        /// </summary>
        public void SetEmissiveColor(Vector3 color)
        {
            UpdateMaterialScalar((ref MaterialData d) => d.EmissiveColor = color);
        }
        
        /// <summary>
        /// Set emissive intensity multiplier.
        /// </summary>
        public void SetEmissiveIntensity(float intensity)
        {
            _emissiveIntensity = intensity;
            ApplyEmissiveIntensity();
        }

        private float _emissiveIntensity = 1.0f;
        private bool _emissiveWhenDark;

        // Materials whose glow follows Daylight.Lamps; rewritten only when the lamps change (while they fade)
        private static readonly List<WeakReference<Material>> _lampMaterials = new();
        private static bool _lampsHooked;

        private void SetEmissiveWhenDark(bool whenDark)
        {
            _emissiveWhenDark = whenDark;
            lock (_lampMaterials)
            {
                _lampMaterials.RemoveAll(r => !r.TryGetTarget(out var m) || m == this);
                if (whenDark)
                    _lampMaterials.Add(new WeakReference<Material>(this));
                if (!_lampsHooked)
                {
                    _lampsHooked = true;
                    Daylight.LampsChanged += ApplyLamps;
                }
            }
            ApplyEmissiveIntensity();
        }

        private static void ApplyLamps()
        {
            lock (_lampMaterials)
            {
                foreach (var reference in _lampMaterials)
                    if (reference.TryGetTarget(out var material))
                        material.ApplyEmissiveIntensity();
            }
        }

        private void ApplyEmissiveIntensity()
        {
            float intensity = _emissiveIntensity * (_emissiveWhenDark ? Daylight.Lamps : 1f);
            UpdateMaterialScalar((ref MaterialData d) => d.EmissiveIntensity = intensity);
        }
        
        private delegate void MaterialDataMutator(ref MaterialData data);
        
        /// <summary>
        /// Generic helper to mutate a single field in this material's MaterialData entry.
        /// </summary>
        private void UpdateMaterialScalar(MaterialDataMutator mutator)
        {
            lock (_materialIdLock)
            {
                if (MaterialID < 0 || MaterialID >= _allMaterials.Count) return;
                var data = _allMaterials[MaterialID];
                mutator(ref data);
                _allMaterials[MaterialID] = data;
                _materialsBufferDirty = true;
            }
        }
        
        /// <summary>
        /// Update this material's data in the global MaterialsBuffer when textures change
        /// </summary>
        private void UpdateMaterialData(string name, uint bindlessIdx)
        {
            lock (_materialIdLock)
            {
                if (MaterialID < 0 || MaterialID >= _allMaterials.Count) return;
                
                var data = _allMaterials[MaterialID];
                switch (name)
                {
                    case "Albedo":
                    case "AlbedoTex":
                        data.AlbedoIdx = bindlessIdx;
                        break;
                    case "Normal":
                    case "NormalTex":
                        data.NormalIdx = bindlessIdx;
                        break;
                    case "Roughness":
                        data.RoughnessIdx = bindlessIdx;
                        break;
                    case "Metallic":
                        data.MetallicIdx = bindlessIdx;
                        break;
                    case "Emissive":
                        data.EmissiveIdx = bindlessIdx;
                        break;
                    case "AO":
                        data.AOIdx = bindlessIdx;
                        break;
                    case "DetailNormal":
                        data.DetailNormalIdx = bindlessIdx;
                        break;
                    case "DetailMask":
                        data.DetailMaskIdx = bindlessIdx;
                        break;
                }
                _allMaterials[MaterialID] = data;
                _materialsBufferDirty = true;
            }
        }



        public void Apply(CommandList commandList, GraphicsDevice device, MaterialBlock? block = null)
        {
             Apply(commandList.Native, device, block);
        }

        public void Apply(ID3D12GraphicsCommandList commandList, GraphicsDevice device, MaterialBlock? block = null)
        {
            // Apply Material Block Overrides if present
            block?.Apply(this);

            // Bind Pipeline - use selected pass's PSO, wireframe variant if enabled
            var pso = _passPipelineStates.TryGetValue(Pass, out var selectedPso) ? selectedPso : _passPipelineStates.GetValueOrDefault(0);
            if (Engine.Settings.Wireframe && !_isFullscreenPass && _passWireframePSOs.TryGetValue(Pass, out var wireframePso))
            {
                pso = wireframePso;
            }
            if (pso?.Native == null) return; // Device lost — PSO wasn't created
            commandList.SetPipelineState(pso.Native);
            commandList.SetGraphicsRootSignature(device.GlobalRootSignature);

            // Apply MaterialBlock and commit all constant buffers (Apex pattern)
            var effectBlock = Effect!.GetMaterialBlock();
            foreach (var cb in _constantBuffers.Values)
            {
                if (cb.Slot >= 0)
                {
                    // 1. Apply global MasterEffect parameters (CameraInverse, etc.)
                    effectBlock.Apply(cb);
                    // 2. Apply per-draw overrides AFTER MasterEffect (Cascades, LightSpaces, etc.)
                    //    This ensures per-draw values take priority over global defaults
                    block?.Apply(cb);
                    cb.Commit();
                    commandList.SetGraphicsRootConstantBufferView((uint)cb.Slot, cb.GpuAddress);
                }
            }

            // Keep block applied to Material for property access
            block?.Apply(this);

            // Data-driven push constants from shader resource bindings
            // Iterate bindings parsed from shader #define patterns
            foreach (var binding in Effect.ResourceBindings)
            {
                uint index = 0;
                
                // Try explicit bindless indices first (set via SetTextureIndex)
                if (_bindlessIndices.TryGetValue(binding.Name, out var bindlessIdx))
                {
                    index = bindlessIdx;
                }
                // Then try textures (set via SetTexture)
                else if (_textures.TryGetValue(binding.Name, out var tex))
                {
                    index = tex.BindlessIndex;
                }
                
                commandList.SetGraphicsRoot32BitConstant(0, index, (uint)binding.Slot);
            }
            
            // Slot 14: Materials buffer bindless index (for GET_MATERIAL macro)
            // Slots 2-13 are used by mesh rendering, so Materials uses 14 to avoid collision
            commandList.SetGraphicsRoot32BitConstant(0, _materialsBufferBindlessIndex, 14);
            
            // NOTE: SetDescriptorHeaps is now called ONCE per frame in RenderView.Prepare
            // Calling it here per-draw was extremely expensive
        }
        
        /// <summary>
        /// Bind the shared MaterialsBuffer to root slot. Call once per frame after materials are set up.
        /// Returns the bindless index for the Materials buffer (to be set as push constant slot 10).
        /// </summary>
        private static uint _materialsBufferBindlessIndex = 0;
        private static int _materialsBufferCapacity = 0;
        
        public static uint BindMaterialsBuffer(ID3D12GraphicsCommandList commandList, GraphicsDevice device)
        {
            int materialCount;
            bool dirty;
            bool needsRecreate;
            lock (_materialIdLock)
            {
                materialCount = _allMaterials.Count;
                dirty = _materialsBufferDirty;
                needsRecreate = _materialsBuffer == null || materialCount > _materialsBufferCapacity;
            }
            
            if (materialCount == 0) return 0;
            
            int bufferSize = materialCount * System.Runtime.InteropServices.Marshal.SizeOf<MaterialData>();
            
            if (needsRecreate)
            {
                // Use power of 2 growth strategy for capacity
                int newCapacity = Math.Max(32, materialCount);
                newCapacity = (int)Math.Pow(2, Math.Ceiling(Math.Log2(newCapacity)));
                
                int newBufferSize = newCapacity * System.Runtime.InteropServices.Marshal.SizeOf<MaterialData>();
                
                _materialsBuffer?.Dispose();
                _materialsBuffer = device.NativeDevice.CreateCommittedResource(
                    new HeapProperties(HeapType.Upload),
                    HeapFlags.None,
                    ResourceDescription.Buffer((ulong)newBufferSize),
                    ResourceStates.GenericRead);
                
                _materialsBufferCapacity = newCapacity;
                
                unsafe
                {
                    void* ptr;
                    _materialsBuffer.Map(0, null, &ptr);
                    _materialsBufferPtr = (IntPtr)ptr;
                }
                
                // Allocate bindless index and create SRV for materials buffer
                if (_materialsBufferBindlessIndex == 0)
                {
                    _materialsBufferBindlessIndex = device.AllocateBindlessIndex();
                }
                device.CreateStructuredBufferSRV(
                    _materialsBuffer,
                    (uint)newCapacity,
                    (uint)System.Runtime.InteropServices.Marshal.SizeOf<MaterialData>(),
                    _materialsBufferBindlessIndex);
                    
                // Force data update after recreate
                dirty = true;
            }

            // The effect settings beside it, at the same capacity; making that buffer the first time gives every
            // MaterialData its index, so the materials are uploaded again
            BindSettingsBuffer(device, _materialsBufferCapacity);
            lock (_materialIdLock) dirty |= _materialsBufferDirty;

            // Update buffer data only when dirty — zero-alloc via CollectionsMarshal.AsSpan
            if (dirty)
            {
                lock (_materialIdLock)
                {
                    var span = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_allMaterials);
                    var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(span);
                    unsafe
                    {
                        fixed (byte* src = bytes)
                        {
                            Buffer.MemoryCopy(src, (void*)_materialsBufferPtr, bufferSize, bufferSize);
                        }
                    }
                    _materialsBufferDirty = false;
                }
            }
            
            return _materialsBufferBindlessIndex;
        }
        
        /// <summary>
        /// Get the bindless index for the Materials buffer (for push constant slot 10).
        /// </summary>
        public static uint MaterialsBufferIndex => _materialsBufferBindlessIndex;

        public void Dispose()
        {
            // Frames in flight may still draw with these buffers: release them when those frames are done
            var device = Freefall.Engine.Device;
            foreach (var cb in _constantBuffers.Values)
            {
                if (device != null) device.DeferDispose(cb);
                else cb.Dispose();
            }
            _constantBuffers.Clear();
        }
        

    }

    public interface IParameterTarget
    {
        void SetParameter<T>(string name, T value) where T : unmanaged;
        void SetTexture(string name, Texture texture);
    }
    
    // Extensions to make Material implement IParameterTarget explicitly or implicitly
    // Since Material already has SetParameter/SetTexture, we can align it easily.
    
    public abstract class ParameterValue
    {
        public string Name = string.Empty;
        /// <summary>
        /// Push constant slot for per-instance GPU buffer binding.
        /// -1 = not yet resolved. Resolved automatically at enqueue time from shader resource bindings.
        /// </summary>
        public int PushConstantSlot = -1;
        public abstract void SetToTarget(Material target);
        public abstract void SetToTarget(ConstantBuffer cb);
        
        /// <summary>Copy raw bytes of the value into dest at the given offset. Returns bytes written.</summary>
        public virtual int CopyToStaging(byte[] dest, int offset) => 0;
        /// <summary>Number of array elements (for ArrayParameterValue). 0 for scalars/textures.</summary>
        public virtual int GetElementCount() => 0;
        /// <summary>Byte stride per array element. 0 for scalars/textures.</summary>
        public virtual int GetElementStride() => 0;
    }

    public sealed class ParameterValue<T> : ParameterValue where T : unmanaged
    {
        public T Value;
        public override void SetToTarget(Material target) => target.SetParameter(Name, Value);
        public override void SetToTarget(ConstantBuffer cb) => cb.SetParameter(Name, Value);
        
        public override int GetElementCount() => 1;
        public override int GetElementStride() => System.Runtime.InteropServices.Marshal.SizeOf<T>();
        public override int CopyToStaging(byte[] dest, int offset)
        {
            int size = System.Runtime.InteropServices.Marshal.SizeOf<T>();
            unsafe
            {
                fixed (T* ptr = &Value)
                {
                    System.Runtime.InteropServices.Marshal.Copy((IntPtr)ptr, dest, offset, size);
                }
            }
            return size;
        }
    }
    
    // For object types like Texture (not unmanaged)
    public sealed class TextureParameterValue : ParameterValue
    {
        public Texture Value = null!;
        public override void SetToTarget(Material target) => target.SetTexture(Name, Value);
        public override void SetToTarget(ConstantBuffer cb) { } // Textures don't apply to CBs
    }
    
    // For array types like Matrix4x4[], Vector4[]
    public sealed class ArrayParameterValue<T> : ParameterValue where T : unmanaged
    {
        public T[] Value = null!;
        public override void SetToTarget(Material target) { } // Arrays don't apply to materials directly
        public override void SetToTarget(ConstantBuffer cb) => cb.SetParameterArray(Name, Value);
        
        public override int GetElementCount() => Value?.Length ?? 0;
        public override int GetElementStride() => System.Runtime.InteropServices.Marshal.SizeOf<T>();
        public override int CopyToStaging(byte[] dest, int offset)
        {
            if (Value == null || Value.Length == 0) return 0;
            var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(Value.AsSpan());
            bytes.CopyTo(dest.AsSpan(offset));
            return bytes.Length;
        }
    }

    public class MaterialBlock
    {
        private Dictionary<int, ParameterValue> _parameters = new();

        public Dictionary<int, ParameterValue> Parameters => _parameters;

        /// <summary>
        /// Raised by every Set*/Clear. Persistent (GPU-resident) draws copy the values once at
        /// registration; their renderer subscribes to copy them again. Mutating a value in place
        /// (through Parameters, or by writing into an array passed to SetParameterArray) does not raise it.
        /// </summary>
        public event Action? Changed;

        public void Clear()
        {
            _parameters.Clear();
            Changed?.Invoke();
        }

        public void SetParameter<T>(string name, T value) where T : unmanaged
        {
            SetParameterCore(name, value);
            Changed?.Invoke();
        }

        private void SetParameterCore<T>(string name, T value) where T : unmanaged
        {
            int hash = name.GetHashCode();
            if (!_parameters.TryGetValue(hash, out var param))
            {
                param = new ParameterValue<T> { Name = name, Value = value };
                _parameters.Add(hash, param);
            }
            else if (param is ParameterValue<T> typedParam)
            {
                typedParam.Value = value;
            }
        }

        public void SetTexture(string name, Texture value)
        {
            // No Changed: textures are not staged per instance
            int hash = name.GetHashCode();
            if (!_parameters.TryGetValue(hash, out var param))
            {
                param = new TextureParameterValue { Name = name, Value = value };
                _parameters.Add(hash, param);
            }
            else if (param is TextureParameterValue typedParam)
            {
                typedParam.Value = value;
            }
        }
        
        // Support for unmanaged array types like Matrix4x4[], Vector4[]
        public void SetParameterArray<T>(string name, T[] value) where T : unmanaged
        {
            int hash = name.GetHashCode();
            if (!_parameters.TryGetValue(hash, out var param))
            {
                param = new ArrayParameterValue<T> { Name = name, Value = value };
                _parameters.Add(hash, param);
            }
            else if (param is ArrayParameterValue<T> typedParam)
            {
                typedParam.Value = value;
            }
            Changed?.Invoke();
        }
        
        public T? GetValue<T>(string name)
        {
            return GetValue<T>(name.GetHashCode());
        }
        
        public T? GetValue<T>(int hash)
        {
            if (_parameters.TryGetValue(hash, out var param))
            {
                // Use reflection to get the Value field
                var valueField = param.GetType().GetField("Value");
                if (valueField != null)
                {
                    var value = valueField.GetValue(param);
                    if (value is T result)
                        return result;
                }
            }
            return default;
        }
        
        public T[]? GetArrayValue<T>(int hash) where T : unmanaged
        {
            if (_parameters.TryGetValue(hash, out var param) && param is ArrayParameterValue<T> typed)
                return typed.Value;
            return null;
        }
        
        public bool TryGetParameter<T>(string name, out T value) where T : unmanaged
        {
            int hash = name.GetHashCode();
            if (_parameters.TryGetValue(hash, out var param) && param is ParameterValue<T> typedParam)
            {
                value = typedParam.Value;
                return true;
            }
            value = default;
            return false;
        }

        public void Apply(Material target)
        {
            foreach (var param in _parameters.Values)
                param.SetToTarget(target);
        }
        
        /// <summary>
        /// Apply all parameters to a ConstantBuffer (Apex pattern)
        /// </summary>
        public void Apply(ConstantBuffer cb)
        {
            foreach (var param in _parameters.Values)
                param.SetToTarget(cb);
        }
    }
}
