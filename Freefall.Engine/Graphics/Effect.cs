using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Freefall.Assets;

namespace Freefall.Graphics
{
    public class EffectPass : IDisposable
    {
        public string Name { get; }
        public Shader? VertexShader { get; private set; }
        public Shader? PixelShader { get; private set; }
        public Shader? HullShader { get; private set; }
        public Shader? DomainShader { get; private set; }
        public Shader? MeshShader { get; private set; }
        public Shader? AmplificationShader { get; private set; }
        public string? RasterizerStateName { get; private set; }
        /// <summary>Per-pass render state override. Null = use global Effect.RenderState.</summary>
        public ShaderRenderState? RenderState { get; private set; }
        public bool IsMeshShaderPass => MeshShader != null;

        public EffectPass(string name, EffectPassDescription desc, string source)
        {
            Name = name;
            RasterizerStateName = desc.RasterizerState;

            // Build per-pass render state override if any fields were specified
            if (desc.RenderTargetCount.HasValue || desc.DepthTest.HasValue || desc.DepthWrite.HasValue || desc.DepthFunc != null || desc.BlendMode != null)
            {
                RenderState = new ShaderRenderState();
                if (desc.RenderTargetCount.HasValue) RenderState.RenderTargetCount = desc.RenderTargetCount.Value;
                if (desc.DepthTest.HasValue) RenderState.DepthTest = desc.DepthTest.Value;
                if (desc.DepthWrite.HasValue) RenderState.DepthWrite = desc.DepthWrite.Value;
                if (desc.DepthFunc != null) RenderState.DepthFunc = desc.DepthFunc;
                if (desc.BlendMode != null) RenderState.BlendMode = desc.BlendMode;
            }

            if (!string.IsNullOrEmpty(desc.VertexShaderEntry))
                VertexShader = new Shader(source, desc.VertexShaderEntry, desc.VertexShaderProfile ?? "vs_6_0");

            if (!string.IsNullOrEmpty(desc.PixelShaderEntry))
                PixelShader = new Shader(source, desc.PixelShaderEntry, desc.PixelShaderProfile ?? "ps_6_0");

            if (!string.IsNullOrEmpty(desc.HullShaderEntry))
                HullShader = new Shader(source, desc.HullShaderEntry, desc.HullShaderProfile ?? "hs_6_0");

            if (!string.IsNullOrEmpty(desc.DomainShaderEntry))
                DomainShader = new Shader(source, desc.DomainShaderEntry, desc.DomainShaderProfile ?? "ds_6_0");

            if (!string.IsNullOrEmpty(desc.MeshShaderEntry))
                MeshShader = new Shader(source, desc.MeshShaderEntry, desc.MeshShaderProfile ?? "ms_6_5");

            if (!string.IsNullOrEmpty(desc.AmplificationShaderEntry))
                AmplificationShader = new Shader(source, desc.AmplificationShaderEntry, desc.AmplificationShaderProfile ?? "as_6_5");
        }

        public void Dispose()
        {
            VertexShader?.Dispose();
            PixelShader?.Dispose();
            HullShader?.Dispose();
            DomainShader?.Dispose();
            MeshShader?.Dispose();
            AmplificationShader?.Dispose();
        }
    }

    public class EffectTechnique : IDisposable
    {
        public string Name { get; }
        public List<EffectPass> Passes { get; } = new List<EffectPass>();

        public EffectTechnique(string name, EffectTechniqueDescription desc, string source)
        {
            Name = name;
            foreach (var passDesc in desc.Passes)
            {
                Passes.Add(new EffectPass(passDesc.Name, passDesc, source));
            }
        }

        public void Dispose()
        {
            foreach (var pass in Passes)
                pass.Dispose();
        }
    }

    [AssetTypeAlias("EffectSourceData")]   // a project's .fx asset (EffectImporter)
    public class Effect : Asset, IDisposable
    {
        public List<EffectTechnique> Techniques { get; } = new List<EffectTechnique>();
        
        // Render state overrides (used when creating PSOs)
        public BlendState BlendState { get; set; } = BlendState.Opaque;
        public DepthStencilState DepthStencilState { get; set; } = DepthStencilState.Default;
        public RasterizerState RasterizerState { get; set; } = RasterizerState.BackCull;
        
        // Resource bindings parsed from shader source (data-driven push constants)
        public List<ShaderResourceBinding> ResourceBindings { get; private set; } = new List<ShaderResourceBinding>();
        
        // Hash→slot dictionary for O(1) push constant slot lookup by param name hash
        private Dictionary<int, int> _resourceSlots = new();
        
        // Render state metadata parsed from shader source (data-driven PSO creation)
        public ShaderRenderState RenderState { get; private set; } = new ShaderRenderState();

        /// <summary>
        /// Settings the effect declares for its materials ("//=" defaults in its ObjectConstants cbuffer,
        /// FXParser.ParseMaterialParams). Shared by every instance of the effect and refilled in place on reload.
        /// </summary>
        public List<MaterialParamDesc> MaterialParams { get; private set; } = new();

        /// <summary>
        /// Values the effect takes per object ("//=" members of its struct Instance, FXParser.ParseInstanceParams):
        /// a renderer's slot of the instance data buffer (InstanceData). Shared like MaterialParams.
        /// </summary>
        public List<MaterialParamDesc> InstanceParams { get; private set; } = new();

        /// <summary>
        /// A slot of InstanceData holding the instance defaults, for draws that bring no slot of their own
        /// (0 when the effect declares no struct Instance). Shared by every instance of the effect.
        /// </summary>
        public int DefaultInstanceSlot => _defaultInstanceSlot[0];
        private int[] _defaultInstanceSlot = new int[1];

        // MasterEffects: Global dictionary of all loaded effects (Apex pattern)
        // Camera.SetShaderParams iterates this to set View/Projection on all effects
        internal static Dictionary<int, Effect> MasterEffects = new Dictionary<int, Effect>();
        
        // MaterialBlock for per-effect parameters set via SetParameter
        private MaterialBlock _materialBlock = new MaterialBlock();

        /// <summary>
        /// What identifies the effect: its file name for an engine effect ("vfx"), "asset:" + GUID for a project's
        /// .fx asset (<see cref="FromAsset"/>). Instances of one key share its compiled state (MasterEffects), and
        /// hot reload and material rebuilds match by it, not by Name: a project "water.fx" is not the engine's water.
        /// </summary>
        public string Key { get; private set; }

        /// <summary>
        /// DXC's messages when the last compile failed (they name the real file and line), else null. A project
        /// effect whose first compile failed has no techniques: its materials draw nothing until it compiles.
        /// </summary>
        public string? LastError { get; private set; }

        public Effect(string filename)
        {
            Name = filename;
            Key = filename;
            int hash = filename.GetHashCode();

            // If already loaded, reuse (Apex pattern)
            if (MasterEffects.TryGetValue(hash, out var existing))
            {
                // Copy references from existing
                Techniques = existing.Techniques;
                ResourceBindings = existing.ResourceBindings;
                _resourceSlots = existing._resourceSlots;
                RenderState = existing.RenderState;
                MaterialParams = existing.MaterialParams;
                InstanceParams = existing.InstanceParams;
                _defaultInstanceSlot = existing._defaultInstanceSlot;
                _materialBlock = existing._materialBlock;
                lock (_instances) _instances.Add(new WeakReference<Effect>(this));
                return;
            }
            
            _sourcePath = ShaderHotReload.ResolvePath(filename + ".fx")
                ?? throw new FileNotFoundException($"Effect file {filename}.fx not found in {ShaderHotReload.Directory}");

            string content = LoadWithIncludes(_sourcePath);
            var techDescs = FXParser.ParseFx(content);

            foreach (var td in techDescs)
            {
                Techniques.Add(new EffectTechnique(td.Name, td, content));
            }

            // Discover push constant slot mappings via shader reflection
            // (same pattern as ComputeShader.DiscoverPushConstants)
            DiscoverPushConstants();

            // Parse render state metadata from shader source (data-driven PSO creation)
            RenderState = FXParser.ParseRenderState(content);
            MaterialParams.AddRange(FXParser.ParseMaterialParams(content));
            InstanceParams.AddRange(FXParser.ParseInstanceParams(content));
            WriteInstanceDefaults();

            Dependencies = ShaderHotReload.CollectDependencies(_sourcePath);

            // Register in MasterEffects (Apex pattern)
            MasterEffects[hash] = this;
            lock (_instances) _instances.Add(new WeakReference<Effect>(this));
            ShaderHotReload.EnsureWatching();
        }

        /// <summary>The key of a project's .fx asset (<see cref="Key"/>).</summary>
        public static string AssetKey(string guid) => "asset:" + guid;

        /// <summary>
        /// The effect of a project's .fx asset (EffectLoader), compiled from <paramref name="source"/>. Its includes are
        /// looked up next to <paramref name="sourcePath"/> (the file in Assets; null when there is none), then among
        /// the engine's shaders, so <c>#include "common.fx"</c> is the engine's. One effect per GUID: a second load
        /// returns the first, recompiled if the source differs. A source that does not compile still gives the
        /// effect, without techniques (<see cref="LastError"/>, logged): its materials keep the reference and their
        /// settings, draw nothing, and come back when a save compiles (hot reload).
        /// </summary>
        public static Effect FromAsset(string guid, string name, string source, string? sourcePath)
        {
            var key = AssetKey(guid);
            if (MasterEffects.TryGetValue(key.GetHashCode(), out var existing))
            {
                existing.Name = name;
                if (existing._sourceText != source && existing.TryReloadFrom(source))
                    Material.RebuildForEffect(existing);
                return existing;
            }
            return new Effect(key, guid, name, source, sourcePath);
        }

        private Effect(string key, string guid, string name, string source, string? sourcePath)
        {
            Key = key;
            Guid = guid;
            Name = name;
            _sourcePath = sourcePath != null ? Path.GetFullPath(sourcePath) : null;
            _sourceText = source;

            // The settings and the render state are read from the text even when it does not compile: the
            // materials' saved values must survive a broken shader (a material saved meanwhile keeps them)
            string content = source;
            try
            {
                content = Flatten(source, _sourcePath);
                foreach (var td in FXParser.ParseFx(content))
                    Techniques.Add(new EffectTechnique(td.Name, td, content));
                if (Techniques.Count == 0 || Techniques[0].Passes.Count == 0)
                    throw new Exception("no technique with a pass found");
                DiscoverPushConstants();
            }
            catch (Exception ex)
            {
                foreach (var t in Techniques) t.Dispose();
                Techniques.Clear();
                ResourceBindings.Clear();
                _resourceSlots.Clear();
                LastError = ex.Message;
                Debug.LogError("Effect", $"{name}.fx does not compile; its materials draw nothing until it does: {ex.Message}");
            }

            RenderState = FXParser.ParseRenderState(content);
            MaterialParams.AddRange(FXParser.ParseMaterialParams(content));
            InstanceParams.AddRange(FXParser.ParseInstanceParams(content));
            WriteInstanceDefaults();

            if (_sourcePath != null)
                Dependencies = ShaderHotReload.CollectDependencies(_sourcePath);

            MasterEffects[key.GetHashCode()] = this;
            lock (_instances) _instances.Add(new WeakReference<Effect>(this));
            ShaderHotReload.EnsureWatching();
        }

        // ── Hot reload ──
        // Only the master (first instance of a name) owns a source path. Later instances share its
        // Techniques / ResourceBindings / slot map by reference, so refilling those in place updates
        // everyone; RenderState is a plain reference and is re-pointed on each live instance.
        private readonly string? _sourcePath;
        // The text last compiled (before includes), so a reimport of an unchanged asset compiles nothing
        private string? _sourceText;
        private static readonly List<WeakReference<Effect>> _instances = new();

        /// <summary>The .fx file the effect compiles from (full path), or null.</summary>
        public string? SourcePath => _sourcePath;

        /// <summary>The .fx file and everything it includes (full paths). Null on non-master instances.</summary>
        internal HashSet<string>? Dependencies { get; private set; }

        // The defaults slot: made the first time the effect declares instance data, rewritten on every (re)load
        private void WriteInstanceDefaults()
        {
            if (InstanceParams.Count == 0) return;

            var last = InstanceParams[^1];
            int bytes = last.Offset + last.Type switch { "float2" => 8, "float3" => 12, "float4" => 16, _ => 4 };
            if (bytes > InstanceData.Stride)
                Debug.LogWarning("Effect", $"{Name}: struct Instance is {bytes} bytes; only the first {InstanceData.Stride} reach the shader");

            if (_defaultInstanceSlot[0] == 0) _defaultInstanceSlot[0] = InstanceData.Allocate();
            InstanceData.WriteDefaults(_defaultInstanceSlot[0], InstanceParams);
        }

        /// <summary>
        /// Recompile from disk and swap the result in. Throws (leaving the effect untouched) when the
        /// source does not compile. Materials must rebuild their pipeline states afterwards
        /// (Material.RebuildForEffect).
        /// </summary>
        internal void Reload()
        {
            if (_sourcePath == null) return;
            ReloadFrom(File.ReadAllText(_sourcePath));
        }

        /// <summary>
        /// Recompile a project effect from a reimported source (EffectLoader.Reload). False when the text is the
        /// one already compiled or does not compile (logged; the previous version stays). On true, materials must
        /// rebuild (Material.RebuildForEffect).
        /// </summary>
        internal bool TryReloadFrom(string source)
        {
            if (source == _sourceText) return false;
            try
            {
                ReloadFrom(source);
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError("Effect", $"{Name}.fx not reloaded, keeping the previous version: {ex.Message}");
                return false;
            }
        }

        private void ReloadFrom(string source)
        {
            // Compile everything first: a failure must not leave a half-swapped effect
            string content;
            var techniques = new List<EffectTechnique>();
            try
            {
                content = Flatten(source, _sourcePath);
                foreach (var td in FXParser.ParseFx(content))
                    techniques.Add(new EffectTechnique(td.Name, td, content));
                if (techniques.Count == 0 || techniques[0].Passes.Count == 0)
                    throw new Exception("no technique/pass found");
            }
            catch (Exception ex)
            {
                foreach (var t in techniques) t.Dispose();
                LastError = ex.Message;
                throw;
            }
            LastError = null;
            _sourceText = source;

            var old = new List<EffectTechnique>(Techniques);
            Techniques.Clear();
            Techniques.AddRange(techniques);

            ResourceBindings.Clear();
            _resourceSlots.Clear();
            DiscoverPushConstants();

            MaterialParams.Clear();
            MaterialParams.AddRange(FXParser.ParseMaterialParams(content));
            InstanceParams.Clear();
            InstanceParams.AddRange(FXParser.ParseInstanceParams(content));
            WriteInstanceDefaults();   // renderers rewrite their own slots when their materials rebuild (EffectChanged)

            var renderState = FXParser.ParseRenderState(content);
            lock (_instances)
            {
                for (int i = _instances.Count - 1; i >= 0; i--)
                {
                    if (!_instances[i].TryGetTarget(out var instance)) { _instances.RemoveAt(i); continue; }
                    if (instance.Key == Key) instance.RenderState = renderState;
                }
            }

            if (_sourcePath != null)
                Dependencies = ShaderHotReload.CollectDependencies(_sourcePath);

            foreach (var t in old) t.Dispose();
        }
        
        /// <summary>
        /// Set parameter on this effect's MaterialBlock (will be applied during render)
        /// </summary>
        public void SetParameter<T>(string name, T value) where T : unmanaged
        {
            _materialBlock.SetParameter(name, value);
        }
        
        /// <summary>
        /// Set array parameter on this effect's MaterialBlock (will be applied during render)
        /// </summary>
        public void SetParameterArray<T>(string name, T[] value) where T : unmanaged
        {
            _materialBlock.SetParameterArray(name, value);
        }
        
        /// <summary>
        /// Get this effect's MaterialBlock for applying parameters
        /// </summary>
        public MaterialBlock GetMaterialBlock() => _materialBlock;
        
        /// <summary>
        /// Look up push constant slot by parameter name hash. Returns -1 if not found.
        /// </summary>
        public int GetPushConstantSlot(int nameHash)
            => _resourceSlots.TryGetValue(nameHash, out int slot) ? slot : -1;

        /// <summary>
        /// Discover push constant slot→name mappings from cbuffer PushConstants reflection.
        /// Reflects on the first compiled shader in the first technique/pass.
        /// Each variable's StartOffset / 4 = push constant slot index.
        /// </summary>
        private void DiscoverPushConstants()
        {
            // Find shader reflection that has a PushConstants cbuffer.
            // Must check ALL stages — some shaders only declare PushConstants in PS.
            Vortice.Direct3D12.Shader.ID3D12ShaderReflection? reflection = null;
            foreach (var tech in Techniques)
            {
                foreach (var pass in tech.Passes)
                {
                    var candidates = new[] { pass.VertexShader, pass.MeshShader, pass.PixelShader };
                    foreach (var shader in candidates)
                    {
                        if (shader?.Reflection == null) continue;
                        for (uint i = 0; i < shader.Reflection.Description.ConstantBuffers; i++)
                        {
                            var cb = shader.Reflection.GetConstantBufferByIndex(i);
                            if (cb.Description.Name == "PushConstants")
                            {
                                reflection = shader.Reflection;
                                break;
                            }
                        }
                        if (reflection != null) break;
                    }
                    if (reflection != null) break;
                }
                if (reflection != null) break;
            }

            if (reflection == null)
            {
                Debug.Log($"[Effect] '{Name}' — no shader reflection available, push constants not discovered");
                return;
            }

            for (uint i = 0; i < reflection.Description.ConstantBuffers; i++)
            {
                var cbReflection = reflection.GetConstantBufferByIndex(i);
                if (cbReflection.Description.Name != "PushConstants") continue;

                var desc = cbReflection.Description;
                for (uint v = 0; v < desc.VariableCount; v++)
                {
                    var variable = cbReflection.GetVariableByIndex(v);
                    var varDesc = variable.Description;
                    int slot = (int)(varDesc.StartOffset / 4);

                    // Skip reserved padding fields
                    string rawName = varDesc.Name;
                    if (rawName.StartsWith("_reserved") || rawName.StartsWith("_pad"))
                        continue;

                    // Strip "Idx" suffix for the semantic name (e.g. "HeightTexIdx" → "HeightTex")
                    string name = rawName;
                    if (name.EndsWith("Idx"))
                        name = name.Substring(0, name.Length - 3);

                    int hash = name.GetHashCode();
                    _resourceSlots[hash] = slot;
                    
                    // Also populate ResourceBindings for backwards compatibility
                    ResourceBindings.Add(new ShaderResourceBinding { Name = name, Slot = slot });

                    //Debug.Log($"[Effect] '{Name}' push constant: '{name}' → slot {slot}");
                }
                break; // Only one PushConstants cbuffer
            }
        }

        private static string LoadWithIncludes(string path) => Flatten(File.ReadAllText(path), path);

        // A directive starts its line: an "#include" quoted in a comment (mesh_common.fx shows how to include
        // itself) is no include
        private static readonly System.Text.RegularExpressions.Regex IncludeRegex = new(
            @"^[ \t]*#include\s+""([^""]+)""", System.Text.RegularExpressions.RegexOptions.Multiline);

        /// <summary>
        /// The source with every <c>#include "x"</c> replaced by that file, looked up next to the including file and
        /// then among the engine's shaders (ShaderHotReload.ResolveInclude), between #line markers: DXC's messages
        /// then name the real file and line, not a line of the flattened text.
        /// </summary>
        private static string Flatten(string source, string? path, int depth = 0)
        {
            if (depth > 16)
                throw new Exception($"includes nested more than 16 deep at {path} (does a file include itself?)");

            string directory = path != null ? Path.GetDirectoryName(path) ?? "" : ShaderHotReload.Directory;
            string here = LineFileName(path);

            string content = IncludeRegex.Replace(source, match =>
            {
                string? includePath = ShaderHotReload.ResolveInclude(directory, match.Groups[1].Value);
                if (includePath == null)
                    return match.Value;     // left for DXC, which reports it

                // The rest of the #include's line keeps its number; the line after the marker is that line
                int line = 1;
                for (int i = 0; i < match.Index; i++)
                    if (source[i] == '\n') line++;
                string text = Flatten(File.ReadAllText(includePath), includePath, depth + 1);
                return $"#line 1 \"{LineFileName(includePath)}\"\n{text}\n#line {line} \"{here}\"\n";
            });

            if (depth > 0) return content;

            if (path != null)
                content = $"#line 1 \"{here}\"\n" + content;

            // Patch legacy HLSL for DXC compatibility
            // Ternary on vectors is not allowed in DXC, use select()
            content = content.Replace("float4 a = n1 >= 0 ? -1 : 1;", "float4 a = select(n1 >= 0, -1.0f.xxxx, 1.0f.xxxx);");
            content = content.Replace("float4 b = n1 >= 0 ? 1 : 0;", "float4 b = select(n1 >= 0, 1.0f.xxxx, 0.0f.xxxx);");

            // No longer patching structs because it breaks other functions in the file.

            return content;
        }

        // A file name for #line: forward slashes (a backslash in its string would be an escape)
        private static string LineFileName(string? path) => path?.Replace('\\', '/') ?? "effect.fx";

        public void Dispose()
        {
            // The compiled techniques are shared by every instance of the effect and the materials using it, and
            // a registered master lives as long as MasterEffects holds it: an asset cache eviction (thumbnail
            // rendering evicts what it loaded) must not release them under live materials.
            if (MasterEffects.TryGetValue(Key.GetHashCode(), out var master) && ReferenceEquals(master.Techniques, Techniques))
                return;
            foreach (var tech in Techniques)
                tech.Dispose();
        }
    }
}
