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
        public string? RasterizerStateName { get; private set; }

        public EffectPass(string name, EffectPassDescription desc, string source)
        {
            Name = name;
            RasterizerStateName = desc.RasterizerState;

            if (!string.IsNullOrEmpty(desc.VertexShaderEntry))
                VertexShader = new Shader(source, desc.VertexShaderEntry, desc.VertexShaderProfile ?? "vs_6_0");

            if (!string.IsNullOrEmpty(desc.PixelShaderEntry))
                PixelShader = new Shader(source, desc.PixelShaderEntry, desc.PixelShaderProfile ?? "ps_6_0");

            if (!string.IsNullOrEmpty(desc.HullShaderEntry))
                HullShader = new Shader(source, desc.HullShaderEntry, desc.HullShaderProfile ?? "hs_6_0");

            if (!string.IsNullOrEmpty(desc.DomainShaderEntry))
                DomainShader = new Shader(source, desc.DomainShaderEntry, desc.DomainShaderProfile ?? "ds_6_0");
        }

        public void Dispose()
        {
            VertexShader?.Dispose();
            PixelShader?.Dispose();
            HullShader?.Dispose();
            DomainShader?.Dispose();
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

    public class Effect : Asset, IDisposable
    {
        public new string Name { get; }
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
        
        // MasterEffects: Global dictionary of all loaded effects (Apex pattern)
        // Camera.SetShaderParams iterates this to set View/Projection on all effects
        internal static Dictionary<int, Effect> MasterEffects = new Dictionary<int, Effect>();
        
        // MaterialBlock for per-effect parameters set via SetParameter
        private MaterialBlock _materialBlock = new MaterialBlock();
        
        public Effect(string filename)
        {
            Name = filename;
            int hash = filename.GetHashCode();
            
            // If already loaded, reuse (Apex pattern)
            if (MasterEffects.TryGetValue(hash, out var existing))
            {
                // Copy references from existing
                Techniques = existing.Techniques;
                ResourceBindings = existing.ResourceBindings;
                _resourceSlots = existing._resourceSlots;
                RenderState = existing.RenderState;
                _materialBlock = existing._materialBlock;
                return;
            }
            
            string resourcesPath = Path.Combine(AppContext.BaseDirectory, "Resources", "Shaders");
            string fullPath = Path.Combine(resourcesPath, filename + ".fx");
            
            if (!File.Exists(fullPath))
            {
                 // Fallback to project root if not found in Shaders folder (for dev convenience)
                 fullPath = Path.Combine(AppContext.BaseDirectory, filename + ".fx");
            }

            if (!File.Exists(fullPath))
                throw new FileNotFoundException($"Effect file {filename}.fx not found at {fullPath}");

            string content = LoadWithIncludes(fullPath);
            var techDescs = FXParser.ParseFx(content);

            foreach (var td in techDescs)
            {
                Techniques.Add(new EffectTechnique(td.Name, td, content));
            }
            
            // Parse resource bindings from shader source (data-driven push constants)
            ResourceBindings = FXParser.ParseResourceBindings(content);
            
            // Build hash→slot lookup for O(1) resolution at enqueue time
            foreach (var binding in ResourceBindings)
                _resourceSlots[binding.Name.GetHashCode()] = binding.Slot;
            
            // Parse render state metadata from shader source (data-driven PSO creation)
            RenderState = FXParser.ParseRenderState(content);
            
            // Register in MasterEffects (Apex pattern)
            MasterEffects[hash] = this;
        }
        
        /// <summary>
        /// Set parameter on this effect's MaterialBlock (will be applied during render)
        /// </summary>
        public void SetParameter<T>(string name, T value) where T : unmanaged
        {
            _materialBlock.SetParameter(name, value);
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

        private string LoadWithIncludes(string path)
        {
            string content = File.ReadAllText(path);
            string directory = Path.GetDirectoryName(path) ?? "";

            // Simple include resolver (regex search for #include "...")
            content = System.Text.RegularExpressions.Regex.Replace(content, @"#include\s+""([^""]+)""", match =>
            {
                string includeFile = match.Groups[1].Value;
                string includePath = Path.Combine(directory, includeFile);
                if (File.Exists(includePath))
                {
                    return LoadWithIncludes(includePath);
                }
                return match.Value; // Leave it if not found, DXC might handle it or error
            });

            // Patch legacy HLSL for DXC compatibility
            // Ternary on vectors is not allowed in DXC, use select()
            content = content.Replace("float4 a = n1 >= 0 ? -1 : 1;", "float4 a = select(n1 >= 0, -1.0f.xxxx, 1.0f.xxxx);");
            content = content.Replace("float4 b = n1 >= 0 ? 1 : 0;", "float4 b = select(n1 >= 0, 1.0f.xxxx, 0.0f.xxxx);");

            // No longer patching structs because it breaks other functions in the file.

            return content;
        }

        public void Dispose()
        {
            foreach (var tech in Techniques)
                tech.Dispose();
        }
    }
}
