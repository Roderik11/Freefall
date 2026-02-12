using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace Freefall.Graphics
{
    public class EffectTechniqueDescription
    {
        public string Name { get; set; } = string.Empty;
        public List<EffectPassDescription> Passes { get; set; } = new List<EffectPassDescription>();
    }

    public class EffectPassDescription
    {
        public string Name { get; set; } = string.Empty;
        public string? RasterizerState { get; set; }
        public string? VertexShaderProfile { get; set; }
        public string? VertexShaderEntry { get; set; }
        public string? PixelShaderProfile { get; set; }
        public string? PixelShaderEntry { get; set; }
        public string? GeometryShaderProfile { get; set; }
        public string? GeometryShaderEntry { get; set; }
        public string? HullShaderProfile { get; set; }
        public string? HullShaderEntry { get; set; }
        public string? DomainShaderProfile { get; set; }
        public string? DomainShaderEntry { get; set; }
    }

    public static class FXParser
    {
        public static List<EffectTechniqueDescription> ParseFx(string content)
        {
            var techniques = new List<EffectTechniqueDescription>();

            // Technique pattern with balancing groups for nested braces
            // Matches: technique11 Name { ... }
            string techniquePattern = @"technique11\s+(\w+)\s*\{((?>[^{}]+|(?<o>{)|(?<-o>}))*(?(o)(?!)))\}";
            
            // Pass pattern
            // Matches: pass Name { ... }
            string passPattern = @"pass\s+(\w+)\s*{(.*?)}";

            foreach (Match techMatch in Regex.Matches(content, techniquePattern, RegexOptions.Singleline))
            {
                var technique = new EffectTechniqueDescription
                {
                    Name = techMatch.Groups[1].Value.Trim()
                };

                // Extract pass blocks within the technique
                string techBody = techMatch.Groups[2].Value;
                foreach (Match passMatch in Regex.Matches(techBody, passPattern, RegexOptions.Singleline))
                {
                    var pass = new EffectPassDescription
                    {
                        Name = passMatch.Groups[1].Value.Trim()
                    };

                    // Parse pass body for states and shaders
                    string passBody = passMatch.Groups[2].Value;

                    // Helper to parse shader compilation: SetVertexShader(CompileShader(vs_5_0, VS()))
                    void ParseShader(string shaderType, Action<string, string> setProfileAndEntry)
                    {
                        // Regex to match: SetVertexShader(CompileShader(profile, entry()))
                        // Note: entry() might have arguments or not, usually VS() or VS(inputs)
                        // We are looking for the function name before the parenthesis
                        string pattern = $@"Set{shaderType}\s*\(\s*CompileShader\s*\(\s*(\w+)\s*,\s*(\w+)\s*\(.*?\)\s*\)\s*\)";
                        
                        var match = Regex.Match(passBody, pattern, RegexOptions.Singleline);
                        if (match.Success)
                        {
                            string profile = match.Groups[1].Value.Trim();
                            string entry = match.Groups[2].Value.Trim();
                            setProfileAndEntry(profile, entry);
                        }
                    }

                    ParseShader("VertexShader", (p, e) => { pass.VertexShaderProfile = p; pass.VertexShaderEntry = e; });
                    ParseShader("PixelShader", (p, e) => { pass.PixelShaderProfile = p; pass.PixelShaderEntry = e; });
                    ParseShader("GeometryShader", (p, e) => { pass.GeometryShaderProfile = p; pass.GeometryShaderEntry = e; });
                    ParseShader("HullShader", (p, e) => { pass.HullShaderProfile = p; pass.HullShaderEntry = e; });
                    ParseShader("DomainShader", (p, e) => { pass.DomainShaderProfile = p; pass.DomainShaderEntry = e; });

                    // RasterizerState
                    var rasterMatch = Regex.Match(passBody, @"SetRasterizerState\s*\(\s*(\w+)\s*\)");
                    if (rasterMatch.Success)
                    {
                        pass.RasterizerState = rasterMatch.Groups[1].Value.Trim();
                    }

                    technique.Passes.Add(pass);
                }

                techniques.Add(technique);
            }

            return techniques;
        }
        
        /// <summary>
        /// Parse push constant resource bindings from shader source.
        /// Extracts #define XxxIdx GET_INDEX(N) patterns.
        /// </summary>
        public static List<ShaderResourceBinding> ParseResourceBindings(string content)
        {
            var bindings = new List<ShaderResourceBinding>();
            
            // Pattern: #define AlbedoIdx GET_INDEX(0)          // Optional comment
            // Captures: (1) DefineName, (2) Slot, (3) Optional comment
            string pattern = @"#define\s+(\w+Idx)\s+GET_INDEX\((\d+)\)(?:\s*//\s*(.*))?";
            
            foreach (Match match in Regex.Matches(content, pattern))
            {
                string defineName = match.Groups[1].Value;
                int slot = int.Parse(match.Groups[2].Value);
                string? comment = match.Groups[3].Success ? match.Groups[3].Value.Trim() : null;
                
                // Derive semantic name by stripping "Idx" suffix
                // "AlbedoIdx" -> "Albedo", "LightBufferIdx" -> "LightBuffer"
                string name = defineName;
                if (name.EndsWith("Idx"))
                    name = name.Substring(0, name.Length - 3);
                
                bindings.Add(new ShaderResourceBinding
                {
                    Name = name,
                    Slot = slot,
                    DefineName = defineName,
                    Comment = comment
                });
            }
            
            return bindings;
        }
        
        /// <summary>
        /// Parse render state metadata from shader source.
        /// Extracts // @RenderState(key=value, ...) comments.
        /// Example: // @RenderState(RenderTargets=3, DepthWrite=false)
        /// </summary>
        public static ShaderRenderState ParseRenderState(string content)
        {
            var state = new ShaderRenderState();
            
            // Pattern: // @RenderState(key=value, key=value, ...)
            string pattern = @"//\s*@RenderState\s*\(([^)]+)\)";
            
            var match = Regex.Match(content, pattern);
            if (match.Success)
            {
                string args = match.Groups[1].Value;
                
                // Parse key=value pairs
                foreach (var pair in args.Split(','))
                {
                    var kv = pair.Split('=');
                    if (kv.Length == 2)
                    {
                        string key = kv[0].Trim();
                        string value = kv[1].Trim();
                        
                        switch (key)
                        {
                            case "RenderTargets":
                                state.RenderTargetCount = int.Parse(value);
                                break;
                            case "DepthWrite":
                                state.DepthWrite = bool.Parse(value);
                                break;
                            case "DepthTest":
                                state.DepthTest = bool.Parse(value);
                                break;
                            case "Blend":
                                state.BlendMode = value;
                                break;
                            case "CullMode":
                                state.CullMode = value;  // "None", "Front", "Back"
                                break;
                            case "DepthFunc":
                                state.DepthFunc = value;  // "Less", "LessEqual", "Greater", "GreaterEqual", etc.
                                break;
                        }
                    }
                }
            }
            
            return state;
        }
    }
    
    /// <summary>
    /// Render state configuration parsed from shader source.
    /// </summary>
    public class ShaderRenderState
    {
        /// <summary>Number of render targets (1 = standard, 3 = GBuffer)</summary>
        public int RenderTargetCount { get; set; } = 1;
        
        /// <summary>Whether depth writes are enabled</summary>
        public bool DepthWrite { get; set; } = true;
        
        /// <summary>Whether depth testing is enabled</summary>
        public bool DepthTest { get; set; } = true;
        
        /// <summary>Blend mode: "Opaque", "Additive", "AlphaBlend"</summary>
        public string BlendMode { get; set; } = "Opaque";
        
        /// <summary>Cull mode: "None", "Front", "Back" (default: Back)</summary>
        public string CullMode { get; set; } = "Back";
        
        /// <summary>Depth comparison function: "Less", "LessEqual", "Greater", "GreaterEqual" (default: GreaterEqual for reverse depth)</summary>
        public string DepthFunc { get; set; } = "GreaterEqual";
    }
}
