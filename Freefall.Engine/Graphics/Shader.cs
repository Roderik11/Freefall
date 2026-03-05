using System;
using Vortice.Direct3D;
using Vortice.Dxc;
using Vortice.Direct3D12; // Added this using directive
using Vortice.Direct3D12.Shader;

namespace Freefall.Graphics
{
    public class Shader : IDisposable
    {
        private byte[] _bytecode;
        public byte[] Bytecode => _bytecode;

        public ID3D12ShaderReflection? Reflection { get; private set; }

        public Shader(string source, string entryPoint, string profile, string? includeDir = null)
        {
            _bytecode = Compile(source, entryPoint, profile, includeDir);
            Reflect();
        }

        private byte[] Compile(string source, string entryPoint, string profile, string? includeDir = null)
        {
            // Manually resolve #include directives (DxcCompiler.Compile(string) has no file context)
            if (!string.IsNullOrEmpty(includeDir))
            {
                source = ResolveIncludes(source, includeDir);
            }
            
            // DXC requires a specific profile format like vs_6_6, ps_6_6
            string dxcProfile = profile;
            if (profile.Contains("_5_")) dxcProfile = profile.Replace("_5_0", "_6_6").Replace("_5_1", "_6_6");
            else if (profile.Contains("_6_0")) dxcProfile = profile.Replace("_6_0", "_6_6");
            else if (profile.Contains("_6_1")) dxcProfile = profile.Replace("_6_1", "_6_6");
            else if (profile.Contains("_6_2")) dxcProfile = profile.Replace("_6_2", "_6_6");
            else if (profile.Contains("_6_3")) dxcProfile = profile.Replace("_6_3", "_6_6");
            else if (profile.Contains("_6_4")) dxcProfile = profile.Replace("_6_4", "_6_6");
            else if (profile.Contains("_6_5")) dxcProfile = profile.Replace("_6_5", "_6_6");

            string[] args = new[]
            {
                "-E", entryPoint,
                "-T", dxcProfile,
                "-O3",
                "-Zi",
                "-Qembed_debug"
            };

            IDxcResult result = DxcCompiler.Compile(source, args);

            if (result.GetStatus().Failure)
            {
                string errors = result.GetErrors();
                Debug.Log($"[Shader] Compilation FAILED: {entryPoint} ({profile})");
                Debug.Log($"[Shader] Errors: {errors}");
                throw new Exception($"Shader compilation failed: {errors}");
            }

            using var blob = result.GetResult();
            return blob.AsSpan().ToArray();
        }

        private void Reflect()
        {
            try
            {
                using IDxcUtils utils = Dxc.CreateDxcUtils();
                unsafe
                {
                    fixed (byte* ptr = _bytecode)
                    {
                        using IDxcBlob blob = utils.CreateBlob((IntPtr)ptr, (uint)_bytecode.Length, 0);
                        Reflection = utils.CreateReflection<ID3D12ShaderReflection>(blob);
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("Shader", $"Could not create shader reflection: {ex.Message}");
            }
        }

        /// <summary>
        /// Manually resolve #include "filename" directives by inlining file contents.
        /// DxcCompiler.Compile(string) has no file context, so includes must be pre-resolved.
        /// </summary>
        private static string ResolveIncludes(string source, string baseDir, System.Collections.Generic.HashSet<string>? visited = null)
        {
            visited ??= new System.Collections.Generic.HashSet<string>(StringComparer.OrdinalIgnoreCase);
            
            return System.Text.RegularExpressions.Regex.Replace(source, 
                @"^\s*#include\s+""([^""]+)""", 
                match =>
                {
                    string filename = match.Groups[1].Value;
                    string fullPath = System.IO.Path.Combine(baseDir, filename);
                    
                    if (!System.IO.File.Exists(fullPath))
                    {
                        Debug.LogWarning("Shader", $"Include file not found: {fullPath}");
                        return $"// #include \"{filename}\" — NOT FOUND";
                    }
                    
                    if (!visited.Add(fullPath))
                        return $"// #include \"{filename}\" — already included (cycle)";
                    
                    string content = System.IO.File.ReadAllText(fullPath);
                    return ResolveIncludes(content, System.IO.Path.GetDirectoryName(fullPath)!, visited);
                },
                System.Text.RegularExpressions.RegexOptions.Multiline);
        }

        public void Dispose()
        {
            Reflection?.Dispose();
        }
    }
}
