using System;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
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

        // Compiled bytecode, in memory and on disk (ShaderCache), keyed by a hash of DXC's version, the arguments and the
        // source with its includes inlined: DXC's output depends on nothing else, so a shader compiled once is never
        // compiled again. In memory: every DeferredRenderer compiles its effects and compute shaders, and the editor's
        // scene view did it all a second time after the main view (2 s on the main thread at project open). On disk:
        // every launch compiled the same 93 shaders again, 6.2 s of the editor's 11 s to its first frame (2026-10-17).
        // An edited file (hot reload) is another source, another key. Any thread.
        private static readonly ConcurrentDictionary<string, byte[]> _compiled = new();
        private static int _memoryHits, _diskHits, _cacheMisses;
        private static long _compileTicks, _cachedBytes;

        /// <summary>
        /// Since start: compiles answered from memory / from the disk cache / run through DXC, DXC's total time in ms,
        /// and the bytecode kept in memory.
        /// </summary>
        /// <summary>
        /// Seconds since DXC last compiled a shader, or infinity when it has not this run: a game's first start compiles
        /// them as they are needed (no cache yet), and its UI can say so while they come (a loading screen staying up).
        /// </summary>
        public static double SecondsSinceCompile
        {
            get
            {
                long t = Interlocked.Read(ref _lastCompile);
                return t == 0 ? double.PositiveInfinity : System.Diagnostics.Stopwatch.GetElapsedTime(t).TotalSeconds;
            }
        }

        private static long _lastCompile;

        public static (int Hits, int DiskHits, int Misses, double CompileMs, long Bytes) CacheStats =>
            (_memoryHits, _diskHits, _cacheMisses, _compileTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency,
             Interlocked.Read(ref _cachedBytes));

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

            string[] args = Arguments(entryPoint, profile);
            string key = ShaderCache.Key(args, source);
            if (_compiled.TryGetValue(key, out var cached))
            {
                Interlocked.Increment(ref _memoryHits);
                return cached;
            }

            if (ShaderCache.TryRead(key, out var bytecode))
            {
                Interlocked.Increment(ref _diskHits);
            }
            else
            {
                long start = System.Diagnostics.Stopwatch.GetTimestamp();
                bytecode = CompileWithDxc(source, args, entryPoint, profile);
                long end = System.Diagnostics.Stopwatch.GetTimestamp();
                Interlocked.Add(ref _compileTicks, end - start);
                Interlocked.Exchange(ref _lastCompile, end);
                Interlocked.Increment(ref _cacheMisses);
                ShaderCache.Write(key, bytecode);
            }

            if (_compiled.TryAdd(key, bytecode))
                Interlocked.Add(ref _cachedBytes, bytecode.Length);
            return bytecode;
        }

        /// <summary>DXC's arguments for a shader (part of the cache key: changing them here recompiles everything).</summary>
        private static string[] Arguments(string entryPoint, string profile)
        {
            // DXC requires a specific profile format like vs_6_6, ps_6_6
            string dxcProfile = profile;
            if (profile.Contains("_5_")) dxcProfile = profile.Replace("_5_0", "_6_6").Replace("_5_1", "_6_6");
            else if (profile.Contains("_6_0")) dxcProfile = profile.Replace("_6_0", "_6_6");
            else if (profile.Contains("_6_1")) dxcProfile = profile.Replace("_6_1", "_6_6");
            else if (profile.Contains("_6_2")) dxcProfile = profile.Replace("_6_2", "_6_6");
            else if (profile.Contains("_6_3")) dxcProfile = profile.Replace("_6_3", "_6_6");
            else if (profile.Contains("_6_4")) dxcProfile = profile.Replace("_6_4", "_6_6");
            else if (profile.Contains("_6_5")) dxcProfile = profile.Replace("_6_5", "_6_6");

            return new[]
            {
                "-E", entryPoint,
                "-T", dxcProfile,
                "-O3",
                "-Zi",
                "-Qembed_debug"
            };
        }

        private static byte[] CompileWithDxc(string source, string[] args, string entryPoint, string profile)
        {
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
                    
                    if (!Freefall.Assets.EngineFiles.Exists(fullPath))
                    {
                        Debug.LogWarning("Shader", $"Include file not found: {fullPath}");
                        return $"// #include \"{filename}\" — NOT FOUND";
                    }
                    
                    if (!visited.Add(fullPath))
                        return $"// #include \"{filename}\" — already included (cycle)";
                    
                    string content = Freefall.Assets.EngineFiles.ReadAllText(fullPath);
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
