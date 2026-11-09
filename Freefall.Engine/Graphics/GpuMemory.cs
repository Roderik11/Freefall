using System;
using System.Collections.Generic;
using System.Linq;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// A ledger of the committed resources the engine makes (GraphicsDevice.CreateCommittedResource): each one's
    /// allocation size as D3D12 counts it (a buffer takes at least 64 KiB), its heap (default = video memory, upload and
    /// readback = system memory on a discrete GPU) and who made it. The maker is the calling method, or, for what the
    /// generic helpers make (GraphicsDevice's texture and buffer helpers, GraphicsBuffer, RenderTarget, ConstantBuffer,
    /// InstanceData), the first method outside them on the stack, looked up for a resource in video memory of 256 KiB
    /// or more (smaller ones count under the helper). Resources are held weakly; a disposed one drops out at the next
    /// report. Not counted: the swap chain, descriptor heaps, pipeline states, command allocators, the driver's own.
    /// For engine_stats' gpuMemory.
    /// </summary>
    public static class GpuMemory
    {
        private sealed class Entry
        {
            public WeakReference<ID3D12Resource> Resource = null!;
            public long Bytes;
            public HeapType Heap;
            public string Owner = "";
            public string? Detail;
        }

        private static readonly List<Entry> _entries = new();
        private static readonly object _lock = new();

        // Their resources belong to whoever called them
        private static readonly HashSet<string> Helpers = new(StringComparer.Ordinal)
        {
            nameof(GraphicsDevice), nameof(GraphicsBuffer), nameof(RenderTarget), nameof(ConstantBuffer), "InstanceData",
            nameof(GpuMemory), nameof(RenderTexture2D), nameof(DepthTextureArray2D), "DepthTexture2D", "RenderTexture3D",
        };

        private const long StackLookupBytes = 256 * 1024;

        /// <summary>
        /// Record a resource just made. <paramref name="owner"/> is "Type.Member" of the call site (or a label it gave);
        /// <paramref name="detail"/> what it is (a texture's name). Any thread.
        /// </summary>
        internal static void Track(ID3D12Device device, ID3D12Resource resource, HeapType heap, string owner, string? detail)
        {
            if (resource == null) return;
            long bytes;
            try { bytes = (long)device.GetResourceAllocationInfo(0, resource.Description).SizeInBytes; }
            catch { return; }

            int dot = owner.IndexOf('.');
            if (dot > 0 && Helpers.Contains(owner[..dot]))
                owner = heap == HeapType.Default && bytes >= StackLookupBytes ? OwnerOnStack() ?? owner : owner + " (small)";

            var entry = new Entry { Resource = new WeakReference<ID3D12Resource>(resource), Bytes = bytes, Heap = heap, Owner = owner, Detail = detail };
            lock (_lock) _entries.Add(entry);
        }

        // The first method on the stack outside the helpers and the runtime: "Type.Method" (a lambda as its method)
        private static string? OwnerOnStack()
        {
            foreach (var frame in new System.Diagnostics.StackTrace(2, false).GetFrames())
            {
                var method = frame.GetMethod();
                var type = method?.DeclaringType;
                while (type is { IsNested: true } && type.Name.StartsWith('<')) type = type.DeclaringType;  // closures
                if (type == null || Helpers.Contains(type.Name)) continue;
                if (type.Namespace is { } ns && (ns.StartsWith("System", StringComparison.Ordinal) || ns.StartsWith("Vortice", StringComparison.Ordinal)))
                    continue;
                var name = method!.Name;
                if (name.StartsWith('<') && name.IndexOf('>') is var end && end > 1) name = name[1..end];   // <Draw>b__12_0
                return $"{type.Name}.{name}";
            }
            return null;
        }

        private static bool Alive(Entry e) => e.Resource.TryGetTarget(out var r) && r.NativePointer != IntPtr.Zero;

        /// <summary>
        /// The ledger as it stands: video and system memory tracked, the process's video memory as the OS counts it and
        /// what of it the ledger does not see, video memory by owner (largest first) and the largest single resources,
        /// texture assets by folder. Main thread (reads the asset cache).
        /// </summary>
        public static object Describe(GraphicsDevice device, int top = 40)
        {
            List<Entry> live;
            lock (_lock)
            {
                _entries.RemoveAll(e => !Alive(e));
                live = new List<Entry>(_entries);
            }

            static double MiB(long b) => Math.Round(b / 1048576.0, 1);
            var vram = live.Where(e => e.Heap == HeapType.Default).ToList();
            long vramBytes = vram.Sum(e => e.Bytes);
            long systemBytes = live.Where(e => e.Heap != HeapType.Default).Sum(e => e.Bytes);

            var byOwner = vram.GroupBy(e => e.Owner)
                .Select(g => (owner: g.Key, bytes: g.Sum(e => e.Bytes), count: g.Count()))
                .OrderByDescending(g => g.bytes)
                .Take(top)
                .Select(g => $"{MiB(g.bytes),8:0.0} MiB  x{g.count,-5} {g.owner}")
                .ToList();

            var largest = vram.OrderByDescending(e => e.Bytes).Take(15)
                .Select(e => $"{MiB(e.Bytes),8:0.0} MiB  {e.Owner}" + (e.Detail != null ? $"  ({e.Detail})" : "") + (DescribeShape(e) is { } s ? $"  {s}" : ""))
                .ToList();

            // The process's video memory as DXGI counts it (local segment)
            double? usage = device.VideoMemory() is { } vm ? MiB(vm.usage) : null;

            // Descriptor heaps are not resources but take video memory too: the shader-visible one is large
            long descriptorBytes = device.DescriptorHeapBytes();

            return new
            {
                vramTrackedMiB = MiB(vramBytes),
                vramTrackedCount = vram.Count,
                systemTrackedMiB = MiB(systemBytes),
                descriptorHeapsMiB = MiB(descriptorBytes),
                processVramMiB = usage,
                untrackedVramMiB = usage is { } total ? Math.Round(total - MiB(vramBytes) - MiB(descriptorBytes), 1) : (double?)null,
                vramByOwner = byOwner,
                largest,
                textureAssetsByFolder = TextureAssetsByFolder(vram),
            };
        }

        private static string? DescribeShape(Entry e)
        {
            if (!e.Resource.TryGetTarget(out var r)) return null;
            try
            {
                var d = r.Description;
                return d.Dimension == ResourceDimension.Buffer
                    ? $"buffer {d.Width:N0} B"
                    : $"{d.Dimension} {d.Width}x{d.Height}x{d.DepthOrArraySize} {d.Format} mips {d.MipLevels}";
            }
            catch { return null; }
        }

        // Texture assets the asset cache holds, by the first two folders of their source path: video memory and count
        private static List<string> TextureAssetsByFolder(List<Entry> vram)
        {
            var bytesOf = new Dictionary<IntPtr, long>();
            foreach (var e in vram)
                if (e.Resource.TryGetTarget(out var r)) bytesOf[r.NativePointer] = e.Bytes;

            var folders = new Dictionary<string, (long bytes, int count)>();
            var assets = Engine.Assets;
            if (assets == null) return new List<string>();
            foreach (var key in assets.GetCachedKeys())
            {
                if (assets.TryGet(key) is not Texture { Native: { } native } texture) continue;
                if (!bytesOf.TryGetValue(native.NativePointer, out long bytes)) continue;
                string folder = "(no source)";
                if (texture.Guid is { } guid && Assets.AssetDatabase.GetMeta(guid)?.SourcePath is { } source)
                {
                    var parts = source.Split('\\', '/');
                    folder = parts.Length > 2 ? $"{parts[0]}/{parts[1]}" : parts.Length > 1 ? parts[0] : "(root)";
                }
                var (b, c) = folders.GetValueOrDefault(folder);
                folders[folder] = (b + bytes, c + 1);
            }
            return folders.OrderByDescending(f => f.Value.bytes).Take(25)
                .Select(f => $"{Math.Round(f.Value.bytes / 1048576.0, 1),8:0.0} MiB  x{f.Value.count,-5} {f.Key}")
                .ToList();
        }
    }
}
