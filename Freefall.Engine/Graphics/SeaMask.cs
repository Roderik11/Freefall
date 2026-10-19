using System;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Where the open sea is. The ocean is one endless plane at its entity's height, so without this it showed
    /// wherever the ground dips below that level: in a lake bed dug deeper than the sea, in a hall under a hill.
    /// The mask floods the terrain's heights from its border across every sample below sea level. What the flood
    /// reaches is sea; the rest (land, and low ground enclosed by higher ground) is not, and the ocean draws nothing
    /// there (ocean.fx OutsideSea). Past the terrain it is all sea.
    ///
    /// Decided by the ground seen from above: a cave under a hill is never sea, a tunnel under the seabed still is
    /// (the plane above it is hidden by its ceiling unless the tunnel reaches up through sea level).
    ///
    /// OceanRenderer builds it on a worker from the CPU height field whenever the terrain's heights are read back
    /// (EngineMsg.TerrainHeightsChanged) or the sea level moves, and uploads it here: R8, laid out like the heightmap.
    /// </summary>
    public sealed class SeaMask : IDisposable
    {
        /// <summary>
        /// How far the sea reaches up the land past the still waterline (m), at most: over land up to
        /// <see cref="BeachReachHeight"/> above the sea. Waves and the swash run up the beach above sea level, and a mask
        /// that ended short of them cut them off along a hard line (the surf's swash runs 15–20 m up a flat beach: the
        /// reach was a flat 8 m until 2026-10-19). Never crosses into enclosed low ground.
        /// </summary>
        public const float BeachReach = 60f;

        /// <summary>How high above the still sea the beach reach goes (m): no swash wets the land higher than this.</summary>
        public const float BeachReachHeight = 2.5f;

        /// <summary>Most cells along a side: a 4097² heightmap is flooded at every second sample (2 m on a 4 km terrain).</summary>
        public const int MaxSize = 2049;

        /// <summary>A flooded mask, ready to upload: Size² bytes, row-major (row = terrain Z), 255 = sea.</summary>
        public sealed class Result
        {
            public byte[] Cells = Array.Empty<byte>();
            public int Size;
            public float SeaLevel;
            public float SeaShare;          // of the terrain's cells, incl. the beaches
            public double Milliseconds;     // the flood
        }

        private ID3D12Resource? _texture;
        private uint _srv;
        private int _size;

        /// <summary>Bindless SRV of the mask; 0 until the first upload (the ocean then draws everywhere).</summary>
        public uint Srv => _srv;

        /// <summary>
        /// Flood the height field (values 0..1 of the terrain's MaxHeight, [x, z]) from its border across every sample
        /// below <paramref name="seaLevel01"/>, then let the sea reach up the land (<see cref="BeachReach"/>,
        /// <see cref="BeachReachHeight"/>). <paramref name="sampleSpacing"/> is the distance between two height samples
        /// (m). Any thread.
        /// </summary>
        public static Result Build(float[,] heights, float seaLevel01, float sampleSpacing, float seaLevel, float maxHeight)
        {
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            int res = Math.Min(heights.GetLength(0), heights.GetLength(1));
            int step = 1;
            while ((res - 1) / step + 1 > MaxSize) step *= 2;
            int size = (res - 1) / step + 1;
            int count = size * size;

            var below = new bool[count];
            for (int z = 0; z < size; z++)
            for (int x = 0; x < size; x++)
                below[z * size + x] = heights[x * step, z * step] < seaLevel01;

            var cells = new byte[count];
            var queue = new int[count];
            int head = 0, tail = 0;

            // The open sea comes in from beyond the terrain: every border cell below sea level is sea
            void Seed(int i)
            {
                if (!below[i] || cells[i] != 0) return;
                cells[i] = 255;
                queue[tail++] = i;
            }
            for (int k = 0; k < size; k++)
            {
                Seed(k);
                Seed((size - 1) * size + k);
                Seed(k * size);
                Seed(k * size + size - 1);
            }

            // Flood across low ground (4-connected: two cells touching at a corner do not let the sea through)
            while (head < tail)
            {
                int i = queue[head++];
                int x = i % size, z = i / size;
                if (x > 0) Flood(i - 1);
                if (x < size - 1) Flood(i + 1);
                if (z > 0) Flood(i - size);
                if (z < size - 1) Flood(i + size);
            }
            void Flood(int n)
            {
                if (cells[n] != 0 || !below[n]) return;
                cells[n] = 255;
                queue[tail++] = n;
            }

            // Up the beaches: land within BeachReach of the sea and lower than BeachReachHeight over it, never into low
            // ground the sea did not reach
            int reach = Math.Clamp((int)MathF.Ceiling(BeachReach / Math.Max(sampleSpacing * step, 0.01f)), 1, 255);
            float reachTop01 = seaLevel01 + BeachReachHeight / Math.Max(maxHeight, 0.001f);
            var distance = new byte[count];
            head = 0;
            tail = 0;
            for (int i = 0; i < count; i++)
                if (cells[i] != 0) queue[tail++] = i;
            while (head < tail)
            {
                int i = queue[head++];
                int d = distance[i];
                if (d >= reach) continue;
                int x = i % size, z = i / size;
                for (int dz = -1; dz <= 1; dz++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = x + dx, nz = z + dz;
                    if ((dx | dz) == 0 || nx < 0 || nz < 0 || nx >= size || nz >= size) continue;
                    int n = nz * size + nx;
                    if (cells[n] != 0 || below[n] || heights[nx * step, nz * step] > reachTop01) continue;
                    cells[n] = 255;
                    distance[n] = (byte)(d + 1);
                    queue[tail++] = n;
                }
            }

            return new Result
            {
                Cells = cells, Size = size, SeaLevel = seaLevel,
                SeaShare = (float)tail / count,
                Milliseconds = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds,
            };
        }

        /// <summary>Copy a built mask into the texture (made anew when its size changed). Main thread; the copy runs
        /// at the start of the next frame's opaque pass, before the forward pass draws the sea.</summary>
        public unsafe void Upload(Result mask)
        {
            var device = Engine.Device;
            int size = mask.Size;

            if (_texture == null || _size != size)
            {
                device.DeferDispose(_texture);
                if (_srv != 0) device.DeferReleaseBindlessIndex(_srv);

                _texture = device.CreateTexture2D(Format.R8_UNorm, size, size, 1, 1, ResourceFlags.None, ResourceStates.Common);
                _srv = device.AllocateBindlessIndex();
                device.NativeDevice.CreateShaderResourceView(_texture,
                    new ShaderResourceViewDescription
                    {
                        Format = Format.R8_UNorm,
                        ViewDimension = ShaderResourceViewDimension.Texture2D,
                        Shader4ComponentMapping = ShaderComponentMapping.Default,
                        Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
                    }, device.GetCpuHandle(_srv));
                _size = size;
            }

            int rowPitch = (size + 255) & ~255;
            var upload = device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Upload), HeapFlags.None,
                ResourceDescription.Buffer((ulong)(rowPitch * size)), ResourceStates.GenericRead, null);

            void* data;
            upload.Map(0, null, &data);
            fixed (byte* src = mask.Cells)
                for (int z = 0; z < size; z++)
                    Buffer.MemoryCopy(src + (long)z * size, (byte*)data + (long)z * rowPitch, size, size);
            upload.Unmap(0);

            var footprint = new PlacedSubresourceFootPrint
            {
                Offset = 0,
                Footprint = new SubresourceFootPrint(Format.R8_UNorm, (uint)size, (uint)size, 1, (uint)rowPitch)
            };
            var tex = _texture;
            CommandBuffer.Enqueue(RenderPass.Opaque, list =>
            {
                list.ResourceBarrierTransition(tex, ResourceStates.Common, ResourceStates.CopyDest);
                list.CopyTextureRegion(new TextureCopyLocation(tex, 0), 0, 0, 0, new TextureCopyLocation(upload, footprint));
                list.ResourceBarrierTransition(tex, ResourceStates.CopyDest, ResourceStates.Common);
            });
            device.DeferDispose(upload);   // until the frames that copy from it are done

            Debug.Log($"[SeaMask] {size}² cells at sea level {mask.SeaLevel:0.##} m: {mask.SeaShare:P1} sea (beaches included), flooded in {mask.Milliseconds:0} ms");
        }

        public void Dispose()
        {
            var device = Engine.Device;
            device.DeferDispose(_texture);
            if (_srv != 0) device.DeferReleaseBindlessIndex(_srv);
            _texture = null;
            _srv = 0;
        }
    }
}
