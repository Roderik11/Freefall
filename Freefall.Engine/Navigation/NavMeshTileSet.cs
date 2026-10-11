using System;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using DotRecast.Core;
using DotRecast.Core.Numerics;
using DotRecast.Detour;
using DotRecast.Detour.Io;

namespace Freefall.Navigation
{
    /// <summary>
    /// A baked navmesh, kept as independent tiles on a fixed world grid.
    ///
    /// Every grid cell stores the hash of the inputs it was built from (terrain heights, the meshes
    /// overlapping it, the bake settings) and, if anything walkable came out, its serialized Detour tile.
    /// A rebake compares hashes and only rebuilds the cells whose inputs changed (see NavMeshBuilder);
    /// the runtime DtNavMesh is assembled from the tiles.
    /// </summary>
    public sealed class NavMeshTileSet
    {
        public const int MaxVertsPerPoly = 6;

        private const int Magic = 0x4D4E4646;   // "FFNM"
        private const int Version = 1;
        private const int HeaderSize = 4 * 10;

        /// <summary>World position of the min corner of tile (0, 0).</summary>
        public readonly Vector3 Origin;

        /// <summary>Tile edge length in world units.</summary>
        public readonly float TileWorldSize;

        public readonly int TilesX;
        public readonly int TilesZ;

        /// <summary>Per grid cell: hash of the inputs the cell was built from. 0 = unknown.</summary>
        public readonly ulong[] Hashes;

        /// <summary>Per grid cell: serialized DtMeshData, or null where nothing is walkable.</summary>
        public readonly byte[]?[] Tiles;

        public int PolyCount;
        public int VertexCount;

        public NavMeshTileSet(Vector3 origin, float tileWorldSize, int tilesX, int tilesZ)
        {
            Origin = origin;
            TileWorldSize = tileWorldSize;
            TilesX = tilesX;
            TilesZ = tilesZ;
            Hashes = new ulong[tilesX * tilesZ];
            Tiles = new byte[]?[tilesX * tilesZ];
        }

        public int CellCount => Tiles.Length;

        /// <summary>Number of cells that hold a tile.</summary>
        public int TileCount
        {
            get
            {
                int count = 0;
                foreach (var tile in Tiles)
                    if (tile != null) count++;
                return count;
            }
        }

        public long ByteSize
        {
            get
            {
                long size = 0;
                foreach (var tile in Tiles)
                    if (tile != null) size += tile.Length;
                return size;
            }
        }

        /// <summary>True if both sets cut the world into the same cells, so tiles can be carried over.</summary>
        public bool SameGrid(Vector3 origin, float tileWorldSize, int tilesX, int tilesZ)
            => TilesX == tilesX && TilesZ == tilesZ
            && TileWorldSize == tileWorldSize
            && Origin.X == origin.X && Origin.Z == origin.Z;

        // ── Runtime navmesh ──

        /// <summary>
        /// Assemble the runtime navmesh from the tiles. Returns null if no cell holds a tile.
        /// Also refreshes PolyCount / VertexCount. Safe to call off the main thread.
        /// </summary>
        public DtNavMesh? CreateNavMesh()
        {
            int tileCount = TileCount;
            PolyCount = 0;
            VertexCount = 0;
            if (tileCount == 0) return null;

            var navParams = new DtNavMeshParams
            {
                orig = new RcVec3f(Origin.X, Origin.Y, Origin.Z),
                tileWidth = TileWorldSize,
                tileHeight = TileWorldSize,
                maxTiles = tileCount,
                maxPolys = 32768,
            };

            var navMesh = new DtNavMesh();
            navMesh.Init(navParams, MaxVertsPerPoly);

            var reader = new DtMeshDataReader();
            foreach (var bytes in Tiles)
            {
                if (bytes == null) continue;

                var data = reader.Read(new RcByteBuffer(bytes), MaxVertsPerPoly);
                navMesh.AddTile(data, 0, 0, out _);
                PolyCount += data.header.polyCount;
                VertexCount += data.header.vertCount;
            }

            return navMesh;
        }

        /// <summary>Serialize one Detour tile. The stream is scratch space and gets reset.</summary>
        internal static byte[] WriteTile(DtMeshData data, DtMeshDataWriter writer, MemoryStream scratch)
        {
            scratch.SetLength(0);
            using (var bw = new BinaryWriter(scratch, System.Text.Encoding.UTF8, leaveOpen: true))
                writer.Write(bw, data, RcByteOrder.LITTLE_ENDIAN, false);
            return scratch.ToArray();
        }

        // ── Storage ──

        public byte[] ToBytes()
        {
            int cells = CellCount;
            long total = HeaderSize + (long)cells * (sizeof(ulong) + sizeof(int)) + ByteSize;
            if (total > Array.MaxLength)
                throw new InvalidOperationException($"NavMesh is too large to store ({total / (1024 * 1024)} MB). Use a larger CellSize.");

            var buffer = new byte[total];
            using var ms = new MemoryStream(buffer);
            using var bw = new BinaryWriter(ms);

            bw.Write(Magic);
            bw.Write(Version);
            bw.Write(Origin.X);
            bw.Write(Origin.Y);
            bw.Write(Origin.Z);
            bw.Write(TileWorldSize);
            bw.Write(TilesX);
            bw.Write(TilesZ);
            bw.Write(PolyCount);
            bw.Write(VertexCount);

            bw.Write(MemoryMarshal.AsBytes(Hashes.AsSpan()));
            foreach (var tile in Tiles)
                bw.Write(tile?.Length ?? 0);
            foreach (var tile in Tiles)
                if (tile != null) bw.Write(tile);

            return buffer;
        }

        /// <summary>
        /// Read a stored navmesh. Also accepts the single-blob format written before navmeshes were
        /// stored per tile; those carry no hashes, so the next bake rebuilds every tile.
        /// </summary>
        public static NavMeshTileSet? FromBytes(byte[]? data)
        {
            if (data == null || data.Length < 8) return null;

            if (BitConverter.ToInt32(data, 0) != Magic)
                return FromMeshSet(data);

            using var ms = new MemoryStream(data);
            using var br = new BinaryReader(ms);

            br.ReadInt32();
            int version = br.ReadInt32();
            if (version != Version)
                throw new InvalidDataException($"Unsupported navmesh data version {version}");

            var origin = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            float tileWorldSize = br.ReadSingle();
            int tilesX = br.ReadInt32();
            int tilesZ = br.ReadInt32();

            var set = new NavMeshTileSet(origin, tileWorldSize, tilesX, tilesZ)
            {
                PolyCount = br.ReadInt32(),
                VertexCount = br.ReadInt32(),
            };

            br.BaseStream.ReadExactly(MemoryMarshal.AsBytes(set.Hashes.AsSpan()));

            var sizes = new int[set.CellCount];
            br.BaseStream.ReadExactly(MemoryMarshal.AsBytes(sizes.AsSpan()));

            for (int i = 0; i < sizes.Length; i++)
                if (sizes[i] > 0) set.Tiles[i] = br.ReadBytes(sizes[i]);

            return set;
        }

        private static NavMeshTileSet? FromMeshSet(byte[] data)
        {
            DtNavMesh navMesh;
            using (var ms = new MemoryStream(data))
            using (var br = new BinaryReader(ms))
                navMesh = new DtMeshSetReader().Read(br, MaxVertsPerPoly);

            int tilesX = 0, tilesZ = 0;
            for (int i = 0; i < navMesh.GetMaxTiles(); i++)
            {
                var header = navMesh.GetTile(i)?.data?.header;
                if (header == null) continue;
                tilesX = Math.Max(tilesX, header.x + 1);
                tilesZ = Math.Max(tilesZ, header.y + 1);
            }
            if (tilesX == 0 || tilesZ == 0) return null;

            var navParams = navMesh.GetParams();
            var set = new NavMeshTileSet(
                new Vector3(navParams.orig.X, navParams.orig.Y, navParams.orig.Z),
                navParams.tileWidth, tilesX, tilesZ);

            var writer = new DtMeshDataWriter();
            using var scratch = new MemoryStream();
            for (int i = 0; i < navMesh.GetMaxTiles(); i++)
            {
                var tileData = navMesh.GetTile(i)?.data;
                if (tileData?.header == null || tileData.header.x < 0 || tileData.header.y < 0) continue;

                set.Tiles[tileData.header.y * tilesX + tileData.header.x] = WriteTile(tileData, writer, scratch);
                set.PolyCount += tileData.header.polyCount;
                set.VertexCount += tileData.header.vertCount;
            }

            return set;
        }
    }
}
