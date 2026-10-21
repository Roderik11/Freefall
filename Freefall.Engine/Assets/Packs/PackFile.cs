using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace Freefall.Assets.Packs
{
    /// <summary>
    /// One artifact in a pack: its GUID (the Cache file's name), its data type (the packer's type name, e.g.
    /// "DdsTextureData") and where its bytes are.
    /// </summary>
    public readonly record struct PackEntry(string Guid, string DataType, long Offset, long Length);

    /// <summary>
    /// The pack file format (*.ffpack): Cache artifacts byte for byte, so the loaders read them as they read the Cache.
    /// <code>
    /// header (64 bytes)  "FFPK", version, entry count, alignment, TOC offset, TOC length, reserved
    /// data               each artifact's bytes, starting on a multiple of the alignment (4 KB: unbuffered reads later)
    /// TOC (at the end)   per entry: GUID (16 bytes), offset, length, stored length, compression (0 = none), data type
    /// </code>
    /// Stored length and compression are there for compression per entry later; v1 stores everything as it is.
    /// </summary>
    public static class PackFormat
    {
        public const string Extension = ".ffpack";
        public const int HeaderSize = 64;
        public const int Version = 1;
        public const int DefaultAlignment = 4096;
        public static ReadOnlySpan<byte> Magic => "FFPK"u8;
    }

    /// <summary>Writes one pack: Add the artifacts in the order they should lie, Dispose writes the TOC.</summary>
    public sealed class PackWriter : IDisposable
    {
        private readonly FileStream _file;
        private readonly int _alignment;
        private readonly List<PackEntry> _entries = new();
        private bool _done;

        public string Path { get; }
        public IReadOnlyList<PackEntry> Entries => _entries;

        /// <summary>Bytes written so far (header and data): what the pack's size will be without its TOC.</summary>
        public long Length => _file.Length;

        /// <summary>What the pack's size would be if it were finished now: the data, the TOC and the padding before a next entry.</summary>
        public long FinishedLength => Length + _tocBytes;

        /// <summary>Size of one TOC entry: GUID, offset, lengths, compression, data type (length-prefixed UTF-8).</summary>
        public static long TocEntrySize(string dataType)
        {
            int n = Encoding.UTF8.GetByteCount(dataType ?? "");
            return 16 + 8 + 8 + 8 + 1 + (n < 128 ? 1 : 2) + n;
        }

        private long _tocBytes;

        public PackWriter(string path, int alignment = PackFormat.DefaultAlignment)
        {
            Path = path;
            _alignment = alignment;
            _file = new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.None, 1 << 20);
            _file.Write(new byte[PackFormat.HeaderSize]);
        }

        /// <summary>Copy an artifact's bytes into the pack.</summary>
        public void Add(string guid, string dataType, Stream source)
        {
            if (_done) throw new InvalidOperationException("The pack is finished");
            Pad();
            long offset = _file.Position;
            source.CopyTo(_file, 1 << 20);
            _entries.Add(new PackEntry(guid.ToLowerInvariant(), dataType ?? "", offset, _file.Position - offset));
            _tocBytes += TocEntrySize(dataType);
        }

        /// <summary>Whether an entry of this size would keep the finished pack within <paramref name="limit"/> bytes.</summary>
        public bool Fits(long size, string dataType, long limit) =>
            FinishedLength + (_alignment - Length % _alignment) % _alignment + size + TocEntrySize(dataType) <= limit;

        private void Pad()
        {
            long rem = _file.Position % _alignment;
            if (rem != 0) _file.Write(new byte[_alignment - rem]);
        }

        public void Dispose()
        {
            if (_done) return;
            _done = true;
            try
            {
                long tocOffset = _file.Position;
                using (var w = new BinaryWriter(_file, Encoding.UTF8, leaveOpen: true))
                {
                    foreach (var e in _entries)
                    {
                        w.Write(System.Guid.ParseExact(e.Guid, "N").ToByteArray());
                        w.Write(e.Offset);
                        w.Write(e.Length);
                        w.Write(e.Length);   // stored length
                        w.Write((byte)0);    // compression: none
                        w.Write(e.DataType);
                    }
                    long tocLength = _file.Position - tocOffset;

                    _file.Position = 0;
                    w.Write(PackFormat.Magic);
                    w.Write(PackFormat.Version);
                    w.Write(_entries.Count);
                    w.Write(_alignment);
                    w.Write(tocOffset);
                    w.Write(tocLength);
                }
                _file.Flush(flushToDisk: true);
            }
            finally
            {
                _file.Dispose();
            }
        }
    }

    /// <summary>
    /// A pack opened for reading: its TOC in memory and one file handle, read at positions (RandomAccess), so loader
    /// threads reading different artifacts do not share a file position.
    /// </summary>
    public sealed class PackReader : IDisposable
    {
        private readonly SafeFileHandle _handle;
        private readonly Dictionary<string, PackEntry> _entries;

        public string Path { get; }
        public IReadOnlyCollection<PackEntry> Entries => _entries.Values;

        private PackReader(string path, SafeFileHandle handle, Dictionary<string, PackEntry> entries)
        {
            Path = path;
            _handle = handle;
            _entries = entries;
        }

        public static PackReader Open(string path)
        {
            var handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.RandomAccess);
            try
            {
                var header = new byte[PackFormat.HeaderSize];
                RandomAccess.Read(handle, header, 0);
                if (!header.AsSpan(0, 4).SequenceEqual(PackFormat.Magic))
                    throw new InvalidDataException($"{path} is not a pack file");
                int version = BitConverter.ToInt32(header, 4);
                if (version != PackFormat.Version)
                    throw new InvalidDataException($"{path}: pack version {version}, this engine reads {PackFormat.Version}");
                int count = BitConverter.ToInt32(header, 8);
                long tocOffset = BitConverter.ToInt64(header, 16);
                long tocLength = BitConverter.ToInt64(header, 24);

                var toc = new byte[tocLength];
                RandomAccess.Read(handle, toc, tocOffset);
                var entries = new Dictionary<string, PackEntry>(count, StringComparer.OrdinalIgnoreCase);
                using var r = new BinaryReader(new MemoryStream(toc), Encoding.UTF8);
                for (int i = 0; i < count; i++)
                {
                    var guid = new System.Guid(r.ReadBytes(16)).ToString("N");
                    long offset = r.ReadInt64(), length = r.ReadInt64();
                    r.ReadInt64();                      // stored length (= length: no compression in v1)
                    byte compression = r.ReadByte();
                    var dataType = r.ReadString();
                    if (compression != 0)
                        throw new InvalidDataException($"{path}: compressed entries are not read by this engine");
                    entries[guid] = new PackEntry(guid, dataType, offset, length);
                }
                return new PackReader(path, handle, entries);
            }
            catch
            {
                handle.Dispose();
                throw;
            }
        }

        public bool TryGet(string guid, out PackEntry entry) => _entries.TryGetValue(guid, out entry);

        /// <summary>
        /// A read-only, seekable stream over one entry's bytes, buffered: the packers read field by field (BinaryReader),
        /// and every unbuffered read is a system call (a scene loaded twelve times slower than from the Cache's
        /// FileStreams). The buffer is the entry's size up to 1 MB, so a small artifact is one read.
        /// </summary>
        public Stream OpenEntry(in PackEntry entry) =>
            new BufferedStream(new PackEntryStream(_handle, entry.Offset, entry.Length), (int)Math.Clamp(entry.Length, 1, 1 << 20));

        public void Dispose() => _handle.Dispose();
    }

    /// <summary>The packs of a game, mounted together: a GUID in a later pack wins over the same GUID in an earlier one (patch packs).</summary>
    public sealed class PackSet : IDisposable
    {
        private readonly List<PackReader> _packs;
        private readonly Dictionary<string, (PackReader Pack, PackEntry Entry)> _index = new(StringComparer.OrdinalIgnoreCase);

        public IReadOnlyList<PackReader> Packs => _packs;
        public int Count => _index.Count;
        public IEnumerable<PackEntry> Entries => _index.Values.Select(v => v.Entry);

        public PackSet(IEnumerable<PackReader> packs)
        {
            _packs = packs.ToList();
            foreach (var pack in _packs)
                foreach (var e in pack.Entries)
                    _index[e.Guid] = (pack, e);
        }

        /// <summary>Open these pack files, in this order (later ones win).</summary>
        public static PackSet Open(IEnumerable<string> paths) => new(paths.Select(PackReader.Open).ToList());

        /// <summary>Every *.ffpack in a folder, in name order.</summary>
        public static PackSet OpenFolder(string folder) =>
            Open(Directory.GetFiles(folder, "*" + PackFormat.Extension).OrderBy(p => p, StringComparer.OrdinalIgnoreCase));

        public bool Contains(string guid) => _index.ContainsKey(guid);

        public bool TryGet(string guid, out PackEntry entry)
        {
            bool found = _index.TryGetValue(guid, out var hit);
            entry = hit.Entry;
            return found;
        }

        /// <summary>The artifact's bytes, or null when no pack holds it.</summary>
        public Stream Open(string guid) => _index.TryGetValue(guid, out var hit) ? hit.Pack.OpenEntry(hit.Entry) : null;

        public void Dispose()
        {
            foreach (var p in _packs) p.Dispose();
        }
    }

    /// <summary>A window onto one entry of a pack file, read at positions so that several can be read at once.</summary>
    internal sealed class PackEntryStream : Stream
    {
        private readonly SafeFileHandle _handle;
        private readonly long _start, _length;
        private long _position;

        public PackEntryStream(SafeFileHandle handle, long start, long length)
        {
            _handle = handle;
            _start = start;
            _length = length;
        }

        public override bool CanRead => true;
        public override bool CanSeek => true;
        public override bool CanWrite => false;
        public override long Length => _length;

        public override long Position
        {
            get => _position;
            set => _position = Math.Clamp(value, 0, _length);
        }

        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

        public override int Read(Span<byte> buffer)
        {
            long left = _length - _position;
            if (left <= 0) return 0;
            if (buffer.Length > left) buffer = buffer[..(int)left];
            int read = RandomAccess.Read(_handle, buffer, _start + _position);
            _position += read;
            return read;
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            Position = origin switch
            {
                SeekOrigin.Begin => offset,
                SeekOrigin.Current => _position + offset,
                _ => _length + offset,
            };
            return _position;
        }

        public override void Flush() { }
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
