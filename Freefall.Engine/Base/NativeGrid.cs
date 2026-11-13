using System;
using System.Runtime.InteropServices;

namespace Freefall.Base
{
    /// <summary>
    /// A zeroed array in native memory with an array's indexer, for the scratch of a large computation on a worker (the
    /// sea mask's flood, the shore warp's marching): it never touches the managed heap, so it is no large object for the GC
    /// to leave a hole of (ram.md), and <see cref="Dispose"/> gives the pages back at once. Bounds are checked as an
    /// array's are. Not thread-safe to dispose while in use; the finalizer frees one that was never disposed.
    /// </summary>
    public sealed unsafe class NativeGrid<T> : IDisposable where T : unmanaged
    {
        private T* _data;
        private int _length;   // 0 once freed: an access after Dispose fails the bounds check instead of reading freed memory

        public int Length => _length;

        public NativeGrid(int length)
        {
            if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
            _data = (T*)NativeMemory.AllocZeroed((nuint)Math.Max(length, 1), (nuint)sizeof(T));
            _length = length;
        }

        /// <summary>A copy of <paramref name="source"/> (what <c>(T[])array.Clone()</c> was).</summary>
        public NativeGrid(NativeGrid<T> source) : this(source.Length) => source.AsSpan().CopyTo(AsSpan());

        public ref T this[int index]
        {
            get
            {
                if ((uint)index >= (uint)_length) throw new IndexOutOfRangeException();
                return ref _data[index];
            }
        }

        public Span<T> AsSpan() => _data == null ? throw new ObjectDisposedException(nameof(NativeGrid<T>)) : new Span<T>(_data, _length);

        public void Fill(T value) => AsSpan().Fill(value);

        public void CopyTo(NativeGrid<T> destination) => AsSpan().CopyTo(destination.AsSpan());

        public void Dispose()
        {
            Free();
            GC.SuppressFinalize(this);
        }

        ~NativeGrid() => Free();

        private void Free()
        {
            if (_data == null) return;
            _length = 0;
            NativeMemory.Free(_data);
            _data = null;
        }
    }
}
