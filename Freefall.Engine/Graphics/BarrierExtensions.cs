using System;
using Vortice.Direct3D12;

namespace Freefall
{
    /// <summary>
    /// Several resource barriers in one call without allocating: Vortice's ResourceBarrier(params ResourceBarrier[])
    /// (or an explicit new[] { ... }) makes a new array each time, which on the per-batch culling and shadow paths
    /// added up to kilobytes of garbage per frame. These fill a reusable per-thread array and use the
    /// (count, array) overload.
    /// </summary>
    public static class BarrierExtensions
    {
        [ThreadStatic] private static ResourceBarrier[]? t_scratch;

        private static ResourceBarrier[] Scratch => t_scratch ??= new ResourceBarrier[6];

        public static void Barriers(this ID3D12GraphicsCommandList list, ResourceBarrier a, ResourceBarrier b)
        {
            var s = Scratch;
            s[0] = a; s[1] = b;
            Submit(list, s, 2);
        }

        public static void Barriers(this ID3D12GraphicsCommandList list, ResourceBarrier a, ResourceBarrier b, ResourceBarrier c)
        {
            var s = Scratch;
            s[0] = a; s[1] = b; s[2] = c;
            Submit(list, s, 3);
        }

        public static void Barriers(this ID3D12GraphicsCommandList list, ResourceBarrier a, ResourceBarrier b, ResourceBarrier c,
            ResourceBarrier d)
        {
            var s = Scratch;
            s[0] = a; s[1] = b; s[2] = c; s[3] = d;
            Submit(list, s, 4);
        }

        public static void Barriers(this ID3D12GraphicsCommandList list, ResourceBarrier a, ResourceBarrier b, ResourceBarrier c,
            ResourceBarrier d, ResourceBarrier e)
        {
            var s = Scratch;
            s[0] = a; s[1] = b; s[2] = c; s[3] = d; s[4] = e;
            Submit(list, s, 5);
        }

        public static void Barriers(this ID3D12GraphicsCommandList list, ResourceBarrier a, ResourceBarrier b, ResourceBarrier c,
            ResourceBarrier d, ResourceBarrier e, ResourceBarrier f)
        {
            var s = Scratch;
            s[0] = a; s[1] = b; s[2] = c; s[3] = d; s[4] = e; s[5] = f;
            Submit(list, s, 6);
        }

        private static void Submit(ID3D12GraphicsCommandList list, ResourceBarrier[] s, int count)
        {
            list.ResourceBarrier((uint)count, s);
            Array.Clear(s, 0, count); // the barriers reference resources: don't keep them alive here
        }
    }
}
