using System;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// PIX event marker helper using WinPixEventRuntime (shipped via Vortice.Pix.Native).
    /// Provides GPU command list markers that appear as collapsible named regions in PIX captures.
    /// On the primary view's command list each region is also timed on the GPU (<see cref="GpuProfiler"/>).
    /// </summary>
    public static class PixMarker
    {

        // WinPixEventRuntime exports
        [DllImport("WinPixEventRuntime", CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe void PIXBeginEventOnCommandList(
            IntPtr commandList,
            ulong color,
            [MarshalAs(UnmanagedType.LPStr)] string formatString);

        [DllImport("WinPixEventRuntime", CallingConvention = CallingConvention.Cdecl)]
        private static extern void PIXEndEventOnCommandList(IntPtr commandList);

        /// <summary>
        /// Begin a named PIX event region on the command list.
        /// </summary>
        public static void Begin(ID3D12GraphicsCommandList list, string name)
        {
            FrameTrace.Step(name);
            GpuProfiler.Begin(list, name);
            try
            {
                PIXBeginEventOnCommandList(list.NativePointer, 0xFF00FF00, name);
            }
            catch (DllNotFoundException) { } // Silently skip if runtime not present
        }

        /// <summary>
        /// Begin a named PIX event region with a custom color (0xAARRGGBB).
        /// </summary>
        public static void Begin(ID3D12GraphicsCommandList list, uint color, string name)
        {
            FrameTrace.Step(name);
            GpuProfiler.Begin(list, name);
            try
            {
                PIXBeginEventOnCommandList(list.NativePointer, color, name);
            }
            catch (DllNotFoundException) { }
        }

        /// <summary>
        /// End the current PIX event region.
        /// </summary>
        public static void End(ID3D12GraphicsCommandList list)
        {
            GpuProfiler.End(list);
            try
            {
                PIXEndEventOnCommandList(list.NativePointer);
            }
            catch (DllNotFoundException) { }
        }
    }
}
