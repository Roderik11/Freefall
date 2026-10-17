using System.Runtime.InteropServices;

namespace Freefall.Graphics
{
    /// <summary>
    /// The parts of meshoptimizer (github.com/zeux/meshoptimizer, MIT) the engine uses, by P/Invoke.
    /// meshoptimizer.dll is built from ThirdParty/meshoptimizer (build.cmd) and copied to the output
    /// by the engine project. Signatures follow meshoptimizer.h (v1.3); size_t is nuint.
    /// </summary>
    internal static unsafe class MeshOptimizer
    {
        private const string Library = "meshoptimizer";

        // meshopt_SimplifyX options
        public const uint SimplifyLockBorder = 1 << 0;
        public const uint SimplifySparse = 1 << 1;
        public const uint SimplifyErrorAbsolute = 1 << 2;
        public const uint SimplifyPrune = 1 << 3;

        /// <summary>
        /// Simplify an indexed triangle list towards target_index_count while keeping the error below
        /// target_error, taking per-vertex attributes into account. Attribute seams (vertices that share a
        /// position but not their data) are kept. Writes up to index_count indices into destination
        /// (all referencing the original vertices) and returns how many.
        /// </summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern nuint meshopt_simplifyWithAttributes(
            uint* destination, uint* indices, nuint index_count,
            float* vertex_positions, nuint vertex_count, nuint vertex_positions_stride,
            float* vertex_attributes, nuint vertex_attributes_stride,
            float* attribute_weights, nuint attribute_count,
            byte* vertex_lock, nuint target_index_count, float target_error, uint options, float* result_error);

        /// <summary>The factor that turns a relative simplification error into an absolute one (the mesh extent).</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern float meshopt_simplifyScale(float* vertex_positions, nuint vertex_count, nuint vertex_positions_stride);

        /// <summary>Reorder triangles for the post-transform vertex cache. destination may be indices.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void meshopt_optimizeVertexCache(uint* destination, uint* indices, nuint index_count, nuint vertex_count);
    }
}
