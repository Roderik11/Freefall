using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>Which half of a mesh an icon part keeps, by its X in the mesh's own space: one glove of a pair.</summary>
    public enum IconSide
    {
        Both,
        /// <summary>X at or over <see cref="IconPart.SideCut"/>.</summary>
        Positive,
        /// <summary>X under <see cref="IconPart.SideCut"/>.</summary>
        Negative,
    }

    /// <summary>One mesh of an icon, placed among the others.</summary>
    public struct IconPart
    {
        public Mesh Mesh;

        /// <summary>By material slot (MeshPart.MaterialSlot). A slot without one draws plain grey.</summary>
        public IReadOnlyList<Material?>? Materials;

        /// <summary>The mesh in the item's space (after the mesh's own RootRotation).</summary>
        public Matrix4x4 Transform;

        public IconSide Side;
        public float SideCut;

        /// <summary>Its albedo's alpha below 0.25 is cut away (hair cards, torn cloth, a slit visor), as the skinned G-buffer does.</summary>
        public bool AlphaTest;

        public IconPart(Mesh mesh, IReadOnlyList<Material?>? materials, Matrix4x4 transform)
        {
            Mesh = mesh;
            Materials = materials;
            Transform = transform;
            Side = IconSide.Both;
            SideCut = 0f;
            AlphaTest = false;
        }
    }

    /// <summary>How an icon looks at its item: the turn, the border and the exposure.</summary>
    public sealed class IconView
    {
        /// <summary>
        /// The item's space turned before the camera looks at it, down +Z with Y up and orthographic (left-handed, as
        /// the engine's views). Applied after <see cref="IconRenderer.LongAxisFrame"/> when the caller uses one.
        /// </summary>
        public Matrix4x4 Rotation = Matrix4x4.Identity;

        /// <summary>The empty border on each side, as a share of the picture's width and height.</summary>
        public float Margin = 0.04f;

        /// <summary>Multiplies the light before the tonemapper.</summary>
        public float Exposure = 1f;

        /// <summary>Samples per pixel along each axis: drawn this much larger and averaged down.</summary>
        public int Supersample = 4;

        /// <summary>
        /// How dark a part's inside goes (0..1): surfaces facing in towards the part's middle, a helmet's lining or the
        /// way into a boot, which the all-round studio light would otherwise show as lit as the outside.
        /// </summary>
        public float Interior = 1f;

        /// <summary>
        /// A turn by degrees: about Y (the item turned to the left or right), then X (its top towards the camera or
        /// away), then Z (in the picture's plane, as a long weapon lies across a tall cell).
        /// </summary>
        public static Matrix4x4 Turn(float yawDegrees, float pitchDegrees, float rollDegrees) =>
            Matrix4x4.CreateRotationY(yawDegrees * MathF.PI / 180f)
          * Matrix4x4.CreateRotationX(pitchDegrees * MathF.PI / 180f)
          * Matrix4x4.CreateRotationZ(rollDegrees * MathF.PI / 180f);
    }

    /// <summary>
    /// Draws an item's own meshes and materials into a picture with a transparent background, for an inventory icon:
    /// a fixed studio of lights (icon.fx), orthographic, framed so the item fills the picture's aspect. Main thread,
    /// outside the frame's rendering: each Render records, submits and waits on its own command list.
    /// </summary>
    public sealed class IconRenderer : IDisposable
    {
        private static IconRenderer? _shared;

        /// <summary>One renderer kept for the editor's tools.</summary>
        public static IconRenderer Shared => _shared ??= new IconRenderer();

        private readonly Material _material;
        private readonly ID3D12CommandAllocator _allocator;
        private readonly ID3D12GraphicsCommandList _cmd;
        private RenderTexture2D? _color;
        private DepthTexture2D? _depth;
        private int _targetWidth, _targetHeight;

        public IconRenderer()
        {
            _material = new Material(new Effect("icon"));
            var device = Engine.Device.NativeDevice;
            _allocator = device.CreateCommandAllocator(CommandListType.Direct);
            _cmd = device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, _allocator);
            _cmd.Close();
        }

        /// <summary>
        /// A turn that stands a long item up along its longest axis and turns its flattest side towards the camera (a
        /// blade's face, a shield's front, the plane a bow bends in). Axes are the item space's own: a generated weapon
        /// lies along them. Which end goes on top and which side faces the camera go by the item's origin, where a
        /// held item is gripped: by default the end farther from it (a blade's tip) and the side farther from it;
        /// <paramref name="nearEndUp"/> puts the near end on top (a staff gripped below its head), and
        /// <paramref name="nearSideFront"/> turns the near side to the camera (a curved shield, whose rim reaches
        /// further back than its grip).
        /// </summary>
        public static Matrix4x4 LongAxisFrame(IReadOnlyList<IconPart> parts, bool nearEndUp = false, bool nearSideFront = false)
        {
            if (!Bounds(parts, Matrix4x4.Identity, out var min, out var max))
                return Matrix4x4.Identity;
            var size = max - min;
            Span<int> order = stackalloc int[] { 0, 1, 2 };
            // Longest first, flattest last
            for (int i = 0; i < 2; i++)
                for (int j = i + 1; j < 3; j++)
                    if (Get(size, order[j]) > Get(size, order[i])) (order[i], order[j]) = (order[j], order[i]);

            int longAxis = order[0], flatAxis = order[2];
            // The far end: the one farther from the origin
            float upSign = Math.Abs(Get(max, longAxis)) >= Math.Abs(Get(min, longAxis)) ? 1f : -1f;
            float frontSign = Math.Abs(Get(max, flatAxis)) >= Math.Abs(Get(min, flatAxis)) ? 1f : -1f;
            if (nearEndUp) upSign = -upSign;
            if (nearSideFront) frontSign = -frontSign;

            var up = Axis(longAxis) * upSign;
            var front = Axis(flatAxis) * frontSign;    // towards the camera, which is at -Z
            var z = -front;
            var x = Vector3.Cross(up, z);

            // Columns: the item-space vectors that become view X, Y, Z
            return new Matrix4x4(
                x.X, up.X, z.X, 0,
                x.Y, up.Y, z.Y, 0,
                x.Z, up.Z, z.Z, 0,
                0, 0, 0, 1);

            static float Get(Vector3 v, int axis) => axis == 0 ? v.X : axis == 1 ? v.Y : v.Z;
            static Vector3 Axis(int axis) => axis == 0 ? Vector3.UnitX : axis == 1 ? Vector3.UnitY : Vector3.UnitZ;
        }

        /// <summary>
        /// A turn that stands the parts up along the direction they spread out most (their vertices' principal axis),
        /// whatever angle that is at in the item's space: a vambrace on an arm held out and bent in the rest pose. The
        /// hints are directions in the item's space: the end towards <paramref name="upHint"/> goes on top, and of the
        /// sides across that axis, the one nearest <paramref name="frontHint"/> faces the camera.
        /// </summary>
        public static Matrix4x4 PrincipalFrame(IReadOnlyList<IconPart> parts, Vector3 upHint, Vector3 frontHint)
        {
            // The vertices' covariance (each part's kept half)
            double n = 0, sx = 0, sy = 0, sz = 0;
            var sums = new double[6]; // xx, yy, zz, xy, xz, yz
            foreach (var part in parts)
            {
                var positions = part.Mesh?.Positions;
                if (positions == null) continue;
                var m = part.Mesh!.RootRotation * part.Transform;
                foreach (var p in positions)
                {
                    if (part.Side == IconSide.Positive && p.X < part.SideCut) continue;
                    if (part.Side == IconSide.Negative && p.X >= part.SideCut) continue;
                    var q = Vector3.Transform(p, m);
                    n++; sx += q.X; sy += q.Y; sz += q.Z;
                    sums[0] += q.X * q.X; sums[1] += q.Y * q.Y; sums[2] += q.Z * q.Z;
                    sums[3] += q.X * q.Y; sums[4] += q.X * q.Z; sums[5] += q.Y * q.Z;
                }
            }
            if (n < 3) return Matrix4x4.Identity;
            double mx = sx / n, my = sy / n, mz = sz / n;
            var a = new double[3, 3];
            a[0, 0] = sums[0] / n - mx * mx; a[1, 1] = sums[1] / n - my * my; a[2, 2] = sums[2] / n - mz * mz;
            a[0, 1] = a[1, 0] = sums[3] / n - mx * my;
            a[0, 2] = a[2, 0] = sums[4] / n - mx * mz;
            a[1, 2] = a[2, 1] = sums[5] / n - my * mz;
            var (values, vectors) = Eigen3(a);
            int longest = values[0] >= values[1] ? (values[0] >= values[2] ? 0 : 2) : (values[1] >= values[2] ? 1 : 2);
            var up = new Vector3((float)vectors[0, longest], (float)vectors[1, longest], (float)vectors[2, longest]);
            if (Vector3.Dot(up, upHint) < 0) up = -up;
            // The side nearest the hint, across the axis
            var front = frontHint - Vector3.Dot(frontHint, up) * up;
            if (front.LengthSquared() < 1e-8f) front = Vector3.Cross(up, Math.Abs(up.Y) < 0.9f ? Vector3.UnitY : Vector3.UnitX);
            front = Vector3.Normalize(front);
            var z = -front;
            var x = Vector3.Cross(up, z);
            return new Matrix4x4(
                x.X, up.X, z.X, 0,
                x.Y, up.Y, z.Y, 0,
                x.Z, up.Z, z.Z, 0,
                0, 0, 0, 1);
        }

        /// <summary>Eigenvalues and eigenvectors (columns) of a symmetric 3x3 matrix, by Jacobi rotations.</summary>
        private static (double[] values, double[,] vectors) Eigen3(double[,] input)
        {
            var a = (double[,])input.Clone();
            var v = new double[3, 3] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
            for (int sweep = 0; sweep < 32; sweep++)
            {
                double off = a[0, 1] * a[0, 1] + a[0, 2] * a[0, 2] + a[1, 2] * a[1, 2];
                if (off < 1e-20) break;
                for (int p = 0; p < 2; p++)
                {
                    for (int q = p + 1; q < 3; q++)
                    {
                        if (Math.Abs(a[p, q]) < 1e-30) continue;
                        double theta = (a[q, q] - a[p, p]) / (2 * a[p, q]);
                        double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                        if (theta == 0) t = 1;
                        double c = 1 / Math.Sqrt(t * t + 1), s = t * c;
                        for (int k = 0; k < 3; k++)
                        {
                            double akp = a[k, p], akq = a[k, q];
                            a[k, p] = c * akp - s * akq;
                            a[k, q] = s * akp + c * akq;
                        }
                        for (int k = 0; k < 3; k++)
                        {
                            double apk = a[p, k], aqk = a[q, k];
                            a[p, k] = c * apk - s * aqk;
                            a[q, k] = s * apk + c * aqk;
                        }
                        for (int k = 0; k < 3; k++)
                        {
                            double vkp = v[k, p], vkq = v[k, q];
                            v[k, p] = c * vkp - s * vkq;
                            v[k, q] = s * vkp + c * vkq;
                        }
                    }
                }
            }
            return (new[] { a[0, 0], a[1, 1], a[2, 2] }, v);
        }

        /// <summary>
        /// The bounds of what the parts draw, in the space <paramref name="space"/> takes the item's space to: every
        /// vertex the CPU kept (each part's kept half only), else the meshes' boxes.
        /// </summary>
        public static bool Bounds(IReadOnlyList<IconPart> parts, Matrix4x4 space, out Vector3 min, out Vector3 max)
        {
            min = new Vector3(float.MaxValue);
            max = new Vector3(float.MinValue);
            bool any = false;
            Span<Vector3> corners = stackalloc Vector3[8];
            foreach (var part in parts)
            {
                var mesh = part.Mesh;
                if (mesh == null) continue;
                var m = mesh.RootRotation * part.Transform * space;
                var positions = mesh.Positions;
                if (positions != null && positions.Length > 0)
                {
                    foreach (var p in positions)
                    {
                        if (part.Side == IconSide.Positive && p.X < part.SideCut) continue;
                        if (part.Side == IconSide.Negative && p.X >= part.SideCut) continue;
                        var q = Vector3.Transform(p, m);
                        min = Vector3.Min(min, q);
                        max = Vector3.Max(max, q);
                        any = true;
                    }
                }
                else
                {
                    var box = mesh.BoundingBox;
                    var bmin = box.Min; var bmax = box.Max;
                    for (int i = 0; i < 8; i++)
                    {
                        var c = new Vector3((i & 1) != 0 ? bmax.X : bmin.X, (i & 2) != 0 ? bmax.Y : bmin.Y, (i & 4) != 0 ? bmax.Z : bmin.Z);
                        var q = Vector3.Transform(c, m);
                        min = Vector3.Min(min, q);
                        max = Vector3.Max(max, q);
                        any = true;
                    }
                }
            }
            return any;
        }

        /// <summary>
        /// The picture, width × height RGBA8 with straight alpha, sRGB: what a PNG holds. Null when nothing could be
        /// drawn (no part with a mesh on the GPU) or the device was lost.
        /// </summary>
        public byte[]? Render(IReadOnlyList<IconPart> parts, IconView view, int width, int height)
        {
            if (parts == null || parts.Count == 0 || width <= 0 || height <= 0) return null;
            if (!Bounds(parts, view.Rotation, out var min, out var max)) return null;

            int ss = Math.Clamp(view.Supersample, 1, 8);
            int w = width * ss, h = height * ss;
            EnsureTargets(w, h);

            // Orthographic framing: the item's box centred, its larger share of the picture filled up to the margin
            var center = (min + max) * 0.5f;
            var extent = max - min;
            float aspect = (float)width / height;
            float fill = Math.Max(0.05f, 1f - 2f * Math.Clamp(view.Margin, 0f, 0.45f));
            float halfWidth = Math.Max(extent.X * 0.5f, extent.Y * 0.5f * aspect) / fill;
            halfWidth = Math.Max(halfWidth, 1e-4f);
            float halfHeight = halfWidth / aspect;
            // Depth: the item's own range (after the translation its box spans min.Z - center.Z .. max.Z - center.Z)
            float depth = Math.Max(extent.Z, 1e-3f);
            float zNear = min.Z - center.Z - depth * 0.05f, zFar = max.Z - center.Z + depth * 0.05f;
            float zScale = 1f / (zFar - zNear), zOffset = -zNear * zScale;
            var toView = view.Rotation * Matrix4x4.CreateTranslation(-center);

            var device = Engine.Device;
            StreamingManager.Instance?.Flush();
            _allocator.Reset();
            _cmd.Reset(_allocator);
            device.WaitForCopyQueue();
            _cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            uint materialsIdx = Material.BindMaterialsBuffer(_cmd, device);

            _cmd.ResourceBarrierTransition(_color!.Native, ResourceStates.Common, ResourceStates.RenderTarget);
            _cmd.OMSetRenderTargets(_color.RtvHandle, _depth!.DsvHandle);
            _cmd.RSSetViewport(new Viewport(0, 0, w, h, 0f, 1f));
            _cmd.RSSetScissorRect(new RectI(0, 0, w, h));
            _cmd.ClearRenderTargetView(_color.RtvHandle, new Color4(0f, 0f, 0f, 0f));
            _cmd.ClearDepthStencilView(_depth.DsvHandle, ClearFlags.Depth, 1f, 0);
            _cmd.SetGraphicsRootSignature(device.GlobalRootSignature);
            _cmd.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);

            int drawn = 0;
            foreach (var part in parts)
            {
                var mesh = part.Mesh;
                if (mesh == null || mesh.PosBufferIndex == 0 || mesh.IndexBufferIndex == 0) continue;
                var m = mesh.RootRotation * part.Transform * toView;
                // Its middle in view space, for what faces in towards it
                var partCenter = Bounds(new[] { part }, view.Rotation, out var pmin, out var pmax)
                    ? (pmin + pmax) * 0.5f - center : Vector3.Zero;
                var indices =mesh.LODs.Count > 0 && mesh.LODs[0].MeshPartIndices != null ? mesh.LODs[0].MeshPartIndices : null;
                int count = indices?.Length ?? mesh.MeshParts.Count;
                for (int k = 0; k < count; k++)
                {
                    int partIndex = indices != null ? indices[k] : k;
                    if (partIndex < 0 || partIndex >= mesh.MeshParts.Count) continue;
                    var meshPart = mesh.MeshParts[partIndex];
                    if (!meshPart.Enabled || meshPart.NumIndices <= 0) continue;
                    Material? material = null;
                    if (part.Materials != null && meshPart.MaterialSlot >= 0 && meshPart.MaterialSlot < part.Materials.Count)
                        material = part.Materials[meshPart.MaterialSlot];
                    int materialId = material != null && material.MaterialID >= 0 ? material.MaterialID : _material.MaterialID;

                    _material.Apply(_cmd, device);
                    uint flags = (uint)part.Side | (part.AlphaTest ? 4u : 0u);
                    Set(0, mesh.PosBufferIndex);
                    Set(1, mesh.NormBufferIndex);
                    Set(2, mesh.UVBufferIndex);
                    Set(3, mesh.IndexBufferIndex);
                    Set(4, (uint)meshPart.BaseIndex);
                    Set(5, materialsIdx);
                    Set(6, (uint)Math.Max(0, materialId));
                    Set(7, flags);
                    // icon.fx's rows: float4s, the projection in their w
                    SetF(8, m.M11); SetF(9, m.M12); SetF(10, m.M13); SetF(11, 1f / halfWidth);
                    SetF(12, m.M21); SetF(13, m.M22); SetF(14, m.M23); SetF(15, 1f / halfHeight);
                    SetF(16, m.M31); SetF(17, m.M32); SetF(18, m.M33); SetF(19, zScale);
                    SetF(20, m.M41); SetF(21, m.M42); SetF(22, m.M43); SetF(23, zOffset);
                    SetF(24, view.Exposure);
                    SetF(25, part.SideCut);
                    SetF(28, partCenter.X); SetF(29, partCenter.Y); SetF(30, partCenter.Z);
                    SetF(31, Math.Clamp(view.Interior, 0f, 1f));
                    _cmd.DrawInstanced((uint)meshPart.NumIndices, 1, 0, 0);
                    drawn++;
                }
            }

            var pixels = Readback(w, h);
            if (pixels == null || drawn == 0) return null;
            return Downsample(pixels, w, h, ss);

            void Set(uint slot, uint value) => _cmd.SetGraphicsRoot32BitConstant(0, value, slot);
            void SetF(uint slot, float value) => _cmd.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(value), slot);
        }

        private void EnsureTargets(int w, int h)
        {
            if (_color != null && _targetWidth == w && _targetHeight == h) return;
            _color?.Dispose();
            _depth?.Dispose();
            _color = new RenderTexture2D(Engine.Device, w, h, Format.R16G16B16A16_Float);
            _depth = new DepthTexture2D(w, h);
            _targetWidth = w;
            _targetHeight = h;
        }

        /// <summary>Copies the target out (recorded on the same list), submits and waits: RGBA half floats, tightly packed.</summary>
        private Half[]? Readback(int w, int h)
        {
            var device = Engine.Device;
            uint rowPitch = (uint)((w * 8 + 255) & ~255);
            ulong size = (ulong)rowPitch * (uint)h;
            var buffer = device.CreateCommittedResource(
                new HeapProperties(HeapType.Readback), HeapFlags.None, ResourceDescription.Buffer(size), ResourceStates.CopyDest);
            try
            {
                var source = _color!.Native;
                _cmd.ResourceBarrierTransition(source, ResourceStates.RenderTarget, ResourceStates.CopySource);
                var dst = new TextureCopyLocation(buffer, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R16G16B16A16_Float, (uint)w, (uint)h, 1, rowPitch),
                });
                _cmd.CopyTextureRegion(dst, 0, 0, 0, new TextureCopyLocation(source, 0));
                _cmd.ResourceBarrierTransition(source, ResourceStates.CopySource, ResourceStates.Common);
                _cmd.Close();
                device.SubmitAndWait(_cmd);
                if (device.IsDeviceLost) return null;

                var result = new Half[w * h * 4];
                unsafe
                {
                    void* mapped;
                    buffer.Map(0, &mapped);
                    var bytes = MemoryMarshal.AsBytes(result.AsSpan());
                    for (int y = 0; y < h; y++)
                        new ReadOnlySpan<byte>((byte*)mapped + (long)y * rowPitch, w * 8).CopyTo(bytes.Slice(y * w * 8, w * 8));
                    buffer.Unmap(0);
                }
                return result;
            }
            finally
            {
                buffer.Dispose();
            }
        }

        /// <summary>Averages each ss × ss block (colour weighted by coverage), then straight alpha and sRGB.</summary>
        private static byte[] Downsample(Half[] src, int w, int h, int ss)
        {
            int ow = w / ss, oh = h / ss;
            var result = new byte[ow * oh * 4];
            float n = ss * ss;
            for (int oy = 0; oy < oh; oy++)
            {
                for (int ox = 0; ox < ow; ox++)
                {
                    float r = 0, g = 0, b = 0, a = 0;
                    for (int sy = 0; sy < ss; sy++)
                    {
                        int row = (oy * ss + sy) * w;
                        for (int sx = 0; sx < ss; sx++)
                        {
                            int i = (row + ox * ss + sx) * 4;
                            float alpha = (float)src[i + 3];
                            r += (float)src[i] * alpha;
                            g += (float)src[i + 1] * alpha;
                            b += (float)src[i + 2] * alpha;
                            a += alpha;
                        }
                    }
                    int o = (oy * ow + ox) * 4;
                    if (a <= 0f) continue;
                    result[o] = Encode(r / a);
                    result[o + 1] = Encode(g / a);
                    result[o + 2] = Encode(b / a);
                    result[o + 3] = (byte)Math.Clamp((int)MathF.Round(a / n * 255f), 0, 255);
                }
            }
            return result;
        }

        private static byte Encode(float linear)
        {
            linear = Math.Clamp(linear, 0f, 1f);
            float s = linear <= 0.0031308f ? linear * 12.92f : 1.055f * MathF.Pow(linear, 1f / 2.4f) - 0.055f;
            return (byte)Math.Clamp((int)MathF.Round(s * 255f), 0, 255);
        }

        public void Dispose()
        {
            _cmd.Dispose();
            _allocator.Dispose();
            _color?.Dispose();
            _depth?.Dispose();
            if (ReferenceEquals(_shared, this)) _shared = null;
        }
    }
}
