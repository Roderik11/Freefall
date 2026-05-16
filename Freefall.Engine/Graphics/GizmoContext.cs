using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Base;
using Freefall.Components;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// Immediate-mode gizmo drawing and interaction API.
    /// 
    /// Drawing: Accumulates camera-facing billboard quads per frame, grouped by color,
    /// then flushes them via CommandBuffer.Enqueue through gbuffer_gizmo.fx.
    /// Uses persistently-mapped upload heap buffers (like SpriteBatch).
    /// 
    /// Handles: GPU-picked interactive handles (FreeMoveHandle, RadiusHandle, SliderHandle).
    /// Each handle renders as a separate MeshPart so the EntityIdBuffer can identify it.
    /// GizmoManager routes pick results back to set HotControl.
    /// </summary>
    public class GizmoContext
    {
        // ── Public State ──

        /// <summary>Current draw color. Set before calling Draw* methods.</summary>
        public Color4 Color = new Color4(1, 1, 1, 1);

        /// <summary>Transform applied to all positions (typically entity.Transform.Matrix).</summary>
        public Matrix4x4 Matrix = Matrix4x4.Identity;

        /// <summary>Line thickness in screen pixels. Constant size regardless of distance.</summary>
        public float LineWidth = 1.5f;

        /// <summary>True if any handle was modified this frame.</summary>
        public bool Changed { get; private set; }

        // ══════════════════════════════════════════════════════
        // ── Handle System (GPU-picked via EntityIdBuffer) ──
        // ══════════════════════════════════════════════════════

        /// <summary>
        /// The handle (MeshPart) currently being dragged. -1 = none.
        /// Set by GizmoManager when a GPU pick matches the gizmo slot.
        /// Cleared on mouse release.
        /// </summary>
        public int HotControl { get; set; } = -1;

        /// <summary>
        /// The handle (MeshPart) that the mouse is hovering over.
        /// Set by GizmoManager from hover picks.
        /// </summary>
        public int HoverControl { get; set; } = -1;

        // Per-frame handle ID counter (maps to MeshPart indices)
        // Handle IDs start at 1 (0 is reserved for the static gizmo geometry)
        private int _nextHandleId;

        // Mouse / camera state (set by GizmoManager each frame)
        public Camera Camera;
        public bool MouseDown;      // left button held

        // Per-handle drag state
        private struct DragState
        {
            public Plane DragPlane;
            public Vector3 DragStartWorld;
            public Vector3 DragStartValue;
            public float DragStartScalar;
            public bool Initialized;
        }

        private readonly Dictionary<int, DragState> _dragStates = new();

        // Per-handle geometry buckets (flushed as separate MeshParts)
        private readonly Dictionary<int, HandleBucket> _handleBuckets = new();

        private class HandleBucket
        {
            public readonly List<Vector3> Positions = new();
            public readonly List<Vector3> Normals = new();
            public readonly List<Vector2> UVs = new();
            public readonly List<uint> Indices = new();
            public uint ColorKey;

            public void Clear()
            {
                Positions.Clear();
                Normals.Clear();
                UVs.Clear();
                Indices.Clear();
            }
        }

        /// <summary>Called at the start of each frame.</summary>
        public void BeginHandles()
        {
            _nextHandleId = 1; // 0 = static gizmo geo
            Changed = false;

            // Clean up drag state for handles that no longer exist
            // (handles are re-registered every frame by DrawGizmos calls)
        }

        /// <summary>Called after all DrawGizmos. Clears hot control on mouse release.</summary>
        public void EndHandles()
        {
            if (!MouseDown && HotControl >= 0)
            {
                _dragStates.Remove(HotControl);
                HotControl = -1;
            }
        }

        private int AllocateHandleId()
        {
            return _nextHandleId++;
        }

        /// <summary>Get a world-space ray from the current mouse position.</summary>
        private Ray GetMouseRay()
        {
            return Camera?.MouseRay() ?? new Ray(Vector3.Zero, Vector3.UnitZ);
        }

        /// <summary>Intersect a ray with a plane.</summary>
        private static bool RayPlane(Ray ray, Plane plane, out Vector3 point)
        {
            return Collision.RayIntersectsPlane(ref ray, ref plane, out point);
        }

        /// <summary>Transform a local-space point to world-space.</summary>
        private Vector3 LocalToWorld(Vector3 local) => Vector3.Transform(local, Matrix);

        /// <summary>Transform a world-space point to local-space.</summary>
        private Vector3 WorldToLocal(Vector3 world)
        {
            Matrix4x4.Invert(Matrix, out var inv);
            return Vector3.Transform(world, inv);
        }

        // ══════════════════════════════════════════════════════════
        // ── Handles (GPU-picked, draw + interact in one call) ──
        // ══════════════════════════════════════════════════════════

        /// <summary>
        /// Draggable point handle on the camera-facing plane.
        /// Like Unity's Handles.FreeMoveHandle.
        /// </summary>
        public Vector3 FreeMoveHandle(Vector3 position, out bool clicked, float size = 8f)
        {
            int id = AllocateHandleId();
            var worldPos = LocalToWorld(position);

            bool isHot = (HotControl == id);
            bool isHover = (HoverControl == id);

            // Draw handle visual (solid camera-facing quad) into handle-specific bucket
            var savedColor = Color;
            Color = isHot ? new Color4(1f, 1f, 1f, 1f) : isHover ? new Color4(1f, 1f, 0f, 1f) : savedColor;
            PushHandleBucket(id);
            EmitSolidQuad(worldPos, size);
            PopHandleBucket();
            Color = savedColor;

            clicked = false;

            // Drag interaction
            if (isHot && MouseDown && Camera != null)
            {
                if (!_dragStates.TryGetValue(id, out var state) || !state.Initialized)
                {
                    clicked = true;

                    state = new DragState
                    {
                        DragPlane = new Plane(Camera.Forward, -Vector3.Dot(Camera.Forward, worldPos)),
                        DragStartValue = position,
                        Initialized = true
                    };
                    RayPlane(GetMouseRay(), state.DragPlane, out state.DragStartWorld);
                    _dragStates[id] = state;
                }

                var ray = GetMouseRay();
                if (RayPlane(ray, state.DragPlane, out var currentWorld))
                {
                    var delta = currentWorld - state.DragStartWorld;
                    var newWorld = LocalToWorld(state.DragStartValue) + delta;
                    Changed = true;
                    return WorldToLocal(newWorld);
                }
            }

            return position;
        }

        /// <summary>
        /// Radius handle: draws a camera-facing circle, drag to change radius.
        /// Like Unity's Handles.RadiusHandle.
        /// </summary>
        public float RadiusHandle(Vector3 center, float radius)
        {
            int id = AllocateHandleId();
            var worldCenter = LocalToWorld(center);

            bool isHot = (HotControl == id);
            bool isHover = (HoverControl == id);

            // Draw the radius circle (camera-facing) into handle bucket
            var savedColor = Color;
            Color = isHot ? new Color4(1f, 1f, 1f, 1f) : isHover ? new Color4(1f, 1f, 0f, 1f) : savedColor;
            var savedWidth = LineWidth;
            LineWidth = isHot || isHover ? LineWidth + 1 : LineWidth;

            if (Camera != null)
            {
                var camDir = Vector3.Normalize(_cameraPosition - worldCenter);
                PushHandleBucket(id);
                var savedMatrix = Matrix;
                Matrix = Matrix4x4.Identity; // draw in world space
                DrawCircle(worldCenter, camDir, radius, 48);
                Matrix = savedMatrix;
                PopHandleBucket();
            }

            Color = savedColor;
            LineWidth = savedWidth;

            // Drag interaction
            if (isHot && MouseDown && Camera != null)
            {
                if (!_dragStates.TryGetValue(id, out var state) || !state.Initialized)
                {
                    var camDir = Vector3.Normalize(_cameraPosition - worldCenter);
                    state = new DragState
                    {
                        DragPlane = new Plane(camDir, -Vector3.Dot(camDir, worldCenter)),
                        DragStartScalar = radius,
                        Initialized = true
                    };
                    RayPlane(GetMouseRay(), state.DragPlane, out state.DragStartWorld);
                    _dragStates[id] = state;
                }

                var ray = GetMouseRay();
                if (RayPlane(ray, state.DragPlane, out var currentWorld))
                {
                    float newRadius = Vector3.Distance(currentWorld, worldCenter);
                    Changed = true;
                    return MathF.Max(0.01f, newRadius);
                }
            }

            return radius;
        }

        /// <summary>
        /// Slider handle: draws a line, drag along a single axis.
        /// Like Unity's Handles.Slider.
        /// </summary>
        public Vector3 SliderHandle(Vector3 position, Vector3 direction, float length = 1f)
        {
            int id = AllocateHandleId();
            direction = Vector3.Normalize(direction);

            var worldPos = LocalToWorld(position);
            var worldEnd = LocalToWorld(position + direction * length);
            var worldDir = Vector3.Normalize(worldEnd - worldPos);

            bool isHot = (HotControl == id);
            bool isHover = (HoverControl == id);

            // Draw the axis line into handle bucket
            var savedColor = Color;
            Color = isHot ? new Color4(1f, 1f, 1f, 1f) : isHover ? new Color4(1f, 1f, 0f, 1f) : savedColor;
            PushHandleBucket(id);
            DrawLine(position, position + direction * length);
            PopHandleBucket();
            Color = savedColor;

            // Drag interaction
            if (isHot && MouseDown && Camera != null)
            {
                if (!_dragStates.TryGetValue(id, out var state) || !state.Initialized)
                {
                    var camDir = Vector3.Normalize(_cameraPosition - worldPos);
                    var planeNormal = Vector3.Normalize(Vector3.Cross(worldDir, Vector3.Cross(camDir, worldDir)));
                    state = new DragState
                    {
                        DragPlane = new Plane(planeNormal, -Vector3.Dot(planeNormal, worldPos)),
                        DragStartValue = position,
                        Initialized = true
                    };
                    RayPlane(GetMouseRay(), state.DragPlane, out state.DragStartWorld);
                    _dragStates[id] = state;
                }

                var ray = GetMouseRay();
                if (RayPlane(ray, state.DragPlane, out var currentWorld))
                {
                    var delta = currentWorld - _dragStates[id].DragStartWorld;
                    float axisDelta = Vector3.Dot(delta, worldDir);
                    var newWorld = LocalToWorld(_dragStates[id].DragStartValue) + worldDir * axisDelta;
                    Changed = true;
                    return WorldToLocal(newWorld);
                }
            }

            return position;
        }

        // ── Handle Bucket Management ──

        // Stack to support nested handle drawing (handle bucket vs static bucket)
        private int _activeHandleId = -1;

        private void PushHandleBucket(int handleId)
        {
            _activeHandleId = handleId;
            if (!_handleBuckets.TryGetValue(handleId, out _))
            {
                _handleBuckets[handleId] = new HandleBucket();
            }
        }

        private void PopHandleBucket()
        {
            _activeHandleId = -1;
        }

        /// <summary>
        /// Emit a solid camera-facing quad at a world-space position.
        /// Size is in screen pixels (constant regardless of distance).
        /// </summary>
        private void EmitSolidQuad(Vector3 worldPos, float sizePixels)
        {
            if (Camera == null) return;

            // Compute world-space half-size from pixel size
            float dist = Vector3.Distance(worldPos, _cameraPosition);
            float fovRad = Camera.FieldOfView * (MathF.PI / 180f);
            float vpH = Camera.Target?.Height ?? 1080;
            float worldPerPixel = 2f * dist * MathF.Tan(fovRad * 0.5f) / vpH;
            float halfSize = sizePixels * worldPerPixel * 0.5f;

            // Camera-facing axes
            Vector3 right = Camera.Right * halfSize;
            Vector3 up = Camera.Up * halfSize;
            Vector3 normal = Vector3.Normalize(worldPos - _cameraPosition);

            // Choose target buffer
            List<Vector3> positions, normals;
            List<Vector2> uvs;
            List<uint> indices;

            if (_activeHandleId >= 0 && _handleBuckets.TryGetValue(_activeHandleId, out var hb))
            {
                positions = hb.Positions;
                normals = hb.Normals;
                uvs = hb.UVs;
                indices = hb.Indices;
                hb.ColorKey = PackColor(Color);
            }
            else
            {
                var bucket = GetBucket();
                positions = bucket.Positions;
                normals = bucket.Normals;
                uvs = bucket.UVs;
                indices = bucket.Indices;
            }

            uint baseIdx = (uint)positions.Count;

            positions.Add(worldPos - right - up);
            positions.Add(worldPos + right - up);
            positions.Add(worldPos - right + up);
            positions.Add(worldPos + right + up);

            normals.Add(normal);
            normals.Add(normal);
            normals.Add(normal);
            normals.Add(normal);

            uvs.Add(new Vector2(0, 0));
            uvs.Add(new Vector2(1, 0));
            uvs.Add(new Vector2(0, 1));
            uvs.Add(new Vector2(1, 1));

            indices.Add(baseIdx + 0);
            indices.Add(baseIdx + 2);
            indices.Add(baseIdx + 1);
            indices.Add(baseIdx + 1);
            indices.Add(baseIdx + 2);
            indices.Add(baseIdx + 3);
        }

        // ════════════════════════════════════════════
        // ── Per-Color Geometry Buckets (static) ──
        // ════════════════════════════════════════════

        private class ColorBucket
        {
            public readonly List<Vector3> Positions = new();
            public readonly List<Vector3> Normals = new();
            public readonly List<Vector2> UVs = new();
            public readonly List<uint> Indices = new();

            public void Clear()
            {
                Positions.Clear();
                Normals.Clear();
                UVs.Clear();
                Indices.Clear();
            }
        }

        // ── Upload-Heap Dynamic Mesh ──

        private class DynamicMesh : IDisposable
        {
            private const int InitialVertexCapacity = 1024;
            private const int InitialIndexCapacity = 1536;

            private ID3D12Resource _posBuffer;
            private ID3D12Resource _normBuffer;
            private ID3D12Resource _uvBuffer;
            private ID3D12Resource _indexBuffer;

            private IntPtr _posPtr, _normPtr, _uvPtr, _indexPtr;

            public uint PosBindless { get; private set; }
            public uint NormBindless { get; private set; }
            public uint UVBindless { get; private set; }
            public uint IndexBindless { get; private set; }

            private int _vertexCapacity;
            private int _indexCapacity;

            public Mesh ProxyMesh { get; private set; }

            public void EnsureCreated(GraphicsDevice device, int vertexCount, int indexCount)
            {
                bool needsRebuild = false;

                if (_posBuffer == null)
                {
                    _vertexCapacity = Math.Max(InitialVertexCapacity, vertexCount);
                    _indexCapacity = Math.Max(InitialIndexCapacity, indexCount);
                    needsRebuild = true;
                }
                else if (vertexCount > _vertexCapacity || indexCount > _indexCapacity)
                {
                    Dispose();
                    _vertexCapacity = Math.Max(_vertexCapacity * 2, vertexCount);
                    _indexCapacity = Math.Max(_indexCapacity * 2, indexCount);
                    needsRebuild = true;
                }

                if (!needsRebuild) return;

                _posBuffer = CreateUploadBuffer(device, _vertexCapacity * 12);
                _normBuffer = CreateUploadBuffer(device, _vertexCapacity * 12);
                _uvBuffer = CreateUploadBuffer(device, _vertexCapacity * 8);
                _indexBuffer = CreateUploadBuffer(device, _indexCapacity * 4);

                _posPtr = MapBuffer(_posBuffer);
                _normPtr = MapBuffer(_normBuffer);
                _uvPtr = MapBuffer(_uvBuffer);
                _indexPtr = MapBuffer(_indexBuffer);

                if (PosBindless == 0) PosBindless = device.AllocateBindlessIndex();
                if (NormBindless == 0) NormBindless = device.AllocateBindlessIndex();
                if (UVBindless == 0) UVBindless = device.AllocateBindlessIndex();
                if (IndexBindless == 0) IndexBindless = device.AllocateBindlessIndex();

                CreateSRV(device, _posBuffer, (uint)_vertexCapacity, 12, PosBindless);
                CreateSRV(device, _normBuffer, (uint)_vertexCapacity, 12, NormBindless);
                CreateSRV(device, _uvBuffer, (uint)_vertexCapacity, 8, UVBindless);
                CreateSRV(device, _indexBuffer, (uint)_indexCapacity, 4, IndexBindless);

                RebuildProxy(device);
            }

            public unsafe void Upload(List<Vector3> positions, List<Vector3> normals, List<Vector2> uvs, List<uint> indices)
            {
                int vertCount = positions.Count;
                int idxCount = indices.Count;
                if (vertCount == 0 || idxCount == 0) return;

                fixed (Vector3* src = CollectionsMarshal.AsSpan(positions))
                    Buffer.MemoryCopy(src, (void*)_posPtr, _vertexCapacity * 12, vertCount * 12);

                fixed (Vector3* src = CollectionsMarshal.AsSpan(normals))
                    Buffer.MemoryCopy(src, (void*)_normPtr, _vertexCapacity * 12, vertCount * 12);

                fixed (Vector2* src = CollectionsMarshal.AsSpan(uvs))
                    Buffer.MemoryCopy(src, (void*)_uvPtr, _vertexCapacity * 8, vertCount * 8);

                fixed (uint* src = CollectionsMarshal.AsSpan(indices))
                    Buffer.MemoryCopy(src, (void*)_indexPtr, _indexCapacity * 4, idxCount * 4);

                // Set NumIndices on MeshPart 0 (always exists)
                if (ProxyMesh != null && ProxyMesh.MeshParts.Count > 0)
                    ProxyMesh.MeshParts[0].NumIndices = idxCount;
            }

            /// <summary>Set NumIndices for a specific MeshPart. Creates parts up to the index if needed.</summary>
            public void SetMeshPartIndices(int partIndex, int numIndices, int baseIndex)
            {
                while (ProxyMesh.MeshParts.Count <= partIndex)
                {
                    ProxyMesh.MeshParts.Add(new MeshPart
                    {
                        NumIndices = 0,
                        BaseIndex = 0,
                        BoundingBox = new BoundingBox(new Vector3(-10000), new Vector3(10000)),
                        BoundingSphere = new Vector4(0, 0, 0, 10000)
                    });
                }
                ProxyMesh.MeshParts[partIndex].NumIndices = numIndices;
                ProxyMesh.MeshParts[partIndex].BaseIndex = baseIndex;
            }

            private void RebuildProxy(GraphicsDevice device)
            {
                ProxyMesh = new Mesh();
                ProxyMesh.MeshParts.Add(new MeshPart
                {
                    NumIndices = 0,
                    BoundingBox = new BoundingBox(new Vector3(-10000), new Vector3(10000)),
                    BoundingSphere = new Vector4(0, 0, 0, 10000)
                });
                ProxyMesh.PosBufferIndex = PosBindless;
                ProxyMesh.NormBufferIndex = NormBindless;
                ProxyMesh.UVBufferIndex = UVBindless;
                ProxyMesh.IndexBufferIndex = IndexBindless;
            }

            private static ID3D12Resource CreateUploadBuffer(GraphicsDevice device, int sizeBytes)
            {
                return device.NativeDevice.CreateCommittedResource(
                    new HeapProperties(HeapType.Upload),
                    HeapFlags.None,
                    ResourceDescription.Buffer((ulong)sizeBytes),
                    ResourceStates.GenericRead);
            }

            private static unsafe IntPtr MapBuffer(ID3D12Resource buffer)
            {
                void* ptr;
                buffer.Map(0, null, &ptr);
                return (IntPtr)ptr;
            }

            private static void CreateSRV(GraphicsDevice device, ID3D12Resource resource, uint numElements, uint stride, uint bindlessIndex)
            {
                var srvDesc = new ShaderResourceViewDescription
                {
                    Format = Format.Unknown,
                    ViewDimension = ShaderResourceViewDimension.Buffer,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Buffer = new BufferShaderResourceView { FirstElement = 0, NumElements = numElements, StructureByteStride = stride }
                };
                device.NativeDevice.CreateShaderResourceView(resource, srvDesc, device.GetCpuHandle(bindlessIndex));
            }

            public void Dispose()
            {
                _posBuffer?.Dispose(); _posBuffer = null;
                _normBuffer?.Dispose(); _normBuffer = null;
                _uvBuffer?.Dispose(); _uvBuffer = null;
                _indexBuffer?.Dispose(); _indexBuffer = null;
            }
        }

        // ════════════════════════
        // ── Rendering State ──
        // ════════════════════════

        // Line gizmo buckets (billboard quads, no depth test)
        private readonly Dictionary<uint, ColorBucket> _buckets = new();
        private readonly Dictionary<uint, DynamicMesh> _dynamicMeshes = new();
        private readonly Dictionary<uint, Material> _materialCache = new();
        private Effect _effect;
        private bool _initialized;

        // Mesh gizmo buckets (solid triangles, depth test + alpha blend)
        private readonly Dictionary<uint, ColorBucket> _meshBuckets = new();
        private readonly Dictionary<uint, DynamicMesh> _meshDynamicMeshes = new();
        private readonly Dictionary<uint, Material> _meshMaterialCache = new();
        private Effect _meshEffect;

        // Deferred draw list for pre-built GPU meshes (enqueued during Flush)
        private readonly List<(Mesh mesh, int part, Material material)> _deferredMeshDraws = new();

        // Single DynamicMesh for all handle geometry (multiple MeshParts)
        private DynamicMesh _handleMesh;

        // Camera position for billboard computation
        private Vector3 _cameraPosition;

        public void SetCamera(Vector3 cameraPosition)
        {
            _cameraPosition = cameraPosition;
        }

        // ═══════════════════════════
        // ── Drawing Primitives ──
        // ═══════════════════════════

        /// <summary>Draw a line segment between two local-space points.</summary>
        public void DrawLine(Vector3 a, Vector3 b)
        {
            var wa = Vector3.Transform(a, Matrix);
            var wb = Vector3.Transform(b, Matrix);
            EmitLineQuad(wa, wb);
        }

        /// <summary>Draw a ray from origin in direction for the given length.</summary>
        public void DrawRay(Vector3 origin, Vector3 direction, float length)
        {
            DrawLine(origin, origin + Vector3.Normalize(direction) * length);
        }

        /// <summary>Draw a wireframe axis-aligned box.</summary>
        public void DrawWireBox(Vector3 center, Vector3 size)
        {
            var half = size * 0.5f;
            var v000 = center + new Vector3(-half.X, -half.Y, -half.Z);
            var v001 = center + new Vector3(-half.X, -half.Y,  half.Z);
            var v010 = center + new Vector3(-half.X,  half.Y, -half.Z);
            var v011 = center + new Vector3(-half.X,  half.Y,  half.Z);
            var v100 = center + new Vector3( half.X, -half.Y, -half.Z);
            var v101 = center + new Vector3( half.X, -half.Y,  half.Z);
            var v110 = center + new Vector3( half.X,  half.Y, -half.Z);
            var v111 = center + new Vector3( half.X,  half.Y,  half.Z);

            DrawLine(v000, v100); DrawLine(v100, v101); DrawLine(v101, v001); DrawLine(v001, v000);
            DrawLine(v010, v110); DrawLine(v110, v111); DrawLine(v111, v011); DrawLine(v011, v010);
            DrawLine(v000, v010); DrawLine(v100, v110); DrawLine(v101, v111); DrawLine(v001, v011);
        }

        /// <summary>Draw a wireframe sphere (3 axis-aligned circles).</summary>
        public void DrawWireSphere(Vector3 center, float radius, int segments = 32)
        {
            DrawCircle(center, Vector3.UnitY, radius, segments);
            DrawCircle(center, Vector3.UnitX, radius, segments);
            DrawCircle(center, Vector3.UnitZ, radius, segments);
        }

        // 2 half-spheres + 2 circles at the "equator" to connect them
        public void DrawCapsule(Vector3 center, float height, float radius, int segments = 16)
        {
            Vector3 top = center + Vector3.UnitY * (height * 0.5f - radius);
            Vector3 bottom = center - Vector3.UnitY * (height * 0.5f - radius);
            DrawHemisphere(top, Vector3.UnitY, radius, segments);
            DrawHemisphere(bottom, -Vector3.UnitY, radius, segments);
            DrawCircle(top, Vector3.UnitY, radius, segments);
            DrawCircle(bottom, Vector3.UnitY, radius, segments);
        }

        public void DrawHemisphere(Vector3 center, Vector3 normal, float radius, int segments = 16)
        {
            normal = Vector3.Normalize(normal);
            Vector3 tangent = GetPerpendicular(normal);
            Vector3 bitangent = Vector3.Cross(normal, tangent);
            for (int i = 0; i <= segments; i++)
            {
                float v = (float)i / segments * (MathF.PI / 2f);
                float r = MathF.Cos(v) * radius;
                float y = MathF.Sin(v) * radius;
                DrawCircle(center + normal * y, normal, r, segments);
            }
        }

        /// <summary>Draw a circle in the plane defined by center and normal.</summary>
        public void DrawCircle(Vector3 center, Vector3 normal, float radius, int segments = 32)
        {
            normal = Vector3.Normalize(normal);
            Vector3 tangent = GetPerpendicular(normal);
            Vector3 bitangent = Vector3.Cross(normal, tangent);

            float step = MathF.PI * 2f / segments;
            Vector3 prev = center + tangent * radius;
            for (int i = 1; i <= segments; i++)
            {
                float angle = step * i;
                Vector3 point = center + (tangent * MathF.Cos(angle) + bitangent * MathF.Sin(angle)) * radius;
                DrawLine(prev, point);
                prev = point;
            }
        }

        /// <summary>Draw an arc from a starting direction, sweeping the given angle.</summary>
        public void DrawArc(Vector3 center, Vector3 normal, Vector3 from, float angle, float radius, int segments = 32)
        {
            normal = Vector3.Normalize(normal);
            from = Vector3.Normalize(from);
            Vector3 bitangent = Vector3.Cross(normal, from);

            float step = angle / segments;
            Vector3 prev = center + from * radius;
            for (int i = 1; i <= segments; i++)
            {
                float a = step * i;
                Vector3 point = center + (from * MathF.Cos(a) + bitangent * MathF.Sin(a)) * radius;
                DrawLine(prev, point);
                prev = point;
            }
        }

        /// <summary>Draw a wireframe cone (circle base + lines to tip).</summary>
        public void DrawWireCone(Vector3 tip, Vector3 direction, float angle, float length, int segments = 16)
        {
            direction = Vector3.Normalize(direction);
            Vector3 baseCenter = tip + direction * length;
            float baseRadius = MathF.Tan(angle * 0.5f) * length;

            DrawCircle(baseCenter, direction, baseRadius, segments);

            Vector3 tangent = GetPerpendicular(direction);
            Vector3 bitangent = Vector3.Cross(direction, tangent);
            float step = MathF.PI * 2f / segments;
            for (int i = 0; i < segments; i += segments / 4)
            {
                float a = step * i;
                Vector3 point = baseCenter + (tangent * MathF.Cos(a) + bitangent * MathF.Sin(a)) * baseRadius;
                DrawLine(tip, point);
            }
        }

        // ══════════════════════════════════
        // ── Mesh Drawing (no billboards) ──
        // ══════════════════════════════════

        /// <summary>
        /// Draw a pre-built triangle mesh directly. No billboard expansion —
        /// positions and indices are pushed straight into the mesh bucket.
        /// Uses depth-tested, alpha-blended shader for proper scene integration.
        /// Much faster than DrawLine per-edge for large meshes (e.g. navmesh overlays).
        /// Positions are in world space (Matrix is NOT applied).
        /// For persistent meshes that don't change, prefer EnqueueMesh() instead.
        /// </summary>
        public void DrawMesh(Vector3[] positions, int[] indices)
        {
            if (positions == null || indices == null || positions.Length == 0 || indices.Length == 0)
                return;

            var bucket = GetMeshBucket();
            uint baseIdx = (uint)bucket.Positions.Count;
            var normal = Vector3.UnitY; // flat normal for unlit gizmo

            for (int i = 0; i < positions.Length; i++)
            {
                bucket.Positions.Add(positions[i]);
                bucket.Normals.Add(normal);
                bucket.UVs.Add(Vector2.Zero);
            }

            for (int i = 0; i < indices.Length; i++)
                bucket.Indices.Add(baseIdx + (uint)indices[i]);
        }

        /// <summary>
        /// Enqueue a pre-built GPU Mesh for rendering with the depth-tested gizmo shader.
        /// The mesh is rendered during Flush — zero per-frame CPU vertex work.
        /// Build the Mesh once (e.g. on bake), create a Material with the mesh gizmo effect,
        /// then call this each frame in DrawGizmos.
        /// </summary>
        public void EnqueueMesh(Mesh mesh, int meshPart, Material material)
        {
            if (mesh == null || material == null) return;
            _deferredMeshDraws.Add((mesh, meshPart, material));
        }

        /// <summary>
        /// Draw a pre-built triangle mesh as wireframe edges.
        /// Each triangle emits 3 billboard line-quads — still faster than
        /// calling DrawLine() N×3 times because it skips Matrix transforms.
        /// For very large meshes (>10k tris) prefer DrawMesh() with solid fill.
        /// Positions are in world space (Matrix is NOT applied).
        /// </summary>
        public void DrawWireframeMesh(Vector3[] positions, int[] indices)
        {
            if (positions == null || indices == null || positions.Length == 0 || indices.Length == 0)
                return;

            for (int i = 0; i < indices.Length; i += 3)
            {
                if (i + 2 >= indices.Length) break;
                int i0 = indices[i];
                int i1 = indices[i + 1];
                int i2 = indices[i + 2];

                if (i0 >= positions.Length || i1 >= positions.Length || i2 >= positions.Length)
                    continue;

                EmitLineQuad(positions[i0], positions[i1]);
                EmitLineQuad(positions[i1], positions[i2]);
                EmitLineQuad(positions[i2], positions[i0]);
            }
        }

        // ═══════════════════
        // ── Lifecycle ──
        // ═══════════════════

        /// <summary>Clear all accumulated geometry for next frame.</summary>
        public void Clear()
        {
            foreach (var bucket in _buckets.Values)
                bucket.Clear();

            foreach (var bucket in _meshBuckets.Values)
                bucket.Clear();

            _deferredMeshDraws.Clear();

            // Fully remove handle buckets — they are recreated by DrawGizmos each frame.
            // Just clearing contents would leave stale entries whose MeshParts persist
            // on the proxy mesh, causing ghost handles from previously-selected entities.
            _handleBuckets.Clear();
        }

        /// <summary>
        /// Submit accumulated geometry via CommandBuffer.Enqueue.
        /// Static gizmo geo goes as MeshPart 0 (per-color meshes).
        /// Handle geo goes into a shared handle mesh with MeshPart per handle.
        /// </summary>
        public void Flush(int gizmoSlot)
        {
            EnsureInitialized();

            TransformBuffer.Instance.SetTransform(gizmoSlot, Matrix4x4.Identity);

            // Zero out ALL existing proxy meshes so stale upload-heap data cannot render.
            void ZeroMeshes(Dictionary<uint, DynamicMesh> meshes)
            {
                foreach (var dynMesh in meshes.Values)
                {
                    if (dynMesh.ProxyMesh != null && dynMesh.ProxyMesh.MeshParts.Count > 0)
                    {
                        dynMesh.ProxyMesh.MeshParts[0].NumIndices = 0;
                        dynMesh.ProxyMesh.RegisterMeshParts();
                    }
                }
            }
            ZeroMeshes(_dynamicMeshes);
            ZeroMeshes(_meshDynamicMeshes);

            // 1. Flush line gizmo geometry (no depth test, always on top)
            FlushBuckets(_buckets, _dynamicMeshes, GetOrCreateMaterial, gizmoSlot);

            // 2. Flush mesh gizmo geometry (depth test + alpha blend)
            FlushBuckets(_meshBuckets, _meshDynamicMeshes, GetOrCreateMeshMaterial, gizmoSlot);

            // 3. Flush deferred pre-built GPU meshes
            foreach (var (mesh, part, material) in _deferredMeshDraws)
                CommandBuffer.Enqueue(mesh, part, material, new MaterialBlock(), gizmoSlot);

            // 4. Flush handle geometry (all handles merged into one mesh, each as a separate MeshPart)
            FlushHandles(gizmoSlot);
        }

        private void FlushBuckets(
            Dictionary<uint, ColorBucket> buckets,
            Dictionary<uint, DynamicMesh> meshes,
            Func<uint, Material> getMaterial,
            int gizmoSlot)
        {
            foreach (var (colorKey, bucket) in buckets)
            {
                if (bucket.Positions.Count == 0) continue;

                if (!meshes.TryGetValue(colorKey, out var dynMesh))
                {
                    dynMesh = new DynamicMesh();
                    meshes[colorKey] = dynMesh;
                }

                dynMesh.EnsureCreated(Engine.Device, bucket.Positions.Count, bucket.Indices.Count);
                dynMesh.Upload(bucket.Positions, bucket.Normals, bucket.UVs, bucket.Indices);
                dynMesh.ProxyMesh.RegisterMeshParts();

                var material = getMaterial(colorKey);
                CommandBuffer.Enqueue(dynMesh.ProxyMesh, 0, material, new MaterialBlock(), gizmoSlot);
            }
        }

        private void FlushHandles(int gizmoSlot)
        {
            // Collect all handle buckets that have geometry
            var activeHandles = new List<(int handleId, HandleBucket bucket)>();
            foreach (var (handleId, bucket) in _handleBuckets)
            {
                if (bucket.Positions.Count > 0)
                    activeHandles.Add((handleId, bucket));
            }

            if (activeHandles.Count == 0) return;

            // Merge all handle geometry into combined arrays
            var allPositions = new List<Vector3>();
            var allNormals = new List<Vector3>();
            var allUVs = new List<Vector2>();
            var allIndices = new List<uint>();

            // Track MeshPart ranges: handleId → (startIndex, numIndices)
            var partRanges = new List<(int handleId, int startIndex, int numIndices)>();

            foreach (var (handleId, bucket) in activeHandles)
            {
                int startIndex = allIndices.Count;
                uint baseVertex = (uint)allPositions.Count;

                allPositions.AddRange(bucket.Positions);
                allNormals.AddRange(bucket.Normals);
                allUVs.AddRange(bucket.UVs);

                // Offset indices by baseVertex
                foreach (var idx in bucket.Indices)
                    allIndices.Add(idx + baseVertex);

                partRanges.Add((handleId, startIndex, bucket.Indices.Count));
            }

            // Upload merged geometry
            _handleMesh ??= new DynamicMesh();
            _handleMesh.EnsureCreated(Engine.Device, allPositions.Count, allIndices.Count);
            _handleMesh.Upload(allPositions, allNormals, allUVs, allIndices);

            // Reset proxy mesh parts to match this frame's handles exactly.
            // Without this, MeshParts from a previously-selected entity (e.g. a
            // 10-point spline) persist when switching to a 1-handle entity,
            // causing out-of-bounds MeshPart lookups and ghost geometry.
            _handleMesh.ProxyMesh.MeshParts.Clear();
            _handleMesh.ProxyMesh.MeshParts.Add(new MeshPart
            {
                NumIndices = 0,
                BoundingBox = new BoundingBox(new Vector3(-10000), new Vector3(10000)),
                BoundingSphere = new Vector4(0, 0, 0, 10000)
            });

            // Create MeshParts for each handle
            foreach (var (handleId, startIndex, numIndices) in partRanges)
            {
                _handleMesh.SetMeshPartIndices(handleId, numIndices, startIndex);
            }

            // Re-register mesh parts so MeshRegistry IDs match the new part count
            _handleMesh.ProxyMesh.RegisterMeshParts();

            // Enqueue each handle as a separate draw call with its MeshPart index
            foreach (var (handleId, startIndex, numIndices) in partRanges)
            {
                var colorKey = _handleBuckets[handleId].ColorKey;
                var material = GetOrCreateMaterial(colorKey);
                CommandBuffer.Enqueue(_handleMesh.ProxyMesh, handleId, material, new MaterialBlock(), gizmoSlot);
            }
        }

        // ═══════════════════════════
        // ── Internal Helpers ──
        // ═══════════════════════════

        private void EnsureInitialized()
        {
            if (_initialized) return;
            _initialized = true;

            _effect = new Effect("gbuffer_gizmo");
            _meshEffect = new Effect("gbuffer_gizmo_mesh");
        }

        /// <summary>
        /// The depth-tested, alpha-blended Effect for mesh gizmos.
        /// Use this to create Materials for EnqueueMesh().
        /// </summary>
        public Effect MeshEffect
        {
            get { EnsureInitialized(); return _meshEffect; }
        }

        private uint PackColor(Color4 color)
        {
            byte r = (byte)(Math.Clamp(color.R, 0f, 1f) * 255f);
            byte g = (byte)(Math.Clamp(color.G, 0f, 1f) * 255f);
            byte b = (byte)(Math.Clamp(color.B, 0f, 1f) * 255f);
            byte a = (byte)(Math.Clamp(color.A, 0f, 1f) * 255f);
            return (uint)((r << 24) | (g << 16) | (b << 8) | a);
        }

        private Material GetOrCreateMaterial(uint colorKey)
        {
            if (_materialCache.TryGetValue(colorKey, out var mat))
                return mat;

            byte r = (byte)((colorKey >> 24) & 0xFF);
            byte g = (byte)((colorKey >> 16) & 0xFF);
            byte b = (byte)((colorKey >> 8) & 0xFF);

            byte[] data = new byte[4 * 4 * 4];
            for (int i = 0; i < data.Length; i += 4)
            {
                data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
            }

            var tex = Texture.CreateFromData(Engine.Device, 4, 4, data, Format.R8G8B8A8_UNorm);
            mat = new Material(_effect);
            mat.SetTexture("AlbedoTex", tex);
            _materialCache[colorKey] = mat;

            return mat;
        }

        private Material GetOrCreateMeshMaterial(uint colorKey)
        {
            if (_meshMaterialCache.TryGetValue(colorKey, out var mat))
                return mat;

            byte r = (byte)((colorKey >> 24) & 0xFF);
            byte g = (byte)((colorKey >> 16) & 0xFF);
            byte b = (byte)((colorKey >> 8) & 0xFF);
            byte a = (byte)(colorKey & 0xFF);

            byte[] data = new byte[4 * 4 * 4];
            for (int i = 0; i < data.Length; i += 4)
            {
                data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = a;
            }

            var tex = Texture.CreateFromData(Engine.Device, 4, 4, data, Format.R8G8B8A8_UNorm);
            mat = new Material(_meshEffect);
            mat.SetTexture("AlbedoTex", tex);
            _meshMaterialCache[colorKey] = mat;

            return mat;
        }

        private ColorBucket GetBucket()
        {
            uint key = PackColor(Color);
            if (!_buckets.TryGetValue(key, out var bucket))
            {
                bucket = new ColorBucket();
                _buckets[key] = bucket;
            }
            return bucket;
        }

        private ColorBucket GetMeshBucket()
        {
            uint key = PackColor(Color);
            if (!_meshBuckets.TryGetValue(key, out var bucket))
            {
                bucket = new ColorBucket();
                _meshBuckets[key] = bucket;
            }
            return bucket;
        }

        /// <summary>
        /// Emit a camera-facing billboard quad for a line segment.
        /// Routes to active handle bucket or static color bucket.
        /// </summary>
        private void EmitLineQuad(Vector3 a, Vector3 b)
        {
            Vector3 lineDir = b - a;
            float lineLen = lineDir.Length();
            if (lineLen < 1e-6f) return;
            lineDir /= lineLen;

            Vector3 midpoint = (a + b) * 0.5f;
            Vector3 viewDir = Vector3.Normalize(midpoint - _cameraPosition);

            Vector3 side = Vector3.Cross(lineDir, viewDir);
            float sideLen = side.Length();
            if (sideLen < 1e-6f)
            {
                side = GetPerpendicular(lineDir);
            }
            else
            {
                side /= sideLen;
            }

            // Compute constant-screen-size width:
            // worldPerPixel = 2 * dist * tan(fov/2) / viewportHeight
            float dist = Vector3.Distance(midpoint, _cameraPosition);
            float fovRad = (Camera?.FieldOfView ?? 45f) * (MathF.PI / 180f);
            float vpH = Camera?.Target?.Height ?? 1080;
            float worldPerPixel = 2f * dist * MathF.Tan(fovRad * 0.5f) / vpH;
            float worldWidth = LineWidth * worldPerPixel;

            Vector3 offset = side * (worldWidth * 0.5f);
            Vector3 normal = Vector3.Cross(lineDir, side);

            // Choose target buffer: handle bucket or static color bucket
            List<Vector3> positions, normals;
            List<Vector2> uvs;
            List<uint> indices;

            if (_activeHandleId >= 0 && _handleBuckets.TryGetValue(_activeHandleId, out var hb))
            {
                positions = hb.Positions;
                normals = hb.Normals;
                uvs = hb.UVs;
                indices = hb.Indices;
                hb.ColorKey = PackColor(Color);
            }
            else
            {
                var bucket = GetBucket();
                positions = bucket.Positions;
                normals = bucket.Normals;
                uvs = bucket.UVs;
                indices = bucket.Indices;
            }

            uint baseIdx = (uint)positions.Count;

            positions.Add(a - offset);
            positions.Add(a + offset);
            positions.Add(b - offset);
            positions.Add(b + offset);

            normals.Add(normal);
            normals.Add(normal);
            normals.Add(normal);
            normals.Add(normal);

            uvs.Add(new Vector2(0, 0));
            uvs.Add(new Vector2(1, 0));
            uvs.Add(new Vector2(0, 1));
            uvs.Add(new Vector2(1, 1));

            indices.Add(baseIdx + 0);
            indices.Add(baseIdx + 2);
            indices.Add(baseIdx + 1);
            indices.Add(baseIdx + 1);
            indices.Add(baseIdx + 2);
            indices.Add(baseIdx + 3);
        }

        /// <summary>Get a vector perpendicular to the given vector.</summary>
        private static Vector3 GetPerpendicular(Vector3 v)
        {
            v = Vector3.Normalize(v);
            if (MathF.Abs(v.Y) < 0.99f)
                return Vector3.Normalize(Vector3.Cross(v, Vector3.UnitY));
            return Vector3.Normalize(Vector3.Cross(v, Vector3.UnitX));
        }
    }
}
