using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Vortice.Direct3D12;
using Vortice.Mathematics;

namespace Freefall.Editor
{
    /// <summary>
    /// Shared GPU context for rendering asset thumbnails.
    /// Owns the render target, command list, and materials.
    /// Created once per batch and passed to IThumbnailGenerator implementations.
    /// </summary>
    public class ThumbnailRenderer : IDisposable
    {
        private readonly int _size;
        private readonly RenderView _view;
        private readonly Material _meshMaterial;
        private readonly Material _textureMaterial;

        // Reusable command list (reset per render via allocator)
        private ID3D12CommandAllocator _allocator;
        private ID3D12GraphicsCommandList _cmd;

        // Thumbnail clear color — neutral dark blue-gray matching the editor
        private static readonly Color4 ClearColor = new(0.18f, 0.18f, 0.22f, 1.0f);

        // Camera constants for consistent framing
        private const float Fov = MathF.PI / 4f;
        private const float YawDeg = 25f;
        private const float PitchDeg = 15f;

        public ThumbnailRenderer(int size = 128)
        {
            _size = size;
            _view = new RenderView(size, size, Engine.Device);
            _view.Enabled = false; // Don't participate in the main render loop
            _meshMaterial = new Material(new Effect("mesh_preview"));
            _textureMaterial = new Material(new Effect("texture_preview"));

            var device = Engine.Device;
            _allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            _cmd = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, _allocator);
            _cmd.Close(); // Start closed — Reset() before each use
        }

        /// <summary>
        /// Begin a new render. Resets the command list for recording.
        /// </summary>
        private void Begin()
        {
            // Flush any pending async uploads (mesh buffers, textures) so they're
            // submitted to the copy queue before we issue the GPU-side wait.
            Graphics.StreamingManager.Instance?.Flush();

            _allocator.Reset();
            _cmd.Reset(_allocator);

            // Ensure any pending copy-queue uploads (mesh buffers, textures) are
            // finished before the direct queue starts drawing.
            Engine.Device.WaitForCopyQueue();

            _cmd.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
        }

        /// <summary>
        /// End a render. Records the readback copy into the SAME command list,
        /// then closes, submits, and waits for GPU completion, then maps and reads pixels.
        /// This avoids creating a second command allocator+list+fence per thumbnail.
        /// </summary>
        private byte[] EndAndReadback()
        {
            var device = Engine.Device;
            var nativeDevice = device.NativeDevice;
            var source = _view.BackBufferTexture.Native;

            // D3D12 requires row pitch aligned to 256 bytes
            uint rowPitch = (uint)((_size * 4 + 255) & ~255);
            ulong bufferSize = (ulong)rowPitch * (uint)_size;

            // Create readback buffer (will be disposed after mapping)
            var readbackBuffer = nativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Readback),
                HeapFlags.None,
                ResourceDescription.Buffer(bufferSize),
                ResourceStates.CopyDest);

            // FinishHeadless already transitioned to Common; promote to CopySource
            _cmd.ResourceBarrierTransition(source, ResourceStates.Common, ResourceStates.CopySource);

            var srcLoc = new TextureCopyLocation(source, 0);
            var dstLoc = new TextureCopyLocation(readbackBuffer,
                new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Vortice.DXGI.Format.R8G8B8A8_UNorm, (uint)_size, (uint)_size, 1, rowPitch)
                });
            _cmd.CopyTextureRegion(dstLoc, 0, 0, 0, srcLoc);

            // Transition back to Common for next render
            _cmd.ResourceBarrierTransition(source, ResourceStates.CopySource, ResourceStates.Common);

            _cmd.Close();
            device.SubmitAndWait(_cmd);

            if (device.IsDeviceLost)
            {
                Debug.LogWarning("ThumbnailRenderer", "GPU device lost after SubmitAndWait in EndAndReadback");
                readbackBuffer.Dispose();
                return null;
            }

            // Map and read pixels
            byte[] pixels = new byte[_size * _size * 4];
            unsafe
            {
                void* mappedPtr;
                readbackBuffer.Map(0, &mappedPtr);
                var mapped = (byte*)mappedPtr;

                for (int y = 0; y < _size; y++)
                {
                    var srcRow = mapped + y * rowPitch;
                    System.Runtime.InteropServices.Marshal.Copy((IntPtr)srcRow, pixels, y * _size * 4, _size * 4);
                }

                readbackBuffer.Unmap(0);
            }
            readbackBuffer.Dispose();
            return pixels;
        }

        /// <summary>
        /// Compute camera matrices that frame a bounding box nicely.
        /// The camera orbits at a fixed yaw+pitch around the origin,
        /// looking at the center of the normalized (origin-centered, unit-scaled) mesh.
        /// </summary>
        private void ComputeCamera(Vector3 boundsCenter, float diagonal,
            out Matrix4x4 viewMatrix, out Matrix4x4 projMatrix, out Vector3 cameraPos)
        {
            // Clamp diagonal to a minimum to prevent degenerate matrices
            diagonal = MathF.Max(diagonal, 0.001f);

            // Distance to fit the bounding sphere in the viewport
            float halfFov = Fov * 0.5f;
            float radius = diagonal * 0.5f;
            float distance = radius / MathF.Tan(halfFov) * 1.15f; // 15% margin

            // Fixed orbit direction
            var yaw = -MathHelper.ToRadians(YawDeg);
            var pitch = MathHelper.ToRadians(PitchDeg);
            var dir = new Vector3(
                MathF.Sin(yaw) * MathF.Cos(pitch),
                MathF.Sin(pitch),
                -MathF.Cos(yaw) * MathF.Cos(pitch));

            cameraPos = dir * distance;
            viewMatrix = Matrix4x4.CreateLookAtLeftHanded(cameraPos, Vector3.Zero, Vector3.UnitY);
            projMatrix = Matrix4x4.CreatePerspectiveFieldOfViewLeftHanded(
                Fov, 1f, distance * 0.01f, distance * 4f);
        }

        /// <summary>
        /// Render a single Mesh thumbnail. Returns RGBA pixels.
        /// </summary>
        public byte[] RenderMeshThumbnail(Mesh mesh)
        {
            if (mesh == null || mesh.VertexCount == 0) return null;
            if (mesh.PosBufferIndex == 0 || mesh.IndexBufferIndex == 0) return null;

            Begin();
            _view.PrepareHeadless(_cmd);

            var bounds = mesh.BoundingBox;
            var center = bounds.Center;
            var extents = bounds.Max - bounds.Min;
            float diagonal = extents.Length();

            ComputeCamera(center, diagonal, out var viewMatrix, out var projMatrix, out var cameraPos);

            // World: apply root rotation, then translate to origin
            var worldMatrix = mesh.RootRotation
                            * Matrix4x4.CreateTranslation(-center);

            SetupMeshMaterial(_cmd, viewMatrix, projMatrix, cameraPos, worldMatrix);
            DrawMeshParts(_cmd, mesh);

            _view.FinishHeadless(_cmd);
            return EndAndReadback();
        }

        /// <summary>
        /// Render a Prefab thumbnail by collecting all MeshRenderers in its hierarchy.
        /// </summary>
        public byte[] RenderPrefabThumbnail(Assets.Prefab prefab)
        {
            if (prefab?.SourceYaml == null || prefab.SourceYaml.Length == 0) return null;

            Entity root;
            try { root = prefab.Instantiate(); }
            catch { return null; }
            if (root == null) return null;

            try
            {
                // Ensure root is at origin — assembled prefabs may have residual transforms
                root.Transform.Position = Vector3.Zero;
                root.Transform.Rotation = Quaternion.Identity;
                root.Transform.Scale = Vector3.One;

                var meshEntries = new List<(Mesh mesh, Matrix4x4 worldMatrix)>();
                CollectMeshes(root, meshEntries);
                if (meshEntries.Count == 0) return null;

                // Combined bounding box in world space (including RootRotation)
                var combinedMin = new Vector3(float.MaxValue);
                var combinedMax = new Vector3(float.MinValue);

                foreach (var (mesh, world) in meshEntries)
                {
                    var corners = new Vector3[8];
                    mesh.BoundingBox.GetCorners(corners, mesh.RootRotation * world);
                    foreach (var corner in corners)
                    {
                        combinedMin = Vector3.Min(combinedMin, corner);
                        combinedMax = Vector3.Max(combinedMax, corner);
                    }
                }

                var center = (combinedMin + combinedMax) * 0.5f;
                var extents = combinedMax - combinedMin;
                float diagonal = extents.Length();

                ComputeCamera(center, diagonal, out var viewMatrix, out var projMatrix, out var cameraPos);

                Begin();
                _view.PrepareHeadless(_cmd);

                foreach (var (mesh, entityWorld) in meshEntries)
                {
                    // World: root rotation + entity transform, then translate to origin
                    var worldMatrix = mesh.RootRotation
                                    * entityWorld
                                    * Matrix4x4.CreateTranslation(-center);

                    SetupMeshMaterial(_cmd, viewMatrix, projMatrix, cameraPos, worldMatrix);
                    DrawMeshParts(_cmd, mesh);
                }

                _view.FinishHeadless(_cmd);
                return EndAndReadback();
            }
            finally
            {
                root.Destroy();
            }
        }

        /// <summary>
        /// Render a Texture thumbnail by blitting it to the RT via texture_preview shader.
        /// Handles BC-compressed formats correctly via GPU sampling.
        /// </summary>
        public byte[] RenderTextureThumbnail(Texture texture)
        {
            if (texture?.Native == null) return null;

            Begin();

            // Use PrepareHeadless for consistent setup (transitions, viewport, clear)
            // but we don't need depth for a fullscreen blit
            _view.PrepareHeadless(_cmd);

            // Bind the texture — use "Texture" to match the Effect reflection name
            _textureMaterial.SetTextureIndex("Texture", texture.BindlessIndex);
            _textureMaterial.SetParameter("ChannelMask", new Vector4(1, 1, 1, 0));
            _textureMaterial.SetParameter("ShowAlpha", 0f);
            _textureMaterial.SetParameter("MipLevel", 0f);
            _textureMaterial.Apply(_cmd, Engine.Device);

            _cmd.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);
            _cmd.DrawInstanced(3, 1, 0, 0);

            _view.FinishHeadless(_cmd);
            return EndAndReadback();
        }

        private void SetupMeshMaterial(ID3D12GraphicsCommandList cmd,
            Matrix4x4 viewMatrix, Matrix4x4 projMatrix, Vector3 cameraPos, Matrix4x4 worldMatrix)
        {
            var effect = _meshMaterial.Effect;
            effect.SetParameter("View", viewMatrix);
            effect.SetParameter("Projection", projMatrix);
            effect.SetParameter("CamPos", cameraPos);

            _meshMaterial.SetParameter("World", worldMatrix);
            _meshMaterial.SetParameter("LightDir", Vector3.Normalize(new Vector3(1, -1, 1)));
            _meshMaterial.SetParameter("LightColor", new Vector3(1f, 0.95f, 0.9f));
            _meshMaterial.SetParameter("MaterialColor", new Vector3(0.8f, 0.8f, 0.8f));
        }

        private void DrawMeshParts(ID3D12GraphicsCommandList cmd, Mesh mesh)
        {
            _meshMaterial.SetTextureIndex("PosBuffer", mesh.PosBufferIndex);
            _meshMaterial.SetTextureIndex("NormBuffer", mesh.NormBufferIndex);
            _meshMaterial.SetTextureIndex("UVBuffer", mesh.UVBufferIndex);
            _meshMaterial.SetTextureIndex("IndexBuffer", mesh.IndexBufferIndex);
            _meshMaterial.Apply(cmd, Engine.Device);

            cmd.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);

            if (mesh.LODs.Count > 0 && mesh.LODs[0].MeshPartIndices != null)
            {
                foreach (var partIdx in mesh.LODs[0].MeshPartIndices)
                {
                    if (partIdx >= mesh.MeshParts.Count) continue;
                    var part = mesh.MeshParts[partIdx];
                    if (!part.Enabled) continue;
                    _meshMaterial.SetTextureIndex("BaseIndex", (uint)part.BaseIndex);
                    _meshMaterial.Apply(cmd, Engine.Device);
                    cmd.DrawInstanced((uint)part.NumIndices, 1, 0, 0);
                }
            }
            else
            {
                foreach (var part in mesh.MeshParts)
                {
                    if (!part.Enabled) continue;
                    _meshMaterial.SetTextureIndex("BaseIndex", (uint)part.BaseIndex);
                    _meshMaterial.Apply(cmd, Engine.Device);
                    cmd.DrawInstanced((uint)part.NumIndices, 1, 0, 0);
                }
            }
        }

        private void CollectMeshes(Entity entity, List<(Mesh mesh, Matrix4x4 worldMatrix)> results)
        {
            var mr = entity.GetComponent<MeshRenderer>();
            if (mr?.Mesh != null && mr.Mesh.VertexCount > 0
                && mr.Mesh.PosBufferIndex != 0 && mr.Mesh.IndexBufferIndex != 0)
                results.Add((mr.Mesh, entity.Transform.WorldMatrix));

            for (int i = 0; i < entity.Transform.Count; i++)
            {
                var child = entity.Transform.GetChild(i);
                if (child?.Entity != null)
                    CollectMeshes(child.Entity, results);
            }
        }

        public void Dispose()
        {
            _cmd?.Dispose();
            _allocator?.Dispose();
            _view?.Dispose();
            RenderView.All.Remove(_view);
        }
    }
}
