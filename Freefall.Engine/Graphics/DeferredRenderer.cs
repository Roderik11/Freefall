using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    public class DeferredRenderer : RenderPipeline
    {
        public static DeferredRenderer Current { get; private set; } = null!;

        public RenderTexture2D Albedo;
        public RenderTexture2D Normals;
        public RenderTexture2D Data;
        public RenderTexture2D DepthGBuffer;  // Linear view-space depth (R32_Float), cleared to 0
        public DepthTexture2D Depth;
        
        public RenderTexture2D LightBuffer = null!;
        public RenderTexture2D Composite = null!;
        public DepthTextureArray2D ShadowTextureArray = null!;

        private Material matClear = null!;
        private Material matCompose = null!;
        private Material matDirectionalLight = null!;
        
        private bool _isFirstFrame = true;
        
        // Cached array to avoid per-frame allocation
        private CpuDescriptorHandle[]? _cachedGBufferRtvHandles;

        public DeferredRenderer()
        {
            Current = this;
        }

        public override void Initialize(int width, int height)
        {
            // Create render textures first
            CreateRenderTextures(width, height);

            // Create Materials
            matClear = new Material(new Effect("cleargbuffer"));
            matCompose = new Material(new Effect("composition"));
            matDirectionalLight = new Material(new Effect("light_directional"));

            // Shadow Map Array (Cascades)
            ShadowTextureArray = new DepthTextureArray2D(2048, 2048, 4);
            
            // Note: Hi-Z pyramid is created in CommandBuffer.InitializeCuller (called after renderer init)
        }

        private void CreateRenderTextures(int width, int height)
        {
            Albedo = new RenderTexture2D(Engine.Device, width, height, Format.R16G16B16A16_Float);
            Normals = new RenderTexture2D(Engine.Device, width, height, Format.R16G16B16A16_SNorm);
            Data = new RenderTexture2D(Engine.Device, width, height, Format.R8G8B8A8_UNorm);
            DepthGBuffer = new RenderTexture2D(Engine.Device, width, height, Format.R32_Float);
            
            // Depth Texture
            Depth = new DepthTexture2D(width, height, Format.D32_Float, true);

            LightBuffer = new RenderTexture2D(Engine.Device, width, height, Format.R16G16B16A16_Float);
            Composite = new RenderTexture2D(Engine.Device, width, height, Format.R8G8B8A8_UNorm);
        }

        public override void Resize(int width, int height)
        {
            Albedo?.Dispose();
            Normals?.Dispose();
            Data?.Dispose();
            DepthGBuffer?.Dispose();
            Depth?.Dispose();
            LightBuffer?.Dispose();
            Composite?.Dispose();

            CreateRenderTextures(width, height);
            _isFirstFrame = true; // Resource states reset to Common/Texture
            
            // Recreate Hi-Z pyramid for new dimensions
            CommandBuffer.Culler?.CreateHiZPyramid(width, height);
        }

        public override void Clear(Camera camera)
        {
            // Clear Render Targets logic? 
            // In DX12, we Clear via CommandList
            // We'll rely on the beginning of Render frame to clear
        }

        public override void Render(Camera camera, ID3D12GraphicsCommandList list)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            // Ensure Materials buffer is bound (creates SRV for bindless access)
            Material.BindMaterialsBuffer(list, Engine.Device);

            // 1. G-Buffer Pass (includes shadow rendering internally)
            var gbufferTime = System.Diagnostics.Stopwatch.StartNew();
            FillGBuffer(camera, list);
            gbufferTime.Stop();

            // 2. Light Pass
            var lightTime = System.Diagnostics.Stopwatch.StartNew();
            FillLightBuffer(camera, list);
            lightTime.Stop();

            // 3. Composition
            var composeTime = System.Diagnostics.Stopwatch.StartNew();
            Compose(camera, list);
            composeTime.Stop();

            // 4. Blit to Backbuffer
            var blitTime = System.Diagnostics.Stopwatch.StartNew();
            BlitToBackBuffer(camera, list);
            blitTime.Stop();
            
            _isFirstFrame = false;
            
            sw.Stop();
            
            // Log every 60 frames
            if (Engine.FrameIndex % 60 == 0)
            {
                Debug.Log($"[DeferredRenderer] Total: {sw.Elapsed.TotalMilliseconds:F2}ms | GBuffer: {gbufferTime.Elapsed.TotalMilliseconds:F2}ms | Light: {lightTime.Elapsed.TotalMilliseconds:F2}ms | Compose: {composeTime.Elapsed.TotalMilliseconds:F2}ms | Blit: {blitTime.Elapsed.TotalMilliseconds:F2}ms");
            }
        }
        
        private void Transition(ID3D12GraphicsCommandList list, ID3D12Resource res, ResourceStates before, ResourceStates after)
        {
            if (before == after) return;
            list.ResourceBarrierTransition(res, before, after);
        }

        private void FillGBuffer(Camera camera, ID3D12GraphicsCommandList list)
        {
             // First frame: textures start in Common state; subsequent frames: PixelShaderResource
             var fromState = _isFirstFrame ? ResourceStates.Common : ResourceStates.PixelShaderResource;
             var depthFromState = _isFirstFrame ? ResourceStates.Common : ResourceStates.PixelShaderResource;
             
             Transition(list, Albedo.Native, fromState, ResourceStates.RenderTarget);
             Transition(list, Normals.Native, fromState, ResourceStates.RenderTarget);
             Transition(list, Data.Native, fromState, ResourceStates.RenderTarget);
             Transition(list, DepthGBuffer.Native, fromState, ResourceStates.RenderTarget);
             Transition(list, Depth.Native, depthFromState, ResourceStates.DepthWrite);

             // Set shader parameters globally on all effects (Apex pattern)
             camera.SetShaderParams();

             // Execute all IDraw components to enqueue draw commands
             // This enqueues DrawShadows callback into ShadowMap pass AND opaque batches
             var drawTime = System.Diagnostics.Stopwatch.StartNew();
             ScriptExecution.Draw();
             drawTime.Stop();

             // Upload transforms AFTER Draw() so per-frame SetTransform() calls
             // (e.g. terrain patch pool-slot reassignment) are captured before GPU reads
             var uploadTime = System.Diagnostics.Stopwatch.StartNew();
             TransformBuffer.Instance?.Upload();
             uploadTime.Stop();

             // === GBuffer Draw ===
             // Cache GBuffer RTV handles to avoid per-frame allocation
             _cachedGBufferRtvHandles ??= new[] { Albedo.RtvHandle, Normals.RtvHandle, Data.RtvHandle, DepthGBuffer.RtvHandle };
             // Update handles in case resize created new resources
             _cachedGBufferRtvHandles[0] = Albedo.RtvHandle;
             _cachedGBufferRtvHandles[1] = Normals.RtvHandle;
             _cachedGBufferRtvHandles[2] = Data.RtvHandle;
             _cachedGBufferRtvHandles[3] = DepthGBuffer.RtvHandle;
             list.OMSetRenderTargets(_cachedGBufferRtvHandles, Depth.DsvHandle);
             
             list.ClearRenderTargetView(Albedo.RtvHandle, new Color4(0,0,0,0));
             list.ClearRenderTargetView(Normals.RtvHandle, new Color4(0,0,0,0));
             list.ClearRenderTargetView(Data.RtvHandle, new Color4(0,0,0,0));
             list.ClearRenderTargetView(DepthGBuffer.RtvHandle, new Color4(0,0,0,0));
             list.ClearDepthStencilView(Depth.DsvHandle, ClearFlags.Depth, 1.0f, 0);

             // Viewport
             var vp = new Viewport(0, 0, Albedo.Native.Description.Width, Albedo.Native.Description.Height);
             var scissor = new RectI(0, 0, (int)Albedo.Native.Description.Width, (int)Albedo.Native.Description.Height);
             
             list.RSSetViewport(vp);
             list.RSSetScissorRect(scissor);

             // 1. Opaque pass FIRST — builds InstanceBatch GPU buffers
             var opaqueTime = System.Diagnostics.Stopwatch.StartNew();
             CommandBuffer.Execute(RenderPass.Opaque, list, Engine.Device);
             opaqueTime.Stop();

             // 2. Shadow pass — uses opaque batches (now built) for CullShadow/DrawShadow
             //    Must run AFTER Opaque so InstanceBatch GPU state is valid.
             var shadowTime = System.Diagnostics.Stopwatch.StartNew();
             if (ShadowTextureArray != null)
             {
                 // Transition shadow texture to DepthWrite for rendering
                 var shadowFrom = _isFirstFrame ? ResourceStates.DepthWrite : ResourceStates.PixelShaderResource;
                 if (shadowFrom != ResourceStates.DepthWrite)
                 {
                     list.ResourceBarrierTransition(ShadowTextureArray.Native,
                         ResourceStates.PixelShaderResource, ResourceStates.DepthWrite);
                 }
                 
                 CommandBuffer.Execute(RenderPass.ShadowMap, list, Engine.Device);
                 
                 // Transition shadow texture to shader resource for light pass sampling
                 list.ResourceBarrierTransition(ShadowTextureArray.Native, 
                     ResourceStates.DepthWrite, ResourceStates.PixelShaderResource);
                 
                 // Shadow pass changes render targets and viewport for depth rendering.
                 // Restore GBuffer state before Sky pass.
                 // Note: camera.SetShaderParams() is NOT needed here because the shadow pass
                 // now uses a dedicated SceneConstants CBV instead of modifying MasterEffects.
                 list.OMSetRenderTargets(_cachedGBufferRtvHandles, Depth.DsvHandle);
                 list.RSSetViewport(vp);
                 list.RSSetScissorRect(scissor);
             }
             shadowTime.Stop();
             
             // 3. Sky pass
             var skyTime = System.Diagnostics.Stopwatch.StartNew();
             CommandBuffer.Execute(RenderPass.Sky, list, Engine.Device);
             skyTime.Stop();
             
             // Log every 60 frames
             if (Engine.FrameIndex % 60 == 0)
             {
                 Debug.Log($"[FillGBuffer] Draw(): {drawTime.Elapsed.TotalMilliseconds:F2}ms | Upload: {uploadTime.Elapsed.TotalMilliseconds:F2}ms | Shadow: {shadowTime.Elapsed.TotalMilliseconds:F2}ms | Opaque: {opaqueTime.Elapsed.TotalMilliseconds:F2}ms | Sky: {skyTime.Elapsed.TotalMilliseconds:F2}ms");
             }

             // End GBuffer
             Transition(list, Albedo.Native, ResourceStates.RenderTarget, ResourceStates.PixelShaderResource);
             Transition(list, Normals.Native, ResourceStates.RenderTarget, ResourceStates.PixelShaderResource);
             Transition(list, Data.Native, ResourceStates.RenderTarget, ResourceStates.PixelShaderResource);
             
             // DepthGBuffer: transition to NonPixelShaderResource for Hi-Z compute pass
             Transition(list, DepthGBuffer.Native, ResourceStates.RenderTarget, ResourceStates.NonPixelShaderResource);
             
             // Generate Hi-Z depth pyramid from LINEAR GBuffer depth (not hardware depth)
             // Linear depth cleared to 0 = no occluder, max() naturally keeps farthest geometry
             // Skip when frustum is frozen — frozen pyramid must match frozen VP for correct occlusion
             if (!Engine.Settings.FreezeFrustum)
                 CommandBuffer.Culler?.GenerateHiZPyramid(list, DepthGBuffer.BindlessIndex);
             
             // SDSM: Analyze depth buffer for adaptive cascade splits
             // DepthGBuffer is still in NonPixelShaderResource — perfect for compute SRV reads
             if (CommandBuffer.Culler?.SdsmReady == true)
             {
                 int depthWidth = (int)DepthGBuffer.Native.Description.Width;
                 int depthHeight = (int)DepthGBuffer.Native.Description.Height;
                 CommandBuffer.Culler.AnalyzeDepth(list, DepthGBuffer.BindlessIndex,
                     depthWidth, depthHeight, camera.NearPlane, camera.FarPlane);
             }
             
             // Transition GBuffer depth to PixelShaderResource for light pass sampling
             Transition(list, DepthGBuffer.Native, ResourceStates.NonPixelShaderResource, ResourceStates.PixelShaderResource);
             
             // Hardware depth to PixelShaderResource for light pass
             Transition(list, Depth.Native, ResourceStates.DepthWrite, ResourceStates.PixelShaderResource);
        }

        private void FillLightBuffer(Camera camera, ID3D12GraphicsCommandList list)
        {
             var fromState = _isFirstFrame ? ResourceStates.Common : ResourceStates.PixelShaderResource;
             Transition(list, LightBuffer.Native, fromState, ResourceStates.RenderTarget);
             list.OMSetRenderTargets(LightBuffer.RtvHandle, null); 
             list.ClearRenderTargetView(LightBuffer.RtvHandle, new Color4(0,0,0,0));
             list.RSSetViewport(new Viewport(0, 0, LightBuffer.Native.Description.Width, LightBuffer.Native.Description.Height));
             list.RSSetScissorRect(new RectI(0, 0, (int)LightBuffer.Native.Description.Width, (int)LightBuffer.Native.Description.Height));

             // Bind Inputs (GBuffer SRVs)
             // Need Desciptor Table logic here? Or just SetGraphicsRootDescriptorTable if we have a RootSignature for Lights.
             
             // Zero-translation CameraInverse: even though GBuffer depth was written with full View,
             // NDC = (worldPos - camPos) × R × P, so inverse(R × P) correctly gives camera-relative pos
             var cvp = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, camera.Forward, camera.Up) * camera.Projection;
             Matrix4x4.Invert(cvp, out var cameraInverse);
             foreach (var pair in Effect.MasterEffects)
             {
                 pair.Value.SetParameter("CameraInverse", cameraInverse);
             }
             
             CommandBuffer.Execute(RenderPass.Light, list, Engine.Device); // Draws DirectionalLight quads
             // TODO: Light batching removed - will be reimplemented
             // CommandBuffer.ExecuteLights(list, Engine.Device, Normals.BindlessIndex, Depth.BindlessIndex);

             Transition(list, LightBuffer.Native, ResourceStates.RenderTarget, ResourceStates.PixelShaderResource);
        }

        private void Compose(Camera camera, ID3D12GraphicsCommandList list)
        {
             var fromState = _isFirstFrame ? ResourceStates.Common : ResourceStates.PixelShaderResource;
             Transition(list, Composite.Native, fromState, ResourceStates.RenderTarget);
             list.OMSetRenderTargets(Composite.RtvHandle, null);
             list.ClearRenderTargetView(Composite.RtvHandle, new Color4(1, 0, 1, 1)); // Magenta Debug
             list.RSSetViewport(new Viewport(0, 0, Composite.Native.Description.Width, Composite.Native.Description.Height));
             list.RSSetScissorRect(new RectI(0, 0, (int)Composite.Native.Description.Width, (int)Composite.Native.Description.Height));
             
             var block = new MaterialBlock();
             block.SetTexture("AlbedoTex", Albedo);
             block.SetTexture("LightTex", LightBuffer);
             block.SetTexture("DataTex", Data);
             block.SetTexture("NormalTex", Normals);
             
             DrawFullscreenQuad(list, matCompose, block);
             
             Transition(list, Composite.Native, ResourceStates.RenderTarget, ResourceStates.PixelShaderResource);
        }
        
        private void DrawFullscreenQuad(ID3D12GraphicsCommandList list, Material material, MaterialBlock? parameters = null)
        {
            material.Apply(list, Engine.Device, parameters);
            list.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleStrip);
            list.DrawInstanced(4, 1, 0, 0);
        }

        private void BlitToBackBuffer(Camera camera, ID3D12GraphicsCommandList list)
        {
             var backBuffer = camera.Target.CurrentBackBuffer;
             
             Transition(list, Composite.Native, ResourceStates.PixelShaderResource, ResourceStates.CopySource);
             Transition(list, backBuffer, ResourceStates.RenderTarget, ResourceStates.CopyDest);
             
             list.CopyResource(backBuffer, Composite.Native);
             
             Transition(list, Composite.Native, ResourceStates.CopySource, ResourceStates.PixelShaderResource);
             Transition(list, backBuffer, ResourceStates.CopyDest, ResourceStates.RenderTarget);
        }

        public override void Dispose()
        {
            Albedo?.Dispose();
            Normals?.Dispose();
            Data?.Dispose();
            Depth?.Dispose();
            LightBuffer?.Dispose();
            Composite?.Dispose();
            ShadowTextureArray?.Dispose();
        }
    }
}
