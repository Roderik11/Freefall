using System;
using System.Numerics;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Screen-space shadow projection using depth buffer ray marching.
    /// Based on Bend Studio's technique (Days Gone / PS5, Apache 2.0).
    /// Produces a single-channel shadow mask at screen resolution.
    /// </summary>
    public class ScreenSpaceShadows : IDisposable
    {
        private ComputeShader _shader;
        private int _kernelIndex;
        private RenderTexture2D? _output;
        private int _width, _height;
        private bool _firstFrame = true;

        private const int WaveSize = 64;

        public uint OutputSrvIndex => _output?.BindlessIndex ?? 0;

        public ScreenSpaceShadows()
        {
            _shader = new ComputeShader("screenspace_shadows.hlsl");
            _kernelIndex = _shader.FindKernel("CSScreenSpaceShadow");
        }

        /// <summary>
        /// Execute screen-space shadow generation for a directional light.
        /// Call after depth buffer is populated, before lighting pass.
        /// </summary>
        public void Execute(
            ID3D12GraphicsCommandList commandList,
            uint depthSrvIndex,
            int texWidth, int texHeight,
            Vector3 lightDirection,
            Matrix4x4 viewProjection,
            float nearPlane)
        {
            // For directional lights: float4(direction, 0) * ViewProjectionMatrix
            var lightProjection = Vector4.Transform(new Vector4(lightDirection, 0f), viewProjection);
            Trace(commandList, depthSrvIndex, texWidth, texHeight, lightProjection, nearPlane,
                new[] { 0, 0 }, new[] { texWidth, texHeight });
        }

        /// <summary>
        /// Screen-space shadows toward a point light: Bend's projection with w = 1, traced only within the screen
        /// rectangle of the light's sphere (no pixel outside it is lit by the light, so the rest of the mask is never
        /// read). False when the sphere is off screen: nothing was traced.
        /// </summary>
        /// <param name="cameraRelativePosition">Light position minus the camera position (the view has no translation).</param>
        public bool ExecutePoint(
            ID3D12GraphicsCommandList commandList,
            uint depthSrvIndex,
            int texWidth, int texHeight,
            Vector3 cameraRelativePosition, float range,
            Matrix4x4 viewProjection,
            float nearPlane)
        {
            if (!SphereBounds(cameraRelativePosition, range, viewProjection, texWidth, texHeight, out var min, out var max))
                return false;
            var lightProjection = Vector4.Transform(new Vector4(cameraRelativePosition, 1f), viewProjection);
            Trace(commandList, depthSrvIndex, texWidth, texHeight, lightProjection, nearPlane, min, max);
            return true;
        }

        // Pixel rectangle of a sphere's bounding box; the whole screen when a corner is behind the near plane
        private static bool SphereBounds(Vector3 centre, float radius, Matrix4x4 viewProjection, int width, int height,
            out int[] min, out int[] max)
        {
            min = new[] { 0, 0 };
            max = new[] { width, height };
            float x0 = float.MaxValue, y0 = float.MaxValue, x1 = float.MinValue, y1 = float.MinValue;
            for (int i = 0; i < 8; i++)
            {
                var corner = centre + new Vector3((i & 1) != 0 ? radius : -radius, (i & 2) != 0 ? radius : -radius,
                                                  (i & 4) != 0 ? radius : -radius);
                var clip = Vector4.Transform(new Vector4(corner, 1f), viewProjection);
                if (clip.W <= 1e-3f)
                    return true;
                x0 = MathF.Min(x0, clip.X / clip.W); x1 = MathF.Max(x1, clip.X / clip.W);
                y0 = MathF.Min(y0, clip.Y / clip.W); y1 = MathF.Max(y1, clip.Y / clip.W);
            }
            if (x1 < -1f || x0 > 1f || y1 < -1f || y0 > 1f)
                return false;
            // NDC y up → pixel y down; a pixel's margin for rounding
            min[0] = Math.Clamp((int)MathF.Floor((x0 * 0.5f + 0.5f) * width) - 1, 0, width);
            max[0] = Math.Clamp((int)MathF.Ceiling((x1 * 0.5f + 0.5f) * width) + 1, 0, width);
            min[1] = Math.Clamp((int)MathF.Floor((0.5f - y1 * 0.5f) * height) - 1, 0, height);
            max[1] = Math.Clamp((int)MathF.Ceiling((0.5f - y0 * 0.5f) * height) + 1, 0, height);
            return max[0] > min[0] && max[1] > min[1];
        }

        private void Trace(ID3D12GraphicsCommandList commandList, uint depthSrvIndex, int texWidth, int texHeight,
            Vector4 lightProjection, float nearPlane, int[] minBounds, int[] maxBounds)
        {
            EnsureOutputTexture(texWidth, texHeight);

            // Transition output to UAV (Common on first frame after creation)
            var fromState = _firstFrame ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
            commandList.ResourceBarrierTransition(_output!.Native, fromState, ResourceStates.UnorderedAccess);
            _firstFrame = false;

            float[] lp = { lightProjection.X, lightProjection.Y, lightProjection.Z, lightProjection.W };
            int[] viewportSize = { texWidth, texHeight };

            var dispatches = BuildDispatchList(lp, viewportSize, minBounds, maxBounds);

            // Set up compute
            commandList.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            commandList.SetComputeRootSignature(Engine.Device.GlobalRootSignature);

            _shader.SetPushConstant("DepthTex", depthSrvIndex);
            _shader.SetPushConstant("OutputUAV", _output!.UavIndex);

            // Params that stay the same across dispatches
            _shader.SetParam("LightCoordinate", new Vector4(
                dispatches.LightCoordinate[0], dispatches.LightCoordinate[1],
                dispatches.LightCoordinate[2], dispatches.LightCoordinate[3]));
            _shader.SetParam("InvDepthTextureSize", new Vector2(1f / texWidth, 1f / texHeight));
            _shader.SetParam("SurfaceThickness", Engine.Settings.SSSSurfaceThickness);
            _shader.SetParam("BilinearThreshold", Engine.Settings.SSSBilinearThreshold);
            _shader.SetParam("ShadowContrast", Engine.Settings.SSSShadowContrast);
            _shader.SetParam("FarDepthValue", 0f);   // Reverse-Z: far=0
            _shader.SetParam("NearDepthValue", 1f);   // Reverse-Z: near=1
            _shader.SetParam("TexWidth", (uint)texWidth);
            _shader.SetParam("TexHeight", (uint)texHeight);
            _shader.SetParam("NearPlane", nearPlane);

            for (int i = 0; i < dispatches.DispatchCount; i++)
            {
                ref var disp = ref dispatches.Dispatches[i];

                // Per-dispatch WaveOffset via push constants (safe across dispatches)
                _shader.SetPushConstant("WaveOffsetX",
                    BitConverter.SingleToUInt32Bits((float)disp.WaveOffsetX));
                _shader.SetPushConstant("WaveOffsetY",
                    BitConverter.SingleToUInt32Bits((float)disp.WaveOffsetY));

                _shader.Dispatch(_kernelIndex, commandList,
                    (uint)disp.WaveCountX, (uint)disp.WaveCountY, (uint)disp.WaveCountZ);

                // No barrier needed between dispatches (non-overlapping writes)
            }

            // Transition output back to SRV for lighting pass
            commandList.ResourceBarrierTransition(_output!.Native,
                ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
        }

        private void EnsureOutputTexture(int width, int height)
        {
            if (_output != null && _width == width && _height == height)
                return;

            _output?.Dispose();
            _output = new RenderTexture2D(Engine.Device, width, height, Format.R8_UNorm, false, true);
            _width = width;
            _height = height;
            _firstFrame = true;
        }

        #region Dispatch Builder (port of bend_sss_cpu.h)

        private struct DispatchData
        {
            public int WaveCountX, WaveCountY, WaveCountZ;
            public int WaveOffsetX, WaveOffsetY;
        }

        private struct DispatchList
        {
            public float[] LightCoordinate; // [4]
            public DispatchData[] Dispatches; // [8]
            public int DispatchCount;
        }

        private static DispatchList BuildDispatchList(
            float[] inLightProjection, int[] inViewportSize,
            int[] inMinRenderBounds, int[] inMaxRenderBounds)
        {
            var result = new DispatchList
            {
                LightCoordinate = new float[4],
                Dispatches = new DispatchData[8],
                DispatchCount = 0
            };

            float xy_light_w = inLightProjection[3];
            float FP_limit = 0.000002f * WaveSize;

            if (xy_light_w >= 0 && xy_light_w < FP_limit) xy_light_w = FP_limit;
            else if (xy_light_w < 0 && xy_light_w > -FP_limit) xy_light_w = -FP_limit;

            result.LightCoordinate[0] = ((inLightProjection[0] / xy_light_w) * +0.5f + 0.5f) * inViewportSize[0];
            result.LightCoordinate[1] = ((inLightProjection[1] / xy_light_w) * -0.5f + 0.5f) * inViewportSize[1];
            result.LightCoordinate[2] = inLightProjection[3] == 0 ? 0 : (inLightProjection[2] / inLightProjection[3]);
            result.LightCoordinate[3] = inLightProjection[3] > 0 ? 1 : -1;

            int light_x = (int)(result.LightCoordinate[0] + 0.5f);
            int light_y = (int)(result.LightCoordinate[1] + 0.5f);

            int bb0 = inMinRenderBounds[0] - light_x;
            int bb1 = -(inMaxRenderBounds[1] - light_y);
            int bb2 = inMaxRenderBounds[0] - light_x;
            int bb3 = -(inMinRenderBounds[1] - light_y);

            for (int q = 0; q < 4; q++)
            {
                bool vertical = q == 0 || q == 3;

                int b0 = Math.Max(0, ((q & 1) != 0 ? bb0 : -bb2)) / WaveSize;
                int b1 = Math.Max(0, ((q & 2) != 0 ? bb1 : -bb3)) / WaveSize;
                int b2 = Math.Max(0, (((q & 1) != 0 ? bb2 : -bb0) + WaveSize * (vertical ? 1 : 2) - 1)) / WaveSize;
                int b3 = Math.Max(0, (((q & 2) != 0 ? bb3 : -bb1) + WaveSize * (vertical ? 2 : 1) - 1)) / WaveSize;

                if ((b2 - b0) > 0 && (b3 - b1) > 0)
                {
                    int bias_x = (q == 2 || q == 3) ? 1 : 0;
                    int bias_y = (q == 1 || q == 3) ? 1 : 0;

                    int idx = result.DispatchCount++;
                    result.Dispatches[idx].WaveCountX = WaveSize;
                    result.Dispatches[idx].WaveCountY = b2 - b0;
                    result.Dispatches[idx].WaveCountZ = b3 - b1;
                    result.Dispatches[idx].WaveOffsetX = ((q & 1) != 0 ? b0 : -b2) + bias_x;
                    result.Dispatches[idx].WaveOffsetY = ((q & 2) != 0 ? -b3 : b1) + bias_y;

                    int axis_delta = +bb0 - bb1;
                    if (q == 1) axis_delta = +bb2 + bb1;
                    if (q == 2) axis_delta = -bb0 - bb3;
                    if (q == 3) axis_delta = -bb2 + bb3;
                    axis_delta = (axis_delta + WaveSize - 1) / WaveSize;

                    if (axis_delta > 0)
                    {
                        int idx2 = result.DispatchCount++;
                        result.Dispatches[idx2] = result.Dispatches[idx]; // copy

                        if (q == 0)
                        {
                            result.Dispatches[idx2].WaveCountZ = Math.Min(result.Dispatches[idx].WaveCountZ, axis_delta);
                            result.Dispatches[idx].WaveCountZ -= result.Dispatches[idx2].WaveCountZ;
                            result.Dispatches[idx2].WaveOffsetY = result.Dispatches[idx].WaveOffsetY + result.Dispatches[idx].WaveCountZ;
                            result.Dispatches[idx2].WaveOffsetX--;
                            result.Dispatches[idx2].WaveCountY++;
                        }
                        if (q == 1)
                        {
                            result.Dispatches[idx2].WaveCountY = Math.Min(result.Dispatches[idx].WaveCountY, axis_delta);
                            result.Dispatches[idx].WaveCountY -= result.Dispatches[idx2].WaveCountY;
                            result.Dispatches[idx2].WaveOffsetX = result.Dispatches[idx].WaveOffsetX + result.Dispatches[idx].WaveCountY;
                            result.Dispatches[idx2].WaveCountZ++;
                        }
                        if (q == 2)
                        {
                            result.Dispatches[idx2].WaveCountY = Math.Min(result.Dispatches[idx].WaveCountY, axis_delta);
                            result.Dispatches[idx].WaveCountY -= result.Dispatches[idx2].WaveCountY;
                            result.Dispatches[idx].WaveOffsetX += result.Dispatches[idx2].WaveCountY;
                            result.Dispatches[idx2].WaveCountZ++;
                            result.Dispatches[idx2].WaveOffsetY--;
                        }
                        if (q == 3)
                        {
                            result.Dispatches[idx2].WaveCountZ = Math.Min(result.Dispatches[idx].WaveCountZ, axis_delta);
                            result.Dispatches[idx].WaveCountZ -= result.Dispatches[idx2].WaveCountZ;
                            result.Dispatches[idx].WaveOffsetY += result.Dispatches[idx2].WaveCountZ;
                            result.Dispatches[idx2].WaveCountY++;
                        }

                        // Remove if too small
                        if (result.Dispatches[idx2].WaveCountY <= 0 || result.Dispatches[idx2].WaveCountZ <= 0)
                            result.Dispatches[idx2] = result.Dispatches[--result.DispatchCount];
                        if (result.Dispatches[idx].WaveCountY <= 0 || result.Dispatches[idx].WaveCountZ <= 0)
                            result.Dispatches[idx] = result.Dispatches[--result.DispatchCount];
                    }
                }
            }

            // Scale wave offsets by wave size (shader expects this)
            for (int i = 0; i < result.DispatchCount; i++)
            {
                result.Dispatches[i].WaveOffsetX *= WaveSize;
                result.Dispatches[i].WaveOffsetY *= WaveSize;
            }

            return result;
        }

        #endregion

        public void Dispose()
        {
            _output?.Dispose();
            _shader?.Dispose();
        }
    }
}
