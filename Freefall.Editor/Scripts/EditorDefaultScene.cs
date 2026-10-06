using System.Collections.Generic;
using System.Numerics;
using Freefall;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Editor
{
    /// <summary>
    /// Default empty editor scene — editor camera, directional light, and skybox.
    /// Ported from Apex's EditorTestScene.
    /// </summary>
    public class EditorDefaultScene
    {
        public EditorDefaultScene()
        {
            // --- Directional Light (Sun) ---
            var lightEntity = new Entity("Sun");
            lightEntity.Transform.Rotation = Quaternion.CreateFromYawPitchRoll(0, MathHelper.PiOver4, 0);
            var light = lightEntity.AddComponent<DirectionalLight>();

            light.Color = new Color3(1, 1, 1);
            light.Intensity = 1.0f;

            EditorCamera.CreateCamera();
        }
    } 
}
