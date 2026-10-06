using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using System;
using System.ComponentModel;
using System.Numerics;
using System.Reflection;

namespace Freefall.Editor
{
    /// <summary>
    /// Terrain painting tool. Sends mouse ray + brush params to the GPU.
    /// The GPU does the heightmap raycast and painting — no CPU heightfield needed.
    /// Activated via F4 in EditorTools.
    /// </summary>

    public enum BrushMode : uint { Raise = 0, Lower = 1, Flatten = 2, Smooth = 3 }

    /// <summary>
    /// Terrain editing mode — controls which ControlMap target the brush paints.
    /// </summary>
    public enum TerrainEditMode { Sculpt, Paint, Foliage }


    public class TerrainBrush : ToolBase
    {
        public static TerrainBrush Instance;
        public static TerrainEditMode EditMode { get; set; } = TerrainEditMode.Sculpt;

        [ValueRange(1, 100)]
        public float BrushSize { get; set; } = 30f;

        [ValueRange(0.1f, 1)]
        public float BrushStrength { get; set; } = 0.1f;

        [ValueRange(0, 2)]
        public float BrushFalloff { get; set; } = 1.0f;

       // [Browsable(false)]
        public BrushMode BrushMode { get; set; } = BrushMode.Raise;

        public static int SelectedLayerIndex { get; private set; } = 0;

        // Cached terrain reference — updated via SelectionChanged or ComponentCache
        private TerrainRenderer _terrainRenderer;

        public override void Initialize()
        {
            Instance = this;
            MessageDispatcher.AddListener(Msg.SelectionChanged, OnSelectionChanged);
            MessageDispatcher.AddListener(Msg.SelectLayer, OnSelectLayer);
        }

        private void OnSelectionChanged(Message msg)
        {
            if (msg.Data is Entity entity)
            {
                var tr = entity.GetComponent<TerrainRenderer>();
                if (tr?.Terrain != null)
                    _terrainRenderer = tr;
            }
        }

        private void OnSelectLayer(Message msg)
        {
            if (_terrainRenderer?.Terrain == null) return;

            switch (msg.Data)
            {
                case Terrain.TextureLayer splatlayer:
                    SelectedLayerIndex = _terrainRenderer.Terrain.Layers.IndexOf(splatlayer);
                    break;
                case HeightLayer heightlayer:
                    SelectedLayerIndex = _terrainRenderer.Terrain.HeightLayers.IndexOf(heightlayer);
                    break;
                case Terrain.Decoration decoration:
                    SelectedLayerIndex = _terrainRenderer.Terrain.Decorations.IndexOf(decoration);
                    break;
            }
        }

        /// <summary>Maps the current EditMode to a ControlMapTarget for the baker.</summary>
        private TerrainBaker.ControlMapTarget GetControlMapTarget()
        {
            return EditMode switch
            {
                TerrainEditMode.Sculpt => TerrainBaker.ControlMapTarget.Height,
                TerrainEditMode.Paint => TerrainBaker.ControlMapTarget.Splatmap,
                TerrainEditMode.Foliage => TerrainBaker.ControlMapTarget.Density,
                _ => TerrainBaker.ControlMapTarget.Height,
            };
        }

        public override void Update(Camera camera)
        {
            MouseCaptured = false;

            if (camera == null) return;

            // Auto-discover terrain if none cached
            if (_terrainRenderer?.Terrain == null)
            {
                var all = ComponentCache<TerrainRenderer>.All;
                for (int i = 0; i < all.Count; i++)
                {
                    if (all[i]?.Terrain != null)
                    {
                        _terrainRenderer = all[i];
                        break;
                    }
                }
            }
            if (_terrainRenderer?.Terrain == null) return;

            // Don't interfere with UI or camera orbit
            if (EditorUI.MouseCaptured) return;
            if (Input.IsMouseDown(1)) return;

            // Handle scroll wheel: Ctrl+Scroll = size, Shift+Scroll = strength
            int wheel = Input.MouseWheelDelta;
            if (wheel != 0)
            {
                if (Input.IsKeyDown(Keys.ControlKey))
                    BrushSize = Math.Clamp(BrushSize + wheel * 0.1f, 1f, 500f);
                else if (Input.Shift)
                    BrushStrength = Math.Clamp(BrushStrength + wheel * 0.001f, 0.01f, 1f);
            }

            // Left-click painting — send ray to GPU, it does the raycast + paint
            if (Input.IsMouseDown(0))
            {
                MouseCaptured = true;

                var ray = camera.MouseRay();
                var target = GetControlMapTarget();
                var mode = BrushMode;

                if (EditMode == TerrainEditMode.Paint || EditMode == TerrainEditMode.Foliage)
                    mode = Input.Shift ? BrushMode.Lower : BrushMode.Raise;
                else if(mode == BrushMode.Raise)
                    mode = Input.Shift ? BrushMode.Lower : BrushMode.Raise;

                _terrainRenderer.Terrain?.MarkDirty();
                var strength = EditMode == TerrainEditMode.Sculpt ? BrushStrength / 100f : BrushStrength;
                _terrainRenderer.EnqueueBrushRaycastAndPaint(
                    ray.Position, ray.Direction,
                    (uint)mode, strength, BrushSize, BrushFalloff,
                    0, target, SelectedLayerIndex);
            }
        }

        public override void Render() { }

        public override void Disable()
        {
            MouseCaptured = false;
            IsClicked = false;
        }
    }
}
