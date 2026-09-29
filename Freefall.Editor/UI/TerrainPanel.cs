using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Freefall.Reflection;
using Squid;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using static Freefall.Editor.ExplorerControl;

namespace Freefall.Editor
{
    /// <summary>
    /// Dedicated terrain editing panel.
    /// </summary>
    public class TerrainPanel : Frame
    {
        private Frame frmTabButtons;
        private Frame frmTools;
        private Frame frmBrush;

        private GenericInspector brushInspector;
        private GenericInspector layerInspector;
        private VirtualList layerList;
        private Terrain selectedTerrain;
        private GUIObject selectedLayer;

        class TextureLayerListItem : Button
        {
            private readonly ImageControl thumbnail;
            private Label lblText;

            private TextureLayerListItem() 
            {
                NoEvents = false;
                Margin = new Margin(0, 0, 0, 1);
                
                Size = new Point(100, 64);
                Dock = DockStyle.Top;
                Style = "propertyElement";

                thumbnail = new ImageControl
                {
                    NoEvents = true,
                    Margin = new Margin(4),
                    Style = "border",
                    Size = new Point(62, 62),
                    Dock = DockStyle.Left,
                };
                Elements.Add(thumbnail);

                lblText = new Label
                {
                    NoEvents = true,
                    Margin = new Margin(4),
                    Dock = DockStyle.Fill,
                };

                Elements.Add(lblText);
            }

            public TextureLayerListItem(Terrain.TextureLayer layer) : this()
            {
                Bind(layer);
            }

            public TextureLayerListItem(Terrain.Decoration layer) : this()
            {
                Bind(layer);
            }

            public TextureLayerListItem(HeightLayer layer) : this()
            {
                Bind(layer);
            }

            public void Bind(Terrain.TextureLayer layer)
            {
                Tag = layer;

                lblText.Text = layer.Diffuse?.Name;
                thumbnail.Texture = AssetDatabase.GetThumbnail(layer.Diffuse);
            }

            public void Bind(Terrain.Decoration layer)
            {
                Tag = layer;

                string? text = layer.Mode switch
                {
                    DecoratorMode.Mesh => layer.Mesh?.Name,
                    DecoratorMode.Billboard => layer.Texture?.Name,
                    DecoratorMode.Cross => layer.Texture?.Name,
                    _ => "Unknown"
                };

                var thumb = layer.Mode switch
                {
                    DecoratorMode.Mesh => AssetDatabase.GetThumbnail(layer.Mesh),
                    DecoratorMode.Billboard => AssetDatabase.GetThumbnail(layer.Texture),
                    DecoratorMode.Cross => AssetDatabase.GetThumbnail(layer.Texture),
                    _ => null
                };

                lblText.Text = text;
                thumbnail.Texture = thumb;
            }

            public void Bind(HeightLayer layer)
            {
                Tag = layer;

                lblText.Text = layer.GetType().Name;
                thumbnail.Texture = AssetDatabase.GetThumbnail((Asset)null);
            }
        }

        public TerrainPanel()
        {
            Style = "frame";
            Dock = DockStyle.Fill;

            var topFrame = new Frame
            {
                Dock = DockStyle.Top,
                Size = new Point(100, 32),
                Margin = new Margin(8),
            };
            Controls.Add(topFrame);

            // this is where Raise/Lower/Smooth/etc buttons will go
            // the selection changes based on selected EditMode
            frmTabButtons = new Frame
            {
                Dock = DockStyle.CenterX,
                Size = new Point(100, 32),
                AutoSize = AutoSize.Horizontal,
            };
            topFrame.Controls.Add(frmTabButtons);

            CreateTabButton("Sculpt", TerrainEditMode.Sculpt);
            CreateTabButton("Paint", TerrainEditMode.Paint);
            CreateTabButton("Foliage", TerrainEditMode.Foliage);

            frmTools = new Frame
            {
                Dock = DockStyle.Top,
                Size = new Point(100, 125),
            };
            //Controls.Add(frmTools);

            frmBrush = new Frame
            {
                Dock = DockStyle.Top,
                Size = new Point(100, 125),
                AutoSize = AutoSize.Vertical,
            };
            Controls.Add(frmBrush);

            layerList = new VirtualList();
            layerList.ItemHeight = 66;
            layerList.Dock = DockStyle.Fill;
            layerList.Scrollbar.ButtonDown.Visible = false;
            layerList.Scrollbar.ButtonUp.Visible = false;
            layerList.Scrollbar.Slider.Ease = false;
            layerList.Scrollbar.Slider.MinHandleSize = 64;
            layerList.CreateItem = CreateNode;
            layerList.BindItem = BindNode;
            Controls.Add(layerList);

            // Listen for entity selection changes
            MessageDispatcher.AddListener(Msg.SelectionChanged, OnSelectionChanged);
        }

        private void BindNode(Control control, int index)
        {
            if(selectedTerrain == null)
                return;

            var node = control as TextureLayerListItem;
            switch(TerrainBrush.EditMode)
            {
                case TerrainEditMode.Sculpt:
                    node.Bind(selectedTerrain.HeightLayers[index]);
                    break;
                case TerrainEditMode.Paint:
                    node.Bind(selectedTerrain.Layers[index]);
                    break;
                case TerrainEditMode.Foliage:
                    node.Bind(selectedTerrain.Decorations[index]);
                    break;
            }

            node.Selected = index == TerrainBrush.SelectedLayerIndex;
        }

        private Control CreateNode(int index)
        {
            var node = CreateLayerItem(index) as TextureLayerListItem;
            if (node == null) return null;

            node.Selected = index == TerrainBrush.SelectedLayerIndex;

            node.MouseClick += (s, e) =>
            {
                selectedLayer?.OnValueChanged -= SelectedLayer_OnValueChanged;
                selectedLayer = new GUIObject(node.Tag);
                selectedLayer.OnValueChanged += SelectedLayer_OnValueChanged;

                layerInspector?.Parent = null;
                layerInspector = new GenericInspector(selectedLayer, true, false);
                layerInspector.Dock = DockStyle.Top;
                frmBrush.Controls.Add(layerInspector);

                MessageDispatcher.Send(Msg.SelectLayer, node.Tag);
                layerList.Refresh();
            };

            return node;
        }

        private void SelectedLayer_OnValueChanged(GUIProperty obj)
        {
            selectedTerrain.MarkDirty();
            MessageDispatcher.Send(Msg.AssetDirty, selectedTerrain);

            // Check for [DirtyFlag] attribute on the changed property
            var dirtyAttr = obj.GetAttribute<DirtyFlagAttribute>();
            if (dirtyAttr != null)
            {
                selectedTerrain.MarkForUpdate(dirtyAttr.Flags);
                return;
            }

            // No attribute — conservative default for general terrain property changes
            selectedTerrain.MarkForUpdate(TerrainDirtyFlags.HeightBake | TerrainDirtyFlags.SplatPack | TerrainDirtyFlags.AlbedoBake);
        }

        private Control CreateLayerItem(int index)
        {
            switch (TerrainBrush.EditMode)
            {
                case TerrainEditMode.Sculpt:
                    return new TextureLayerListItem(selectedTerrain.HeightLayers[index]);
                case TerrainEditMode.Paint:
                    return new TextureLayerListItem(selectedTerrain.Layers[index]);
                case TerrainEditMode.Foliage:
                    return new TextureLayerListItem(selectedTerrain.Decorations[index]);
                default:
                    throw new Exception("Invalid edit mode");
            }
        }

        private Button CreateTabButton(string text, TerrainEditMode mode)
        {
            var btn = new Button
            {
                Text = text,
                Dock = DockStyle.Left,
                Size = new Point(100, 32),
            };
            frmTabButtons.Controls.Add(btn);

            btn.MouseClick += (s, e) =>
            {
                TerrainBrush.EditMode = mode;
                SwitchLayers(mode);
            };
            return btn;
        }

        void SwitchLayers(TerrainEditMode mode)
        {
            if (selectedTerrain == null)
                return;

            //brushInspector?.Parent = null;
            //brushInspector = new GenericInspector(new GUIObject(TerrainBrush.Instance));
            //brushInspector.Dock = DockStyle.Top;
            //frmBrush.Controls.Add(brushInspector);

            switch (mode)
            {
                case TerrainEditMode.Sculpt:
                    layerList.DataSource = selectedTerrain.HeightLayers;
                    break;
                case TerrainEditMode.Paint:
                    layerList.DataSource = selectedTerrain.Layers;
                    break;
                case TerrainEditMode.Foliage:
                    layerList.DataSource = selectedTerrain.Decorations;
                    break;
            }

            layerInspector?.Parent = null;
        }


        private void OnSelectionChanged(Message msg)
        {
            Terrain newTerrain = null;

            // Check if the selected entity has a TerrainRenderer
            if (msg.Data is Entity entity)
            {
                var tr = entity.GetComponent<TerrainRenderer>();
                if (tr?.Terrain != null)
                    newTerrain = tr.Terrain;
            }

            // Also accept direct Terrain asset selection from asset browser
            if (msg.Data is Terrain terrain)
                newTerrain = terrain;

            if (newTerrain == selectedTerrain)
                return;
            
            selectedTerrain = newTerrain;
            brushInspector?.Parent = null;
            layerInspector?.Parent = null;
            layerList.Parent = null;

            if(selectedTerrain == null)
                return;

            brushInspector = new GenericInspector(new GUIObject(TerrainBrush.Instance));
            brushInspector.Dock = DockStyle.Top;
            frmBrush.Controls.Add(brushInspector);
            layerList.Parent = this;

            SwitchLayers(TerrainBrush.EditMode);
        }
    }
}
