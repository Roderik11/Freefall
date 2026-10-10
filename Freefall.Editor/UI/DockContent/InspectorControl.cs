using System;
using Squid;
using Freefall.Base;
using Freefall.Graphics;
using Freefall.Reflection;
using Freefall.Assets;
using PhysX;

namespace Freefall.Editor
{
    public class InspectorControl : Frame
    {
        private readonly Frame toolbar;
        private readonly SearchBox searchbox;
        private readonly ScrollPanel scrollpanel;
        private readonly SplitContainer split;
        private Point lastSize = new Point(200, 400);
        private Control lastPreview;

        private GUIObject currentTarget;
        private bool isTargetDirty = false;

        private bool IsImporterTarget => typeof(IImporter).IsAssignableFrom(currentTarget?.Type);

        // --- Preview viewport (ViewportControl pattern: headless RenderView + ImageControl) ---
        private RenderView _previewView;
        private ImageControl _previewImage;
        private string _previewTextureId;

        /// <summary>
        /// The shared preview RenderView. Previews set OnRender to their render callback.
        /// </summary>
        public RenderView PreviewView => _previewView;

        /// <summary>
        /// The ImageControl displaying the preview. Previews can wire mouse events to this.
        /// </summary>
        public ImageControl PreviewImage => _previewImage;

        public InspectorControl()
        {
            Size = new Point(420, 200);
            Dock = DockStyle.Right;

            split = new SplitContainer();
            split.Dock = DockStyle.Fill;
            split.RetainAspect = false;
            split.Orientation = Orientation.Vertical;
            split.SplitFrame1.Size = new Point(200, 200);
            split.SplitButton.Size = new Point(4, 4);
            split.SplitButton.Margin = new Margin(1, 0, 1, 0);
            Controls.Add(split);

            toolbar = new Frame
            {
                Style = "colorGrey170",
                Size = new Point(16, 40),
                Dock = DockStyle.Top,
                Margin = new Margin(0, 0, 0, 1)
            };
            
            searchbox = new SearchBox
            {
                Size = new Point(200, 16),
                Dock = DockStyle.Fill,
                Margin = new Margin(28, 8, 8, 8)
            };

            scrollpanel = new ScrollPanel
            {
                Style = "colorGrey170",
                Dock = DockStyle.Fill
            };

            scrollpanel.Content.Style = "frame";
            scrollpanel.VScroll.Ease = true;

            toolbar.Controls.Add(searchbox);
            split.SplitFrame1.Controls.Add(toolbar);
            split.SplitFrame1.Controls.Add(scrollpanel);
            
            split.SplitFrame2.Visible = false;
            split.SplitButton.Visible = false;
            split.SplitFrame1.Dock = DockStyle.Fill;
                
            searchbox.TextChanged += Searchbox_TextChanged;
            MessageDispatcher.AddListener(Msg.SelectionChanged, OnSelectionChanged);
            MessageDispatcher.AddListener(Msg.RefreshInspector, OnSelectionChanged);
            MessageDispatcher.AddListener(Msg.AssetReloaded, OnAssetReloaded);

            // --- Initialize preview viewport (same pattern as ViewportControl) ---
            InitPreviewViewport();
        }

        private void InitPreviewViewport()
        {
            _previewTextureId = System.IO.Path.GetRandomFileName();

            _previewImage = new ImageControl
            {
                Texture = _previewTextureId,
                Dock = DockStyle.Fill,
                NoEvents = false,
            };

            // Viewport border overlay
            _previewImage.GetElements().Add(new Frame
            {
                Size = new Point(100, 100),
                Style = "viewport",
                Dock = DockStyle.Fill
            });

            int w = 256, h = 256;
            _previewView = new RenderView(w, h, Engine.Device);
            // Rendered by RenderGui headless loop when OnRender is set by active preview

            var renderer = Gui.Renderer as SquidRenderer;
            renderer.InsertTexture(_previewTextureId, _previewView.BackBufferTexture.BindlessIndex, w, h);

            _previewView.OnResized += () =>
            {
                var rend = Gui.Renderer as SquidRenderer;
                rend.UpdateTexture(_previewTextureId, _previewView.BackBufferTexture.BindlessIndex, _previewView.Width, _previewView.Height);
                _previewImage.TextureRect = new Squid.Rectangle();
            };

            _previewImage.SizeChanged += s =>
            {
                int pw = Math.Max(64, _previewImage.Size.x);
                int ph = Math.Max(64, _previewImage.Size.y);
                _previewView.Resize(pw, ph);
            };
        }

        private void Searchbox_TextChanged(Control sender)
        {
            foreach (var control in scrollpanel.Content.Controls)
            {
                var inspector = control as GUIInspector;
                inspector.FilterBy(searchbox.Text);
            }

            PerformLayout();
        }

        void OnSelectionChanged(Message msg)
        {
            if(IsImporterTarget && isTargetDirty)
            {
                // if selection changes but we have a dirty importer
                // open a dialog to ask if we want to save the changes or discard them
            }

            currentTarget?.OnValueChanged -= Inspector_OnValueChanged;
            currentTarget = null;

            scrollpanel.Scroll(0);
            scrollpanel.Content.Controls.Clear();

            var array = Selector.Selection.Cast<object>().ToArray();
            GUIObject target = null;

            if(Selector.Selection.Count > 0)
                target = new GUIObject(array);
            else if(msg.Data != null)
                target = new GUIObject(msg.Data);

            if (target == null) return;

            target.OnValueChanged += (p) => Inspector_OnValueChanged(p);
            
            currentTarget = target;
            isTargetDirty = false;

            var inspector = GUIInspector.GetInspector(target);
            if (inspector != null)
            {
                scrollpanel.Content.Controls.Add(inspector);
                inspector.PerformLayout();
                scrollpanel.Content.PerformLayout();
                ActivatePreview(inspector);
            }

            if(IsImporterTarget)
            {
                // if an IImporter is selected,
                // we need Apply/Revert buttons to save or discard the changes
            }
        }

        /// <summary>
        /// The asset on display was hot-reloaded from its file: its lists and nested objects are new ones, and
        /// the controls still edit the old.
        /// </summary>
        void OnAssetReloaded(Message msg)
        {
            if (msg.Data != null && ReferenceEquals(currentTarget?.Target, msg.Data))
                OnSelectionChanged(msg);
        }

        private void Inspector_OnValueChanged(GUIProperty obj)
        {
            isTargetDirty = true;

            if (currentTarget?.Target is Asset asset)
            {
                asset.MarkDirty();
                MessageDispatcher.Send(Msg.AssetDirty, asset);
            }

            if (currentTarget?.Target is Terrain terrain)
            {
                // Check for [DirtyFlag] attribute on the changed property
                var dirtyAttr = obj.GetAttribute<DirtyFlagAttribute>();
                if (dirtyAttr != null)
                {
                    terrain.MarkForUpdate(dirtyAttr.Flags);
                    return;
                }

                // No attribute — conservative default for general terrain property changes
                terrain.MarkForUpdate(TerrainDirtyFlags.HeightBake | TerrainDirtyFlags.SplatPack | TerrainDirtyFlags.AlbedoBake);
            }
        }

        private void ActivatePreview(GUIInspector inspector)
        {
            var preview = inspector.GetPreview();

            bool isOpen = split.SplitFrame2.Visible;

            if (preview == null && isOpen)
                lastSize = split.SplitFrame1.Size;

            split.SplitFrame2.Controls.Clear();

            // Clear the OnRender callback when no preview is active
            _previewView.OnRender = null;

            split.SplitFrame2.Visible = preview != null;
            split.SplitButton.Visible = preview != null;
            split.SplitFrame1.Dock = preview == null ? DockStyle.Fill : DockStyle.Top;

            if (preview != null && !isOpen)
                split.SplitFrame1.Size = lastSize;

            if (lastPreview is IPreview iprev)
                iprev.OnDisable();

            lastPreview = preview;

            if (preview == null)
                return;

            // Add the preview's toolbar/controls, then the shared preview image
            preview.Dock = DockStyle.Top;
            split.SplitFrame2.Controls.Add(preview);
            split.SplitFrame2.Controls.Add(_previewImage);

            if(preview is IPreview prev)
                prev.OnEnable();
        }
    }
}
