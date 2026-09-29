using Squid;
using Freefall.Graphics;
using Freefall.Assets;
using Freefall.Assets.Importers;
using Freefall.Reflection;

namespace Freefall.Editor
{
    /// <summary>
    /// Inspects a loaded Texture: shows importer settings, format, dimensions, and preview with channel toggles.
    /// Resolves the TextureImporter via the asset GUID so import settings can be edited alongside the preview.
    /// </summary>
    [GUIInspector(typeof(Texture))]
    public class TextureInspector : GUIInspector
    {
        private static TexturePreview _preview;
        private static InspectorControl _inspectorControl;

        public TextureInspector(GUIObject target) : base(target)
        {
            var tex = target.Target as Texture;

            // Resolve the importer for this texture's GUID
            TextureImporter importer = null;
            if (tex != null && !string.IsNullOrEmpty(tex.Guid))
                importer = AssetDatabase.GetImporter(tex.Guid) as TextureImporter;

            // Import settings (editable)
            if (importer != null)
            {
                AddCategory("Import Settings");
                var importerObj = new GUIObject(importer);
                foreach (var prop in importerObj.GetProperties())
                    AddProperty(prop);

                // "Apply & Reimport" button
                var applyButton = new Button
                {
                    Text = "Apply & Reimport",
                    Dock = DockStyle.Top,
                    Size = new Point(100, 30),
                    Margin = new Margin(4, 4, 4, 4),
                    Style = "button",
                };

                var capturedGuid = tex.Guid;
                var capturedImporter = importer;
                applyButton.MouseClick += (s, e) =>
                {
                    if (e.Button > 0) return;
                    capturedImporter.UserConfigured = true;
                    AssetDatabase.SaveImporterAndReimport(capturedGuid, capturedImporter);
                };

                AddControl(applyButton);
            }

            // Texture info (read-only)
            AddCategory("Texture");
            AddText("Name", tex?.Name ?? "(unnamed)");

            if (tex?.Native != null)
            {
                var desc = tex.Native.Description;
                AddText("Guid", tex.Guid.ToString());
                AddText("Width", desc.Width.ToString());
                AddText("Height", desc.Height.ToString());
                AddText("Format", desc.Format.ToString());
                AddText("Mip Levels", desc.MipLevels.ToString());
                AddText("Array Size", desc.DepthOrArraySize.ToString());
            }

            AddText("Bindless Index", tex?.BindlessIndex.ToString() ?? "-");

            if (_preview == null)
                _preview = new TexturePreview();

            _pendingTexture = tex;
        }

        private Texture _pendingTexture;

        public override Control GetPreview()
        {
            if (_inspectorControl == null)
            {
                Control parent = Parent;
                while (parent != null)
                {
                    if (parent is InspectorControl ic) { _inspectorControl = ic; break; }
                    parent = parent.Parent;
                }
            }

            if (_inspectorControl != null && _pendingTexture != null)
                _preview.Bind(_pendingTexture, _inspectorControl);

            return _preview;
        }
    }
}
