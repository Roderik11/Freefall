using Squid;
using Freefall.Assets;
using Freefall.Assets.Importers;
using Freefall.Graphics;
using Freefall.Reflection;

namespace Freefall.Editor
{
    /// <summary>
    /// Inspects a loaded Mesh: shows vertex/index counts, bounds, LODs, parts, and 3D preview.
    /// </summary>
    [GUIInspector(typeof(Mesh))]
    public class MeshInspector : GUIInspector
    {
        private static MeshPreview _preview;
        private static InspectorControl _inspectorControl;

        private Mesh _mesh;

        public MeshInspector(GUIObject target) : base(target)
        {
            _mesh = target.Target as Mesh;

            if (_mesh == null) return;

            AddCategory("Mesh Information");

            AddText("Name", _mesh.Name ?? "(unnamed)");
            AddText("Guid", _mesh.Guid.ToString());
            AddText("Vertices", _mesh.VertexCount.ToString());
            AddText("Indices", _mesh.IndexCount.ToString());

            if (_mesh.Bones?.Length > 0)
                AddText("Bones", _mesh.Bones.Length.ToString());

            if (_mesh.LODs.Count > 0)
            {
                AddText("LOD Levels", _mesh.LODs.Count.ToString());
                AddText("LOD Triangles", string.Join(" / ", _mesh.LODs.Select(CountTriangles)));
            }

            var bounds = _mesh.BoundingBox;
            AddText("Min", $"{bounds.Min.X:F2}, {bounds.Min.Y:F2}, {bounds.Min.Z:F2}");
            AddText("Max", $"{bounds.Max.X:F2}, {bounds.Max.Y:F2}, {bounds.Max.Z:F2}");

            var extents = bounds.Max - bounds.Min;
            AddText("Size", $"{extents.X:F2} x {extents.Y:F2} x {extents.Z:F2}");

            // Mesh Parts with enable toggles
            var meshObj = new GUIObject(_mesh);
            var partsProperty = meshObj.GetProperty("MeshParts");
            int count = partsProperty?.GetArrayLength() ?? 0;

            AddCategory($"{count} Mesh Parts");

            for (int i = 0; i < count; i++)
            {
                var element = partsProperty.GetArrayElementAtIndex(i);
                var part = element.GetValue();
                var partObj = new GUIObject(part);
                var enabled = partObj.GetProperty("Enabled");
                AddProperty(enabled, _mesh.MeshParts[i].Name);
            }

            // LOD bias
            AddCategory("Settings");
            var lodBias = target.GetProperty(nameof(Mesh.LODBias));
            AddProperty(lodBias);
            lodBias.OnValueChanged += SaveMeshConfig;

            // LODs generated at import: settings of the whole model file, applied by reimporting it
            if (!string.IsNullOrEmpty(_mesh.Guid) && AssetDatabase.GetImporter(_mesh.Guid) is ModelImporter importer)
            {
                AddCategory("Generated LODs (model import)");
                var importerObj = new GUIObject(importer);
                foreach (var member in ModelImporter.LODGenerationMembers)
                {
                    var property = importerObj.GetProperty(member);
                    if (property != null) AddProperty(property);
                }

                var applyButton = new Button
                {
                    Text = "Apply & Reimport",
                    Dock = DockStyle.Top,
                    Size = new Point(100, 30),
                    Margin = new Margin(4, 4, 4, 4),
                    Style = "button",
                };
                var guid = _mesh.Guid;
                applyButton.MouseClick += (s, e) =>
                {
                    if (e.Button > 0) return;
                    AssetDatabase.SaveImporterAndReimport(guid, importer);
                };
                AddControl(applyButton);
            }

            if (_preview == null)
                _preview = new MeshPreview();
        }

        private long CountTriangles(MeshLOD lod)
        {
            long indices = 0;
            if (lod.MeshPartIndices != null)
                foreach (var idx in lod.MeshPartIndices)
                    if (idx < _mesh.MeshParts.Count)
                        indices += _mesh.MeshParts[idx].NumIndices;
            return indices / 3;
        }

        /// <summary>
        /// Persist per-mesh settings on the source file's ModelImporter (applied at load, no reimport).
        /// </summary>
        private void SaveMeshConfig()
        {
            if (string.IsNullOrEmpty(_mesh.Guid)) return;
            if (AssetDatabase.GetImporter(_mesh.Guid) is not ModelImporter importer) return;

            var meshName = AssetDatabase.ResolveFriendlyName(_mesh.Guid);
            importer.GetOrAddMeshConfig(meshName).LODBias = _mesh.LODBias;
            AssetDatabase.SaveImporter(_mesh.Guid, importer);
        }

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

            if (_inspectorControl != null && _mesh != null)
                _preview.Bind(_mesh, _inspectorControl);

            return _preview;
        }
    }
}
