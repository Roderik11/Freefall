using Squid;
using Freefall.Graphics;
using Freefall.Reflection;

namespace Freefall.Editor
{
    /// <summary>
    /// Inspects a loaded Material: shows effect name, texture slots, and reflected properties.
    /// </summary>
    [GUIInspector(typeof(Material))]
    public class MaterialInspector : GUIInspector
    {
        public MaterialInspector(GUIObject target) : base(target)
        {
            var mat = target.Target as Material;

            AddCategory("Material");
            AddProperty(target.GetProperty(nameof(Material.Effect)));

            // Texture slots — bound to live TextureEffectParameter objects on the material
            AddCategory("Textures");

            if (mat != null)
            {
                var texValueProp = typeof(TextureEffectParameter).GetProperty("Value");
                var texField = new Field(texValueProp);

                foreach (var texParam in mat.TextureParameters)
                {
                    var property = new GUIProperty(target, texField, texParam);
                    AddProperty(property, texParam.Name);
                }
            }

            // Material scalar properties (emissive color, intensity, detail tiling, etc.)
            if (mat != null && mat.MaterialProperties.Count > 0)
            {
                AddCategory("Properties");
                foreach (var prop in mat.MaterialProperties)
                {
                    var valueField = new Field(prop.GetType().GetProperty("Value"));
                    var property = new GUIProperty(target, valueField, prop);
                    AddProperty(property, prop.Name);
                }
            }

            // Constant buffer parameters discovered from shader reflection
            if (mat != null)
            {
                // Skip global/system cbuffers — only show material-relevant ones
                var skipBuffers = new HashSet<string> { "SceneConstants", "PushConstants" };

                foreach (var cb in mat.ConstantBuffers)
                {
                    if (skipBuffers.Contains(cb.Name)) continue;
                    if (cb.Parameters.Count == 0) continue;

                    AddCategory(cb.Name);
                    foreach (var param in cb.Parameters)
                    {
                        var valueField = new Field(param.GetType().GetProperty("Value"));
                        var property = new GUIProperty(target, valueField, param);
                        AddProperty(property, param.Name);
                    }
                }
            }
        }
    }
}
