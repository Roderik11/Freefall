using Squid;
using Freefall.Base;
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
        private readonly Material? _material;

        public MaterialInspector(GUIObject target) : base(target)
        {
            var mat = target.Target as Material;
            _material = mat;

            AddCategory("Material");

            // The slots and settings below are the effect's: list them again when it changes. With the material as
            // what to show: an asset is not in the selection, and a refresh without it leaves the inspector empty.
            // Both ways a swap is seen (this field, Material.EffectChanged) refresh only once.
            var effectProperty = target.GetProperty(nameof(Material.Effect));
            effectProperty.OnValueChanged += Refresh;
            AddProperty(effectProperty);

            // ...and when the effect recompiles (a saved .fx can add, drop or retype settings)
            if (mat != null)
                Material.EffectChanged += OnEffectChanged;

            // A project effect that does not compile: its materials draw nothing until it does
            if (mat?.Effect?.LastError is { } error)
            {
                AddCategory("Effect does not compile");
                AddControl(new Label
                {
                    Text = error.Trim(),
                    TextWrap = true,
                    AutoSize = AutoSize.Vertical,
                    Dock = DockStyle.Top,
                    Padding = new Margin(8, 4, 8, 4),
                });
            }

            // Texture slots, bound to live TextureEffectParameter objects on the material: the MaterialData slots
            // (Albedo, Normal, ...), or for an effect with its own settings its texture settings (TextureRef)
            if (mat != null && mat.TextureParameters.Count > 0)
            {
                AddCategory("Textures");
                var texField = new Field(typeof(TextureEffectParameter).GetProperty("Value"));
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
                    var property = new GUIProperty(target, valueField, prop) { DescriptionOverride = prop.Description };
                    // A ranged number is a slider: scrub it and watch the result
                    if (prop.Range is { } range)
                        property.RangeOverride = new ValueRangeAttribute(range.X, range.Y, (range.Y - range.X) / 100f);
                    AddProperty(property, prop.Name);
                }
            }

            // Constant buffer parameters discovered from shader reflection
            if (mat != null)
            {
                // Skip global/system cbuffers — only show material-relevant ones. (An effect's own settings, "//="
                // in its struct Settings, are not a cbuffer: they are the properties above, saved with the material.)
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

        private void OnEffectChanged(Material material)
        {
            // A static event: an inspector no longer shown lets go of it the next time it fires
            if (Desktop == null)
            {
                Material.EffectChanged -= OnEffectChanged;
                return;
            }
            if (material == _material)
                Refresh();
        }

        private bool _refreshed;

        private void Refresh()
        {
            if (_refreshed) return;
            _refreshed = true;
            Material.EffectChanged -= OnEffectChanged;      // the rebuilt inspector subscribes again
            MessageDispatcher.Send(Msg.RefreshInspector, _material);
        }
    }
}
