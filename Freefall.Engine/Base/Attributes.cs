using System;

namespace Freefall
{
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class UpdateInEditorAttribute : Attribute { }

    /// <summary>
    /// The <see cref="Base.SystemGroup"/> a system is updated by. Without it a system lands in
    /// <see cref="Base.UpdateGroup"/>.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class UpdateInGroupAttribute(Type group) : Attribute
    {
        public Type Group = group;
    }

    /// <summary>Update this system before another system of the same group.</summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class UpdateBeforeAttribute(Type system) : Attribute
    {
        public Type SystemType = system;
    }

    /// <summary>Update this system after another system of the same group.</summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class UpdateAfterAttribute(Type system) : Attribute
    {
        public Type SystemType = system;
    }

    /// <summary>
    /// Constrains a numeric field to a [min, max] range in the inspector.
    /// Ported from Apex/Spark.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class ValueRangeAttribute : Attribute
    {
        public float Min;
        public float Max;
        public float Step = 1;

        public ValueRangeAttribute(float min, float max)
        {
            Min = min;
            Max = max;
        }

        public ValueRangeAttribute(float min, float max, float step)
        {
            Min = min;
            Max = max;
            Step = step;
        }
    }

    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class FilePathAttribute(string filter, string title) : Attribute
    {
        public string Filter = filter;
        public string Title = title;
    }

    [AttributeUsage(AttributeTargets.All)]
    public sealed class IconAttribute(string name) : Attribute
    {
        public string Name = name;
    }

    /// <summary>
    /// Shows a public method without parameters as a button in its component's inspector; a click calls it
    /// on every selected component (MCP: entity_press_button). Buttons that name the same <see cref="Row"/>
    /// stand side by side, as many as fit, and wrap onto the next line; a button without one has a line of
    /// its own across the inspector. <c>[Category]</c> on the method puts it at the end of that header's
    /// fields (under a header of its own when no field has it); without one it comes after all the fields.
    /// <c>[Description]</c> on the method is its tooltip.
    /// </summary>
    /// <example>
    /// <c>[Button("Fire impulse", Icon = "icon_play.png")]</c> (icon and text),
    /// <c>[Button("", Icon = "icon_plus.png", Row = "Points")]</c> (icon only, in the row "Points").
    /// </example>
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class ButtonAttribute : Attribute
    {
        /// <summary>What the button says. Null = the method's name in words; "" with an <see cref="Icon"/> = the icon alone.</summary>
        public string Text;

        /// <summary>An image of the editor's (Freefall.Editor/Resources/Images), e.g. "icon_play.png", left of the text.</summary>
        public string Icon;

        /// <summary>Buttons naming the same row stand side by side in it, wrapping when the inspector is narrow.</summary>
        public string Row;

        public ButtonAttribute() { }

        public ButtonAttribute(string text)
        {
            Text = text;
        }
    }

    [AttributeUsage(AttributeTargets.Class)]
    public sealed class CreateAssetAttribute(string caption) : Attribute
    {
        public string Caption = caption;
    }
}
