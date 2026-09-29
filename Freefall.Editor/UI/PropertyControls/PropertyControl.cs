using Squid;
using Freefall.Reflection;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    public class PropertyControl : Frame
    {
        protected static readonly float Interval = 0.15f;

        protected GUIProperty property;
        protected float Timer;

        public int? RowHeight;
        public bool Expandable;
        public bool FullRow = false;

        public PropertyControl(GUIProperty property)
        {
            this.property = property;
            this.Timer = Interval;
        }

        protected void NotifyChange()
        {
        }

        public virtual void OnRowSelect() { }
    }
}
