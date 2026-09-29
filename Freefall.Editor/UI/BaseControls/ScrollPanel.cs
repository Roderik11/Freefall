using Squid;

namespace Freefall.Editor
{
    public class ScrollPanel : Panel
    {
        public ScrollPanel()
        {
            Dock = DockStyle.Fill;

            VScroll.Size = new Point(12, 16);
            VScroll.ButtonDown.Visible = false;
            VScroll.ButtonUp.Visible = false;
            VScroll.Slider.Button.Margin = new Margin(2, 4, 0, 4);
            VScroll.Slider.Ease = false;
            VScroll.Slider.MinHandleSize = 128;
            VScroll.Dock = DockStyle.Right;
            HScroll.Size = new Point(0, 0);
        }

        public void Scroll(int value)
        {
            VScroll.Value = value;
        }
    }
}
