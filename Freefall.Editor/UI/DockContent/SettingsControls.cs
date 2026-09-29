using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Squid;
using System.Reflection;
using System.IO;
using Freefall.Reflection;
using Freefall.Graphics;
using System.ComponentModel;

namespace Freefall.Editor
{
    public class SettingsControl : ScrollPanel
    {
        public SettingsControl()
        {
            Style = "frame";
            var obj = new GUIObject(Engine.Settings);
            var inspector = GUIInspector.GetInspector(obj, false);
            Content.Controls.Add(inspector);
        }
    }
}
