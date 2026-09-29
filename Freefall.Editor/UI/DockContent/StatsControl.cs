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
    public class StatsControl : ScrollPanel
    {
        public StatsControl()
        {
            Style = "frame";
            var obj = new GUIObject(true, new DebugStats());
            var inspector =  GUIInspector.GetInspector(obj, false);
            Content.Controls.Add(inspector);
        }
    
        class DebugStats
        {
            public int BatchCount => CommandBuffer.LastBatchCount;
            public int DrawCallCount => CommandBuffer.LastDrawCallCount;
            public int VisibleCount => CommandBuffer.Culler?.LastVisibleCount ?? 0;
            public int OccludedCount => CommandBuffer.Culler?.LastHiZOccludedCount ?? 0;
            public int GrassDispatchN => Components.TerrainRenderer.LastDispatchN;
            public int GrassMaxInstances => Components.TerrainRenderer.LastMaxInstances;
            public int GrassInstanceCount => Components.TerrainRenderer.LastInstanceCount;
            public int MeshInstanceCount => Components.TerrainRenderer.LastMeshInstanceCount;
            public int MeshDrawCount => Components.TerrainRenderer.LastMeshDrawCount;
        }
    }
}
