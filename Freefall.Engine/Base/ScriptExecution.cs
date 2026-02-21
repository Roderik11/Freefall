using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Freefall.Components;

namespace Freefall.Base
{
    public static class ScriptExecution
    {
        internal static List<IComponentCache> list = new List<IComponentCache>();

        internal static void Add(IComponentCache cache)
        {
            list.Add(cache);
        }

        public static void Update()
        {
            for (int i = 0; i < list.Count; i++)
            {
                list[i].Update();
            }
        }

        public static void Draw()
        {
            for (int i = 0; i < list.Count; i++)
            {
                list[i].Draw();
            }
        }
    }
}
