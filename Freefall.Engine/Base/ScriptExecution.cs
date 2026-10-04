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

        /// <summary>
        /// Drop the caches of component types matching the predicate (script types whose assembly is being
        /// unloaded), so the dispatch list stops referencing ComponentCache&lt;OldType&gt;.
        /// Main thread only, never from inside Update/Draw.
        /// </summary>
        internal static int RemoveCaches(Func<Type, bool> predicate)
        {
            return list.RemoveAll(c => predicate(c.ComponentType));
        }

        public static void Update()
        {
            for (int i = 0; i < list.Count; i++)
                list[i].Early();

            for (int i = 0; i < list.Count; i++)
                list[i].Awake();

            for (int i = 0; i < list.Count; i++)
                list[i].Update();
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
