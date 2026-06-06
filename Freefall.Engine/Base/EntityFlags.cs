using System;

namespace Freefall.Base
{
    [Flags]
    public enum EntityFlags
    {
        None = 0,
        DontDestroy = 1,
        DontSave = 2,
        HideInHierarchy = 4,
        HideAndDontSave = DontSave | HideInHierarchy
    }
}
