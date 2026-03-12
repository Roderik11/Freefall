using System;

namespace Freefall.Base
{
    [Flags]
    public enum EntityFlags
    {
        None = 0,
        DontDestroy = 1,
        HideAndDontSave = 2
    }
}
