using Freefall.Base;

namespace Freefall.Scripts
{
    /// <summary>
    /// Player marker component with static Instance for global access.
    /// </summary>
    public class Player : Component
    {
        public static Player Instance;

        protected override void Awake()
        {
            Instance = this;
        }
    }
}
