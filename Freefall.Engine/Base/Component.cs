using System;
using System.ComponentModel;
using Freefall.Components;
using Freefall.Reflection;

namespace Freefall.Base
{
    public abstract class Component : IInstanceId, IUniqueId
    {
        private bool _awake;

        public int Id { get; } = IDGenerator.GetId();

        public ulong UID { get; set; } = IDGenerator.GetUID();

        [Browsable(false)]
        public Entity? Entity { get; internal set; }

        [Browsable(false)]
        public Transform? Transform => Entity?.Transform;

        [DefaultValue(true)]
        [Browsable(false)]
        public bool Enabled { get; set; } = true;

        internal void WakeUp()
        {
            if (_awake) return;
            _awake = true;
            Awake();
        }

        protected virtual void Awake() { }
        public virtual void Destroy() { }
    }
}
