using System;
using System.ComponentModel;
using Freefall.Components;

namespace Freefall.Base
{
    public abstract class Component : IInstanceId
    {
        private bool _awake;

        private static volatile int _instanceCount;
        private readonly int _instanceId = Interlocked.Increment(ref _instanceCount);
        public int GetInstanceId() => _instanceId;

        [Browsable(false)]
        public Entity? Entity { get; internal set; }

        [Browsable(false)]
        public Transform? Transform => Entity?.Transform;

        [DefaultValue(true)]
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
