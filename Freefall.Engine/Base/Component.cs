using System;
using System.ComponentModel;
using Freefall.Components;
using Freefall.Reflection;

namespace Freefall.Base
{
    public abstract class Component : IInstanceId, IUniqueId
    {
        private bool _awake;
        private bool _early;

        public int Id { get; } = IDGenerator.GetId();

        public ulong UID { get; set; } = IDGenerator.GetUID();

        [Browsable(false)]
        public Entity? Entity { get; internal set; }

        [Browsable(false)]
        public Transform? Transform => Entity?.Transform;

        [DefaultValue(true)]
        [Browsable(false)]
        public bool Enabled
        {
            get => _enabled;
            set
            {
                if (_enabled == value) return;
                _enabled = value;
                OnEnabledChanged();
            }
        }
        private bool _enabled = true;

        /// <summary>
        /// Called when Enabled changes. Components with GPU-resident draws use it to add or remove
        /// them, since nothing polls Enabled per frame for those.
        /// </summary>
        protected virtual void OnEnabledChanged() { }

        /// <summary>
        /// Set by Entity.Destroy / RemoveComponent just before Destroy() is called. A component that
        /// is destroyed in the frame it was added is still in its cache's wake-up list; it must not
        /// get Early()/Awake() after Destroy(), or it would set itself up again (subscribe to events,
        /// register GPU-resident draws) with nothing left to tear it down.
        /// </summary>
        [Browsable(false)]
        public bool IsDestroyed { get; internal set; }

        internal void WakeUp()
        {
            if (_awake || IsDestroyed) return;
            _awake = true;
            Awake();
        }

        internal void EarlyBird()
        {
            if (_early || IsDestroyed) return;
            _early = true;
            Early();
        }

        /// <summary>
        /// Called when the component is added to an entity, including when an already-awake
        /// component is moved to another entity (prefab hydration). Awake() only runs once, so
        /// anything tied to the entity or its Transform must be redone here.
        /// </summary>
        protected internal virtual void OnAttached() { }

        protected virtual void Early() { }

        protected virtual void Awake() { }

        public virtual void Destroy() { }

        public virtual void OnMemberChanged() { }
    }
}
