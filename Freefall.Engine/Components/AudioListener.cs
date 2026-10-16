using Vortice.XAudio2;
using System.Numerics;
using Freefall.Base;

namespace Freefall.Components
{
    [Icon("icon_listener.png")]
    public class AudioListener : Component, IUpdate
    {
        public Listener Listener { get; private set; }

        /// <summary>
        /// The listener every sound is heard from: the one made last, unless one was chosen with
        /// <see cref="Activate"/>, which keeps it until it is destroyed.
        /// </summary>
        public static AudioListener Main { get; private set; }

        // Main was chosen with Activate: a listener made later (a scene loaded in the editor brings
        // its game camera's) does not take over
        private static bool _activated;

        public AudioListener()
        {
            if (Main == null || !_activated)
                Main = this;
        }

        /// <summary>
        /// Makes this the listener every sound is heard from, until it is destroyed. The editor does
        /// this with its camera's, so outside play sounds are heard from the viewport; play destroys
        /// that camera, and the game's own listener takes over.
        /// </summary>
        public void Activate()
        {
            Main = this;
            _activated = true;
        }

        protected override void Awake()
        {
            Listener = new Listener
            {
                Position = Vector3.Zero,
                Velocity = Vector3.Zero,
                OrientFront = Vector3.UnitZ,
                OrientTop = Vector3.UnitY,
            };
        }

        public void Update()
        {
            var newpos = Transform.WorldPosition;
            Listener.Velocity = Time.SmoothDelta > 0 ? (newpos - Listener.Position) / Time.SmoothDelta : Vector3.Zero;
            Listener.Position = newpos;
            Listener.OrientFront = Transform.Forward;
            Listener.OrientTop = Transform.Up;
        }

        public override void Destroy()
        {
            if (Main == this)
            {
                Main = null;
                _activated = false;
            }

            base.Destroy();
        }
    }
}
