using Vortice.XAudio2;
using System.Numerics;
using Freefall.Base;

namespace Freefall.Components
{
    public class AudioListener : Component, IUpdate
    {
        public Listener Listener { get; private set; }

        public static AudioListener Main { get; private set; }

        public AudioListener()
        {
            Main = this;
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
            var newpos = Transform.Position;
            Listener.Velocity = Time.SmoothDelta > 0 ? (newpos - Listener.Position) / Time.SmoothDelta : Vector3.Zero;
            Listener.Position = newpos;
            Listener.OrientFront = Transform.Forward;
            Listener.OrientTop = Transform.Up;
        }

        public override void Destroy()
        {
            if (Main == this)
                Main = null;

            base.Destroy();
        }
    }
}
