using System.Runtime.InteropServices;
using Vortice.Multimedia;
using Vortice.XAudio2;

namespace Freefall.Assets
{
    public class AudioClip : Asset
    {
        public AudioBuffer Buffer { get; set; }
        public WaveFormat WaveFormat { get; set; }
        public uint[] DecodedPacketsInfo { get; set; }
        internal GCHandle _pinnedHandle;

        /// <summary>
        /// The compressed file (an .ogg) of a clip imported with Stream on: decoded a little at a time while
        /// it plays (<see cref="Freefall.Audio.AudioStream"/>), never whole. Null for a clip held decoded,
        /// which has a Buffer instead.
        /// </summary>
        public byte[] StreamData { get; set; }

        public bool IsStreamed => StreamData != null;

        /// <summary>How long it plays, in seconds.</summary>
        public float Length { get; set; }
    }
}
