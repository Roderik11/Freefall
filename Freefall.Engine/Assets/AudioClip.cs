using Vortice.Multimedia;
using Vortice.XAudio2;

namespace Freefall.Assets
{
    public class AudioClip : Asset
    {
        public AudioBuffer Buffer { get; set; }
        public WaveFormat WaveFormat { get; set; }
        public uint[] DecodedPacketsInfo { get; set; }
    }
}
