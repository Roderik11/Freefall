using System.IO;
using Vortice.Multimedia;
using Vortice.XAudio2;

namespace Freefall.Assets
{
    public class AudioClipReader
    {
        public AudioClip Load(string filepath)
        {
            var audioClip = new AudioClip();
            var stream = new SoundStream(File.OpenRead(filepath));

            audioClip.DecodedPacketsInfo = stream.DecodedPacketsInfo;
            audioClip.WaveFormat = stream.Format;
            audioClip.Buffer = new AudioBuffer(stream);

            stream.Close();

            return audioClip;
        }
    }
}
