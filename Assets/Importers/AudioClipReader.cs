using System.IO;
using Vortice.Multimedia;
using Vortice.XAudio2;

namespace Freefall.Assets
{
    [AssetReader(".wav")]
    public class AudioClipReader : AssetReader<AudioClip>
    {
        public override AudioClip Import(string filepath)
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
