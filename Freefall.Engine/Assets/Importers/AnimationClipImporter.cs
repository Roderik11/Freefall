using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;
using System.Xml;
using Freefall.Animation;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports animation clips from DAE-anim files.
    /// Uses custom XML parsing like Apex (Assimp doesn't support .dae-anim extension).
    /// </summary>
    [AssetImporter(".dae-anim")]
    public class AnimationClipImporter : AssetImporter<AnimationClip>
    {
        // MATCH APEX: DAE animations are in cm, need 0.01 scale to match mesh
        public float Scale = 0.01f;

        public override AnimationClip Load(string filename)
        {
            float scaleFactor = Scale;
            if (scaleFactor <= 0) scaleFactor = 1;

            // Namespaces are handled wrongly by XPath 1.0 and also we don't need
            // them anyway, so all namespaces are simply removed
            string xmlWithoutNamespaces = Regex.Replace(File.ReadAllText(filename), @"xmlns="".+?""", "");

            var doc = new XmlDocument();
            doc.LoadXml(xmlWithoutNamespaces);

            var root = doc.DocumentElement;

            // Import animations from library_animations
            XmlNodeList xmlAnimations = root.SelectNodes("/COLLADA/library_animations/animation");
            AnimationClip result = new AnimationClip();
            float duration = 0;

            foreach (XmlNode xmlAnim in xmlAnimations)
            {
                XmlNodeList xmlChannels = xmlAnim.SelectNodes(".//channel");

                foreach (XmlNode xmlChannel in xmlChannels)
                {
                    // Target joint
                    string target = xmlChannel.Attributes["target"].Value;
                    string jointId = ExtractNodeIdFromTarget(target);
                    string targetType = target.Substring(target.IndexOf('/') + 1);

                    // Sampler
                    string samplerId = xmlChannel.Attributes["source"].Value.Substring(1);
                    XmlNode xmlSampler = xmlAnim.SelectSingleNode("//sampler[@id='" + samplerId + "']");

                    // Input and Output sources
                    ColladaSource input = ColladaSource.FromInput(xmlSampler.SelectSingleNode("input[@semantic='INPUT']"), xmlAnim);
                    ColladaSource output = ColladaSource.FromInput(xmlSampler.SelectSingleNode("input[@semantic='OUTPUT']"), xmlAnim);

                    // It is assumed that TIME is used for input
                    var times = input.GetData<float>();
                    var channel = new AnimationChannel { Target = jointId };
                    duration = Math.Max(duration, times[times.Count - 1]);

                    if (targetType == "matrix")
                    {
                        // Baked matrices were used
                        var transforms = output.GetData<Matrix4x4>();

                        var positions = new List<VectorKey>();
                        var scales = new List<VectorKey>();
                        var rotations = new List<QuaternionKey>();

                        for (int i = 0; i < times.Count; i++)
                        {
                            // Match Apex: no transpose, apply coordinate conversion
                            var transform = transforms[i];
                            Matrix4x4.Decompose(transform, out var scale, out var rotate, out var translate);
                            
                            // Coordinate conversion: mesh uses MakeLeftHanded (negates Z)
                            // Animation file is still right-handed, so we need to convert
                            translate.Z *= -1;
                            rotate.X *= -1;
                            rotate.Y *= -1;

                            scales.Add(new VectorKey { Time = times[i], Value = scale });
                            positions.Add(new VectorKey { Time = times[i], Value = translate * scaleFactor });
                            rotations.Add(new QuaternionKey { Time = times[i], Value = rotate });
                        }

                        channel.Position = new VectorKeys(positions);
                        channel.Scale = new VectorKeys(scales);
                        channel.Rotation = new QuaternionKeys(rotations);

                        result.AddChannel(channel);
                    }
                }
            }

            result.Duration = duration;
            result.TicksPerSecond = 1.0f; // COLLADA uses time in seconds
            result.Name = Path.GetFileNameWithoutExtension(filename);

            Debug.Log("AnimationClipImporter", $"Loaded '{result.Name}' - Duration: {result.Duration:F2}s, Channels: {result.Channels.Count}");
            // Debug: Print first few channel names
            for (int i = 0; i < Math.Min(10, result.Channels.Count); i++)
            {
                var ch = result.Channels[i];
                Debug.Log($"  Channel [{i}]: {ch.Target}");
            }
            return result;
        }

        string ExtractNodeIdFromTarget(string target)
        {
            return target.Substring(0, target.IndexOf('/'));
        }
    }

    /// <summary>
    /// Helper class to parse Collada source elements.
    /// </summary>
    internal class ColladaSource
    {
        public Type DataType { get; private set; }
        private object _data;

        public List<T> GetData<T>()
        {
            return _data as List<T>;
        }

        public static ColladaSource FromInput(XmlNode inputNode, XmlNode animNode)
        {
            var source = new ColladaSource();
            string sourceId = inputNode.Attributes["source"].Value.Substring(1);
            XmlNode sourceNode = animNode.SelectSingleNode("//source[@id='" + sourceId + "']");

            // Check if it has float_array
            XmlNode floatArray = sourceNode.SelectSingleNode("float_array");
            if (floatArray != null)
            {
                // DAE files use spaces AND newlines as separators
                var values = floatArray.InnerText.Trim()
                    .Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(s => float.Parse(s, System.Globalization.CultureInfo.InvariantCulture))
                    .ToList();

                // Check accessor to determine stride
                XmlNode accessor = sourceNode.SelectSingleNode("technique_common/accessor");
                int stride = int.Parse(accessor.Attributes["stride"]?.Value ?? "1");

                if (stride == 1)
                {
                    source.DataType = typeof(float);
                    source._data = values;
                }
                else if (stride == 3)
                {
                    source.DataType = typeof(Vector3);
                    var vectors = new List<Vector3>();
                    for (int i = 0; i < values.Count; i += 3)
                    {
                        vectors.Add(new Vector3(values[i], values[i + 1], values[i + 2]));
                    }
                    source._data = vectors;
                }
                else if (stride == 16)
                {
                    source.DataType = typeof(Matrix4x4);
                    var matrices = new List<Matrix4x4>();
                    for (int i = 0; i < values.Count; i += 16)
                    {
                        // Read Row-Major from DAE, then Transpose to match Apex Pipeline
                        // This produces M_inv (Inverse Rotation, Correct Translation in Row 4)
                        var mat = new Matrix4x4(
                            values[i], values[i + 1], values[i + 2], values[i + 3],
                            values[i + 4], values[i + 5], values[i + 6], values[i + 7],
                            values[i + 8], values[i + 9], values[i + 10], values[i + 11],
                            values[i + 12], values[i + 13], values[i + 14], values[i + 15]
                        );
                        matrices.Add(Matrix4x4.Transpose(mat));
                    }
                    source._data = matrices;
                }
            }

            return source;
        }
    }
}
