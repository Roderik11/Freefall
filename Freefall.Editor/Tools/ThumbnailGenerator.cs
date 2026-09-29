using Freefall;
using Freefall.Assets;
using Freefall.Graphics;

namespace Freefall.Editor
{
    /// <summary>
    /// Generates thumbnails for Mesh assets by rendering them with the mesh_preview shader.
    /// </summary>
    [ThumbnailGenerator(typeof(Mesh))]
    public class MeshThumbnailGenerator : IThumbnailGenerator
    {
        public bool Generate(string guid, AssetManager assets, object renderer)
        {
            var mesh = assets.LoadByGuid<Mesh>(guid);
            if (mesh == null || mesh.VertexCount == 0) return false;

            var tr = (ThumbnailRenderer)renderer;
            var pixels = tr.RenderMeshThumbnail(mesh);
            if (pixels == null) return false;

            var thumbPath = AssetDatabase.GetThumbnailPath(guid);
            if (thumbPath == null) return false;

            TextureReadback.SavePng(thumbPath, pixels, 128, 128);
            AssetDatabase.RegisterThumbnail(guid, thumbPath);
            return true;
        }
    }

    /// <summary>
    /// Generates thumbnails for Prefab assets by instantiating the hierarchy
    /// and rendering all MeshRenderers with correct relative transforms.
    /// </summary>
    [ThumbnailGenerator(typeof(Prefab))]
    public class PrefabThumbnailGenerator : IThumbnailGenerator
    {
        public bool Generate(string guid, AssetManager assets, object renderer)
        {
            var prefab = assets.LoadByGuid<Prefab>(guid);
            if (prefab == null) return false;

            var tr = (ThumbnailRenderer)renderer;
            var pixels = tr.RenderPrefabThumbnail(prefab);
            if (pixels == null) return false;

            var thumbPath = AssetDatabase.GetThumbnailPath(guid);
            if (thumbPath == null) return false;

            TextureReadback.SavePng(thumbPath, pixels, 128, 128);
            AssetDatabase.RegisterThumbnail(guid, thumbPath);
            return true;
        }
    }

    /// <summary>
    /// Generates thumbnails for Texture assets by blitting to a small RT
    /// via the texture_preview shader (handles BC-compressed formats correctly).
    /// </summary>
    [ThumbnailGenerator(typeof(Texture))]
    public class TextureThumbnailGenerator : IThumbnailGenerator
    {
        public bool Generate(string guid, AssetManager assets, object renderer)
        {
            var texture = assets.LoadByGuid<Texture>(guid);
            if (texture?.Native == null) return false;

            var tr = (ThumbnailRenderer)renderer;
            var pixels = tr.RenderTextureThumbnail(texture);
            if (pixels == null) return false;

            var thumbPath = AssetDatabase.GetThumbnailPath(guid);
            if (thumbPath == null) return false;

            TextureReadback.SavePng(thumbPath, pixels, 128, 128);
            AssetDatabase.RegisterThumbnail(guid, thumbPath);
            return true;
        }
    }
}
