namespace Freefall.Base
{
    /// <summary>
    /// Marker interface for components that implement Draw().
    /// </summary>
    public interface IDraw : IInstanceId
    {
        void Draw();
    }
}
