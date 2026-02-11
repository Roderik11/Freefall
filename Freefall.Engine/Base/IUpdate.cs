namespace Freefall.Base
{
    /// <summary>
    /// Marker interface for components that implement Update().
    /// </summary>
    public interface IUpdate : IInstanceId
    {
        void Update();
    }
}
