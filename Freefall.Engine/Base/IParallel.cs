namespace Freefall.Base
{
    /// <summary>
    /// Marker interface for components that are safe to update in parallel.
    /// Components implementing this interface will be updated concurrently via Parallel.ForEach.
    /// Components not implementing this interface will be updated serially.
    /// </summary>
    public interface IParallel { }
}
