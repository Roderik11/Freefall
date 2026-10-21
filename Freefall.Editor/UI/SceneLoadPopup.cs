using Freefall.Base;

namespace Freefall.Editor
{
    /// <summary>
    /// The modal progress popup while a scene the editor asked for loads (open, Play, Stop): the load's step i/n, what it
    /// is doing, the bar. Closes when the load ends; a load that failed leaves a message with the reason. The popup only
    /// follows the SceneLoad, which runs in its slices at the start of every frame either way.
    /// </summary>
    internal static class SceneLoadPopup
    {
        public static void Follow(SceneLoad load, string title)
        {
            var desktop = Program.EditorUI;
            if (desktop == null || load.IsDone) return;

            var popup = ModalDialogs.ShowProgress(desktop, () => load.IsDone ? null : new ModalDialogs.Progress(
                title,
                $"{SceneLoad.Describe(load.Current)}   ({(int)load.Current + 1}/{SceneLoad.StepCount})",
                load.Activity,
                load.Progress));

            load.Completed += done =>
            {
                popup.Close();
                if (done.Error != null)
                    ModalDialogs.ShowMessage(desktop, $"{title} failed", $"{SceneLoad.Describe(done.FailedIn)}: {done.Error.Message}", ("OK", null));
            };
        }
    }
}
