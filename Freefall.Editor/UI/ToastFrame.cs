using System;
using System.Collections.Generic;
using System.Text;
using Freefall.Base;
using Squid;

namespace Freefall.Editor
{
    public class ToastFrame : Frame
    {
        private Queue<ToastData> toasts = new Queue<ToastData>();
        private List<ToastItem> activeToasts = new List<ToastItem>();

        public int ToastHeight = 32;

        struct ToastData
        {
            public string Text;
            public float Duration;
        }

        class ToastItem : Frame
        {
            public float Duration;

            public ToastItem(ToastData data)
            {
                Style = "tile";
                Margin = new Margin(0, 0, 0, 8);

                Duration = data.Duration;
                var label = new Label();
                label.Dock = DockStyle.Fill;
                label.Text = data.Text;
                label.Style = "header";
                Controls.Add(label);
            }
        }


        public ToastFrame()
        {
            Toast.OnShow += HandleToastOnShow;
            Padding = new Margin(8);
        }

        private void HandleToastOnShow(string text, float duration)
        {
            var toast = new ToastData
            {
                Text = text,
                Duration = duration
            };
            toasts.Enqueue(toast);
        }

        private void CreateToast(ToastData data)
        {
            var toastItem = new ToastItem(data);
            toastItem.Size = new Point(100, ToastHeight);
            toastItem.Dock = DockStyle.Top;
           
            activeToasts.Add(toastItem);
            Controls.Add(toastItem);
        }

        protected override void OnUpdate()
        {
            for (int i = 0; i < activeToasts.Count; i++)
            {
                var toast = activeToasts[i];
                toast.Duration -= Time.Delta;

                if (toast.Duration <= 0)
                {
                    if (toast.Opacity > 0)
                    {
                        toast.Opacity -= Time.Delta * 2; // Fade out over 0.5 seconds
                        break;
                    }
                    else if (toast.Size.y > 0)
                    {
                        float height = toast.Size.y;
                        height -= (Time.Delta * 2); // Fade out over 0.5 seconds
                        toast.Size = new Point(toast.Size.x, Math.Max(0, (int)height));
                        break;
                    }
                }
            }

            // Remove expired toasts
            for (int i = activeToasts.Count - 1; i >= 0; i--)
            {
                var toast = activeToasts[i];
                toast.Duration -= Time.Delta;

                if (toast.Duration > 0)
                    continue;

                if (toast.Size.y > 0)
                    continue;
             
                Controls.Remove(toast);
                activeToasts.RemoveAt(i);
            }


            // Create new toasts if there are any in the queue
            int totalToasts = activeToasts.Count;
            while(toasts.TryPeek(out var toastData))
            {
                int availableHeight = Size.y - totalToasts * ToastHeight;

                if(availableHeight > ToastHeight)
                {
                    toastData = toasts.Dequeue();
                    CreateToast(toastData);
                    totalToasts++;
                }
                else
                {
                    break;
                }
            }
        }
    }

    public static class Toast
    {
        public static event Action<string, float> OnShow;

        public static void Show(string text, float duration = 2f)
        {
            OnShow?.Invoke(text, duration);
        }
    }
}
