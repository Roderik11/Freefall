using System;
using System.Collections.Generic;

namespace Freefall.Base
{
    public static class MessageDispatcher
    {
        private static Dictionary<string, List<Action<Message>>> callbacks = [];

        public static void AddListener(string type, Action<Message> action)
        {
            if (!callbacks.TryGetValue(type, out var list))
            {
                list = [];
                callbacks.Add(type, list);
            }

            list.Add(action);
        }

        public static void RemoveListener(string type, Action<Message> action)
        {
            if (callbacks.TryGetValue(type, out var list))
                list.Remove(action);
        }

        public static void Send(string message)
        {
            if (!callbacks.TryGetValue(message, out var list))
                return;

            var msg = new Message { MessageType = message };

            int count = list.Count;
            while (count > 0)
            {
                count--;
                if (list.Count > count)
                    list[count].Invoke(msg);
            }
        }

        public static void Send(string message, object data)
        {
            if (!callbacks.TryGetValue(message, out var list))
                return;

            var msg = new Message { MessageType = message, Data = data };

            int count = list.Count;
            while (count > 0)
            {
                count--;
                if (list.Count > count)
                    list[count].Invoke(msg);
            }
        }
    }

    public struct Message
    {
        public string MessageType;
        public object? Data;
    }
}
