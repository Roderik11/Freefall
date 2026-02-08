using System;
using System.Collections.Generic;
using Freefall.Graphics;

namespace Freefall.Base
{
    public static class EntityManager
    {
        private static readonly List<Entity> _entities = new List<Entity>();
        private static readonly List<Entity> _pendingAdditions = new List<Entity>();
        private static readonly object _lock = new object();

        public static void AddEntity(Entity entity)
        {
            lock (_lock)
            {
                if (!_entities.Contains(entity) && !_pendingAdditions.Contains(entity))
                    _pendingAdditions.Add(entity);
            }
        }

        public static void RemoveEntity(Entity entity)
        {
            lock (_lock)
            {
                _entities.Remove(entity);
                _pendingAdditions.Remove(entity);
            }
        }

        /// <summary>
        /// Flush pending entity additions to the main list.
        /// Call this at a safe point between frames (before Update/Render).
        /// </summary>
        public static void FlushPending()
        {
            lock (_lock)
            {
                if (_pendingAdditions.Count > 0)
                {
                    _entities.AddRange(_pendingAdditions);
                    _pendingAdditions.Clear();
                }
            }
        }

        public static void Update()
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            ScriptExecution.Update();
            sw.Stop();
            
            if (Engine.FrameIndex % 60 == 0) // Log every 60 frames
            {
                Debug.Log($"[EntityManager.Update] {sw.Elapsed.TotalMilliseconds:F2}ms");
            }
        }

        public static IEnumerable<T> FindComponents<T>() where T : Component
        {
            // Take snapshot under lock to avoid modification during iteration
            List<Entity> snapshot;
            lock (_lock)
            {
                snapshot = new List<Entity>(_entities);
            }
            
            foreach (var entity in snapshot)
            {
                var component = entity.GetComponent<T>();
                if (component != null)
                {
                    yield return component;
                }
            }
        }
    }
}
