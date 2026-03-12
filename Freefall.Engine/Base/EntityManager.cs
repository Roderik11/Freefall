using System;
using System.Collections.Generic;
using Freefall.Graphics;

namespace Freefall.Base
{
    public static class EntityManager
    {
        private static readonly List<Entity> _entities = [];
        private static readonly List<Entity> _pendingAdditions = [];
        private static readonly Lock _lock = new();

        public static IReadOnlyList<Entity> Entities => _entities;

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
        /// Remove all entities except those flagged DontDestroyOnLoad.
        /// Call before loading a new scene.
        /// </summary>
        public static void ClearScene()
        {
            lock (_lock)
            {
                for (int i = _entities.Count - 1; i >= 0; i--)
                {
                    if (!_entities[i].DontDestroy)
                        _entities[i].Destroy();
                }

                for (int i = _pendingAdditions.Count - 1; i >= 0; i--)
                {
                    if (!_pendingAdditions[i].DontDestroy)
                        _pendingAdditions[i].Destroy();
                }
            }
        }

        /// <summary>
        /// Flush pending entity additions to the main list.
        /// Call this at a safe point between frames (before Update/Render).
        /// Calls Awake on all components of newly added entities.
        /// </summary>
        public static void FlushPending()
        {
            List<Entity> flushed = null;
            
            lock (_lock)
            {
                if (_pendingAdditions.Count > 0)
                {
                    flushed = new List<Entity>(_pendingAdditions);
                    _entities.AddRange(_pendingAdditions);
                    _pendingAdditions.Clear();
                }
            }
            
            // Awake outside the lock — component initialization may create new entities
            if (flushed != null)
            {
                foreach (var entity in flushed)
                {
                    foreach (var component in entity.Components)
                        component.WakeUp();
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
               // Debug.Log($"[EntityManager.Update] {sw.Elapsed.TotalMilliseconds:F2}ms");
            }
        }

        public static T? FindComponent<T>() where T : Component
        {
            lock (_lock)
            {
                foreach (var entity in _entities)
                {
                    var component = entity.GetComponent<T>();
                    if (component != null)
                        return component;
                }
            }
            return null;
        }

        public static IEnumerable<T> FindComponents<T>() where T : Component
        {
            // Return snapshot-free results — callers should not add/remove entities during enumeration
            lock (_lock)
            {
                var results = new List<T>();
                foreach (var entity in _entities)
                {
                    var component = entity.GetComponent<T>();
                    if (component != null)
                        results.Add(component);
                }
                return results;
            }
        }
    }
}
