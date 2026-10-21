using System;
using System.Collections.Generic;
using System.Reflection;
using PhysX;

namespace Freefall.Base
{
    /// <summary>
    /// Disposing PhysX.Net objects without the wrapper's cost per object. PhysX.Net's ObjectTable walks every object it
    /// knows twice for each one disposed (GetDependents over its whole ownership map, then Remove(Object) over its whole
    /// pointer map: ObjectTable.Disposal.cpp, which has a TODO about it), so with ~25,000 shapes and actors alive one
    /// Dispose took ~1.7 ms, and a play-mode scene clear on Eldermoor ~32 s.
    /// <see cref="Release"/> does what that disposal does, by key: it takes the table's OnDisposing handler off the
    /// object, disposes it, and removes it with ObjectTable.Remove(pointer), which only looks up dictionaries. The
    /// pointers come from the table's own ObjectAdded event.
    /// <para>What the slow path also did and this does not: dispose the object's dependents (an actor's shapes). Release
    /// those first (Collider and RigidBody do).</para>
    /// </summary>
    internal static class PhysXRelease
    {
        private static readonly Dictionary<object, long> _pointers = new(ReferenceEqualityComparer.Instance);
        private static EventInfo? _onDisposing;
        private static Delegate? _tableHandler;

        /// <summary>PhysicsWorld.Initialize, before the first object is made. Without the members it needs, Release is a plain Dispose.</summary>
        public static void Initialize()
        {
            try
            {
                var table = ObjectTable.Instance;
                table.ObjectAdded += (_, e) => { lock (_pointers) _pointers[e.ManagedObject] = e.UnmanagedObject; };
                table.ObjectRemoved += (_, e) => { lock (_pointers) _pointers.Remove(e.ManagedObject); };

                var handler = typeof(ObjectTable).GetMethod("disposableObject_OnDisposing", BindingFlags.Instance | BindingFlags.NonPublic);
                _onDisposing = typeof(PhysX.IDisposable).GetEvent("OnDisposing");
                if (handler != null && _onDisposing != null)
                    _tableHandler = Delegate.CreateDelegate(typeof(EventHandler), table, handler);
                else
                    Debug.LogWarning("PhysX", "ObjectTable internals not found: PhysX objects are disposed the slow way");
            }
            catch (Exception ex)
            {
                _tableHandler = null;
                Debug.LogWarning("PhysX", $"Fast release not available: {ex.Message}");
            }
        }

        /// <summary>
        /// Dispose <paramref name="obj"/> (null or disposed: nothing). With <paramref name="releaseNative"/> false the
        /// wrapper goes but the PhysX object is left to whoever releases it along (an exclusive shape whose actor is
        /// released next takes it with it): no detach. Main thread, like every PhysX.Net call that adds or removes objects.
        /// </summary>
        public static void Release(PhysX.IDisposable? obj, bool releaseNative = true)
        {
            if (obj == null || obj.Disposed) return;

            long pointer;
            bool known;
            lock (_pointers) known = _pointers.TryGetValue(obj, out pointer);
            if (_tableHandler == null || !known)
            {
                obj.Dispose();
                return;
            }

            _onDisposing!.RemoveEventHandler(obj, _tableHandler);
            if (!releaseNative && !SetUnmanagedOwner(obj, false))
                releaseNative = true;
            obj.Dispose();
            ObjectTable.Instance.Remove(pointer);
        }

        // The wrapper releases the PhysX object only while it "owns" it (Shape and Actor: an internal property)
        private static bool SetUnmanagedOwner(object obj, bool value)
        {
            var property = obj.GetType().GetProperty("UnmanagedOwner", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            if (property?.SetMethod == null) return false;
            property.SetValue(obj, value);
            return true;
        }
    }
}
