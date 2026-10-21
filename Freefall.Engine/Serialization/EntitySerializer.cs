using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Reflection;

namespace Freefall.Serialization
{
    /// <summary>
    /// Unified serializer/deserializer for entities and their components.
    /// Handles both scene files and in-memory round-trips (prefab instantiation, duplication).
    /// Three-phase loading:
    ///   1. Parse YAML with deferred asset stubs (no LoadByGuid during parse)
    ///   2. Build UID lookup tables and resolve entity/component references
    ///   3. Resolve asset stubs via LoadByGuid
    /// </summary>
    public class EntitySerializer
    {
        private readonly YAMLSerializer _yaml = new();
        public bool DuplicateMode { get; set; } = false;

        /// <summary>What the phases of the last <see cref="LoadFromBytes"/> took, in ms.</summary>
        public readonly record struct Timings(double ParseMs, double ReferencesMs, double AssetsMs, double PrefabsMs);

        public Timings LastTimings { get; private set; }

        private static readonly ProfilerMarker _parseMarker = ProfilerMarker.Get("Parse YAML");
        private static readonly ProfilerMarker _readMarker = ProfilerMarker.Get("Read YAML");          // text to objects (entities register here)
        private static readonly ProfilerMarker _attachMarker = ProfilerMarker.Get("Attach components");
        private static readonly ProfilerMarker _referencesMarker = ProfilerMarker.Get("Resolve references");
        private static readonly ProfilerMarker _assetsMarker = ProfilerMarker.Get("Resolve assets");
        private static readonly ProfilerMarker _prefabsMarker = ProfilerMarker.Get("Hydrate prefabs");

        // ── Save ──────────────────────────────────────────────────

        public void Save(string path, IEnumerable<Entity> entities)
        {
            File.WriteAllText(path, SaveToString(entities), Encoding.UTF8);
        }

        public string SaveToString(IEnumerable<Entity> entities)
        {
            var emitter = _yaml.Begin();

            foreach (var entity in entities)
            {
                if (entity.DontSave) continue;

                // Skip child entities of prefab instances — they're reconstructed on load
                if (entity.IsChildOfPrefabInstance)
                    continue;

                emitter.SetTag("---");
                _yaml.Serialize(entity, ref emitter);

                if (entity.IsPrefabInstance)
                {
                    // Prefab instance: only emit Transform (position/rotation/scale).
                    // All other components come from the prefab on load.
                    emitter.SetTag("---");
                    _yaml.Serialize(entity.Transform, ref emitter);
                }
                else
                {
                    // Inline entity: emit all components
                    foreach (var component in entity.Components)
                    {
                        emitter.SetTag("---");
                        _yaml.Serialize(component, ref emitter);
                    }
                }
            }

            return _yaml.ToString(emitter);
        }

        // ── Load ──────────────────────────────────────────────────

        public List<Entity> Load(string path)
        {
            var bytes = File.ReadAllBytes(path);
            return LoadFromBytes(bytes);
        }

        public List<Entity> LoadFromString(string yaml)
        {
            return LoadFromBytes(Encoding.UTF8.GetBytes(yaml));
        }

        public List<Entity> LoadFromBytes(byte[] bytes, bool skipPrefabHydration = false) =>
            Load(bytes, null, skipPrefabHydration);

        /// <summary>
        /// <see cref="LoadFromBytes(byte[], bool)"/>, recording the parse into <paramref name="record"/> so that the same
        /// YAML loads again with <see cref="LoadFromRecording"/> (prefabs: PCG points, kit pieces).
        /// </summary>
        public List<Entity> LoadFromBytes(byte[] bytes, YAMLSerializer.Recording record, bool skipPrefabHydration = false) =>
            Load(bytes, record, skipPrefabHydration);

        /// <summary>
        /// The entities a recorded parse loaded, made anew without reading the YAML again (fresh UIDs in DuplicateMode,
        /// the mode it was recorded in). Everything after the parse (references, assets, prefabs) runs as for YAML.
        /// </summary>
        public List<Entity> LoadFromRecording(YAMLSerializer.Recording recording, bool skipPrefabHydration = false) =>
            Load(null, recording, skipPrefabHydration);

        private List<Entity> Load(byte[]? bytes, YAMLSerializer.Recording? recording, bool skipPrefabHydration)
        {
            var job = new Job(this, bytes, recording, skipPrefabHydration);
            job.Step(long.MaxValue);
            return job.Entities;
        }

        /// <summary>
        /// Start a load that can stop between entities and carry on later (SceneLoad gives it a slice per frame): the
        /// parse and the references at once, then the asset stubs and the prefab instances entity by entity. Call
        /// <see cref="Job.Step"/> until it returns true. <see cref="LoadFromBytes(byte[], bool)"/> is one run to the end.
        /// </summary>
        public Job BeginLoad(byte[] bytes, bool skipPrefabHydration = false) => new(this, bytes, null, skipPrefabHydration);

        /// <summary>A load in phases (<see cref="BeginLoad"/>). Main thread.</summary>
        public sealed class Job
        {
            private enum Phase { Parse, Assets, Prefabs, Done }

            private readonly EntitySerializer _owner;
            private byte[]? _bytes;
            private YAMLSerializer.Recording? _recording;
            private readonly bool _skipPrefabHydration;
            private Phase _phase = Phase.Parse;
            private int _cursor;
            private int _prefabCount, _uidCount;
            private long _parseTicks, _referencesTicks, _assetsTicks, _prefabsTicks;

            /// <summary>The entities, once the parse has run (the first <see cref="Step"/>).</summary>
            public List<Entity> Entities { get; private set; } = new();

            public bool IsDone => _phase == Phase.Done;

            /// <summary>What it is doing, for a progress line.</summary>
            public string Activity => _phase switch
            {
                Phase.Parse => "Reading the scene",
                Phase.Assets => $"Loading assets ({_cursor}/{Entities.Count})",
                Phase.Prefabs => $"Placing prefabs ({_cursor}/{Entities.Count})",
                _ => "Read",
            };

            /// <summary>0..1, by phase and entity.</summary>
            public float Progress
            {
                get
                {
                    float within = Entities.Count == 0 ? 1f : (float)_cursor / Entities.Count;
                    return _phase switch
                    {
                        Phase.Parse => 0f,
                        Phase.Assets => 0.1f + 0.6f * within,
                        Phase.Prefabs => 0.7f + 0.3f * within,
                        _ => 1f,
                    };
                }
            }

            internal Job(EntitySerializer owner, byte[]? bytes, YAMLSerializer.Recording? recording, bool skipPrefabHydration)
            {
                // Strip BOM if present
                if (bytes != null && bytes.Length >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF)
                {
                    var trimmed = new byte[bytes.Length - 3];
                    Array.Copy(bytes, 3, trimmed, 0, trimmed.Length);
                    bytes = trimmed;
                }

                _owner = owner;
                _bytes = bytes;
                _recording = recording;
                _skipPrefabHydration = skipPrefabHydration;
            }

            /// <summary>Carry on until done or <paramref name="deadline"/> (a Stopwatch timestamp) has passed; checked between entities. True when done.</summary>
            public bool Step(long deadline)
            {
                if (_phase == Phase.Parse)
                {
                    Parse();
                    _phase = Phase.Assets;
                    _cursor = 0;
                    if (Engine.Assets == null) _phase = Phase.Prefabs;
                    if (Stopwatch.GetTimestamp() >= deadline) return false;
                }

                if (_phase == Phase.Assets)
                {
                    long start = Stopwatch.GetTimestamp();
                    using (_assetsMarker.Auto())
                    {
                        // ── Phase 3: Resolve deferred Asset stubs ──
                        // Now that parsing is done, resolve every Asset stub via LoadByGuid.
                        while (_cursor < Entities.Count)
                        {
                            foreach (var component in Entities[_cursor].Components)
                                ResolveObjectStubs(component, component.GetType().Name);
                            _cursor++;
                            if (Stopwatch.GetTimestamp() >= deadline) break;
                        }
                    }
                    _assetsTicks += Stopwatch.GetTimestamp() - start;
                    if (_cursor < Entities.Count) return false;
                    _phase = Phase.Prefabs;
                    _cursor = 0;
                }

                if (_phase == Phase.Prefabs)
                {
                    long start = Stopwatch.GetTimestamp();
                    using (_prefabsMarker.Auto())
                    {
                        // ── Phase 4: Hydrate prefab instances ──
                        // Entities saved as prefab references only have Entity + Transform in the scene file.
                        // ResolveAssetStubs only walks component fields — resolve Entity.Prefab explicitly.
                        // Skipped when called from Prefab.ApplyTo to prevent infinite recursion.
                        while (!_skipPrefabHydration && _cursor < Entities.Count)
                        {
                            var entity = Entities[_cursor++];
                            if (entity.Prefab != null && !string.IsNullOrEmpty(entity.Prefab.Guid))
                            {
                                // Resolve the stub → fully loaded Prefab with SourceYaml
                                var loaded = Engine.Assets?.LoadByGuid(entity.Prefab.Guid, typeof(Prefab)) as Prefab;
                                if (loaded != null)
                                {
                                    entity.Prefab = loaded;
                                    loaded.ApplyTo(entity);
                                    _prefabCount++;
                                }
                            }
                            if (Stopwatch.GetTimestamp() >= deadline) break;
                        }
                    }
                    _prefabsTicks += Stopwatch.GetTimestamp() - start;
                    if (!_skipPrefabHydration && _cursor < Entities.Count) return false;
                    Finish();
                }

                return true;
            }

            private void Parse()
            {
                var yaml = _owner._yaml;
                long start = Stopwatch.GetTimestamp();
                var parseScope = _parseMarker.Auto();

                // ── Phase 1: Parse with deferred asset loading ──
                // Asset fields get stub objects (Guid set, nothing else).
                // Entity/Component ref fields collect UIDs into DeferredUniqueIdRefs.
                // No LoadByGuid calls during parse = no parser corruption.
                yaml.DeferAssetLoading = true;
                yaml.DeferredUniqueIdRefs.Clear();
                yaml.DuplicateMode = _owner.DuplicateMode;
                yaml.DuplicateUidMap.Clear();

                List<object> objects;
                try
                {
                    using (_readMarker.Auto())
                    {
                        objects = _bytes == null ? yaml.Replay(_recording!)
                                : _recording != null ? yaml.DeserializeAll(_bytes, _recording)
                                : yaml.DeserializeAll(_bytes);
                    }
                }
                finally
                {
                    yaml.DeferAssetLoading = false;
                    yaml.DuplicateMode = false;
                }
                _bytes = null;

                var entities = Entities;
                Entity current = null;
                var attachScope = _attachMarker.Auto();

                foreach (var obj in objects)
                {
                    if (obj is Entity entity)
                    {
                        current = entity;
                        entities.Add(entity);
                    }
                    else if (obj is Component component && current != null)
                    {
                        if (component is Transform t)
                        {
                            CopyFields(t, current.Transform);

                            // Retarget deferred UID refs from temp Transform to entity's real Transform.
                            // Without this, Phase 2 would set Parent on the discarded temp object.
                            var deferredRefs = yaml.DeferredUniqueIdRefs;
                            for (int i = 0; i < deferredRefs.Count; i++)
                            {
                                if (ReferenceEquals(deferredRefs[i].Parent, t))
                                {
                                    var fixup = deferredRefs[i];
                                    fixup.Parent = current.Transform;
                                    deferredRefs[i] = fixup;
                                }
                            }

                            // Retarget DuplicateUidMap mapping from temp Transform to entity's real Transform
                            if (_owner.DuplicateMode)
                            {
                                ulong oldTransformUid = 0;
                                foreach (var kvp in yaml.DuplicateUidMap)
                                {
                                    if (kvp.Value == t.UID)
                                    {
                                        oldTransformUid = kvp.Key;
                                        break;
                                    }
                                }
                                if (oldTransformUid != 0)
                                {
                                    yaml.DuplicateUidMap[oldTransformUid] = current.Transform.UID;
                                }
                            }
                        }
                        else
                        {
                            current.AddComponent(component);
                        }
                    }
                }

                attachScope.Dispose();
                parseScope.Dispose();
                long parsed = Stopwatch.GetTimestamp();
                var referencesScope = _referencesMarker.Auto();

                // ── Phase 2: Build UID table and resolve IUniqueId references ──
                var uidLookup = new Dictionary<ulong, IUniqueId>();

                foreach (var entity in entities)
                {
                    if (entity.UID != 0)
                        uidLookup.TryAdd(entity.UID, entity);

                    foreach (var component in entity.Components)
                    {
                        if (component.UID != 0)
                            uidLookup.TryAdd(component.UID, component);
                    }
                }

                // Resolve deferred IUniqueId refs (entities + components)
                foreach (var deferred in yaml.DeferredUniqueIdRefs)
                {
                    ulong targetUid = deferred.UID;
                    if (_owner.DuplicateMode && yaml.DuplicateUidMap.TryGetValue(targetUid, out var newUid))
                    {
                        targetUid = newUid;
                    }

                    if (uidLookup.TryGetValue(targetUid, out var resolved))
                    {
                        try
                        {
                            deferred.Field.SetValue(deferred.Parent, resolved);
                        }
                        catch (Exception ex)
                        {
                            Debug.LogWarning("EntitySerializer",
                                $"UID ref resolve failed: UID {targetUid} (original: {deferred.UID}) on " +
                                $"{deferred.Parent.GetType().Name}.{deferred.Field.Name}: {ex.Message}");
                        }
                    }
                    else
                    {
                        Debug.LogWarning("EntitySerializer",
                            $"UID ref not found: UID {targetUid} (original: {deferred.UID}) for " +
                            $"{deferred.Parent.GetType().Name}.{deferred.Field.Name}");
                    }
                }

                yaml.DeferredUniqueIdRefs.Clear();
                _uidCount = uidLookup.Count;

                referencesScope.Dispose();
                long referenced = Stopwatch.GetTimestamp();
                _parseTicks = parsed - start;
                _referencesTicks = referenced - parsed;
            }

            private void Finish()
            {
                _phase = Phase.Done;
                static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;
                _owner.LastTimings = new Timings(Ms(_parseTicks), Ms(_referencesTicks), Ms(_assetsTicks), Ms(_prefabsTicks));

                Debug.Log($"[EntitySerializer] Loaded {Entities.Count} entities " +
                          $"({_prefabCount} prefab instances, {_uidCount} UIDs resolved)");
            }
        }

        // ── Asset stub resolution ─────────────────────────────────

        /// <summary>
        /// Walk all components on all entities. For each Asset-typed field
        /// that holds a stub (has Guid but not loaded), resolve via LoadByGuid.
        /// Also recurses into List fields of [Serializable] elements.
        /// </summary>
        private static void ResolveAssetStubs(List<Entity> entities)
        {
            foreach (var entity in entities)
            {
                foreach (var component in entity.Components)
                    ResolveObjectStubs(component, component.GetType().Name);
            }
        }

        // Which fields of a type can hold an asset stub, worked out once per type: testing every field of every
        // component with reflection on each load cost more than the asset lookups themselves. Keyed by the
        // Reflector's mapping, which a script reload replaces, so a plan never outlives the types it names.
        private enum StubSlot : byte { Nested, Asset, AssetList, NestedList }
        private sealed record StubPlan((Field Field, StubSlot Slot, Type? Element)[] Slots, bool Recurses);
        private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Mapping, StubPlan> _stubPlans = new();

        private static StubPlan BuildStubPlan(Mapping mapping)
        {
            var slots = new List<(Field, StubSlot, Type?)>();
            foreach (var field in mapping)
            {
                // A nested [Serializable] object (the item inside an equipment entry): its own fields, as the
                // serializer wrote them. Strings and primitives also count as serializable to reflection, so
                // only plain classes the serializer writes as a mapping qualify.
                if (field.CanWrite && !field.Ignored && field.Type.IsClass && field.Type != typeof(string)
                    && !typeof(Asset).IsAssignableFrom(field.Type) && !typeof(IList).IsAssignableFrom(field.Type)
                    && field.Type.IsSerializable)
                {
                    slots.Add((field, StubSlot.Nested, null));
                    continue;
                }

                // Direct Asset field
                if (typeof(Asset).IsAssignableFrom(field.Type))
                {
                    slots.Add((field, StubSlot.Asset, null));
                    continue;
                }

                // List of Asset references, or of nested [Serializable] objects
                if (typeof(IList).IsAssignableFrom(field.Type))
                {
                    var elType = field.Type.IsArray
                        ? field.Type.GetElementType()
                        : field.Type.IsGenericType
                            ? field.Type.GetGenericArguments()[0]
                            : null;

                    if (elType != null && typeof(Asset).IsAssignableFrom(elType))
                        slots.Add((field, StubSlot.AssetList, elType));
                    else if (elType != null && Reflector.GetAttribute<SerializableAttribute>(elType) != null)
                        slots.Add((field, StubSlot.NestedList, elType));
                }
            }
            return new StubPlan(slots.ToArray(), slots.Exists(s => s.Item2 is StubSlot.Nested or StubSlot.NestedList));
        }

        private static void ResolveObjectStubs(object obj, string context, int depth = 0, HashSet<object> seen = null)
        {
            // Data that refers back to itself (an animation state and its transitions) is walked once
            if (seen != null && !seen.Add(obj)) return;

            var plan = _stubPlans.GetValue(Reflector.GetMapping(obj.GetType()), BuildStubPlan);
            if (plan.Recurses && seen == null)
            {
                seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
                seen.Add(obj);
            }

            foreach (var (field, slot, elType) in plan.Slots)
            {
                if (slot == StubSlot.Nested)
                {
                    if (depth >= 16) continue;
                    var nested = field.GetValue(obj);
                    if (nested != null) ResolveObjectStubs(nested, $"{context}.{field.Name}", depth + 1, seen);
                    continue;
                }

                if (slot == StubSlot.Asset)
                {
                    var stub = field.GetValue(obj) as Asset;
                    if (stub == null || string.IsNullOrEmpty(stub.Guid))
                        continue;

                    try
                    {
                        var loaded = Engine.Assets.LoadByGuid(stub.Guid, field.Type);
                        if (loaded != null)
                            field.SetValue(obj, loaded);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning("EntitySerializer",
                            $"Stub resolve failed: {field.Type.Name} " +
                            $"'{stub.Guid}' on {context}: {ex.Message}");
                    }
                    continue;
                }

                // A list: of Asset references (each element resolved), or of nested [Serializable] objects
                {
                    var list = field.GetValue(obj) as IList;
                    if (list == null || list.Count == 0) continue;

                    if (slot == StubSlot.AssetList)
                    {
                        for (int i = 0; i < list.Count; i++)
                        {
                            var stub = list[i] as Asset;
                            if (stub == null || string.IsNullOrEmpty(stub.Guid)) continue;
                            try
                            {
                                var loaded = Engine.Assets.LoadByGuid(stub.Guid, elType);
                                if (loaded != null)
                                    list[i] = loaded;
                            }
                            catch (Exception ex)
                            {
                                Debug.LogWarning("EntitySerializer",
                                    $"List stub resolve failed: {elType.Name} " +
                                    $"'{stub.Guid}' on {context}.{field.Name}[{i}]: {ex.Message}");
                            }
                        }
                        continue;
                    }

                    if (depth < 16)
                    {
                        for (int i = 0; i < list.Count; i++)
                        {
                            if (list[i] != null)
                                ResolveObjectStubs(list[i], $"{context}.{field.Name}[{i}]", depth + 1, seen);
                        }
                    }
                }
            }
        }

        private static void CopyFields(object source, object target)
        {
            var mapping = Reflector.GetMapping(source.GetType());
            foreach (var field in mapping)
            {
                if (!field.CanWrite) continue;
                // Never overwrite Entity backref — it's set by AddComponent
                if (field.Name == "Entity") continue;
                var value = field.GetValue(source);
                if (value != null)
                    field.SetValue(target, value);
            }
        }
    }
}
