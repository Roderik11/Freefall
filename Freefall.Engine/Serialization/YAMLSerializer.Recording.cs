using System;
using System.Collections;
using System.Collections.Generic;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Reflection;

namespace Freefall.Serialization
{
    /// <summary>
    /// Recording a parse and replaying it. Prefabs are instantiated thousands of times from the same YAML (PCG points,
    /// kit pieces), and reading the text again costs ~20-30 µs even for a three-document prefab, mostly the text and
    /// its type names. The first parse records what it did; later parses of the same YAML replay that: the same objects
    /// created, the same values handed to the same setters in the same order, the same asset stubs and UID references
    /// collected. Only values that cannot change are shared between instances (converter output: numbers, vectors,
    /// strings); everything else is made anew per replay, as a parse would.
    /// </summary>
    public partial class YAMLSerializer
    {
        /// <summary>What parsing one YAML did: fill it with <see cref="DeserializeAll(byte[], Recording)"/>, use it with <see cref="Replay"/>.</summary>
        public sealed class Recording
        {
            internal readonly List<Recipe> Documents = new();
            internal bool DuplicateMode;
            internal int Generation = -1;   // Reflector.Generation it was recorded under
            internal bool Failed;           // the parse did something a replay cannot repeat

            /// <summary>No script assembly came or went since it was recorded: the types and fields it holds are still the loaded ones.</summary>
            public bool IsCurrent => Generation == Reflector.Generation;

            /// <summary>The parse did something a replay cannot repeat (an error, a value that could be changed): parse that YAML instead.</summary>
            public bool HasFailed => Failed;

            /// <summary>Recorded without trouble, and still current.</summary>
            public bool IsUsable => !Failed && IsCurrent;

            /// <summary>Recorded in <see cref="YAMLSerializer.DuplicateMode"/> (fresh UIDs, old ones mapped): replay it the same way.</summary>
            public bool IsDuplicateMode => DuplicateMode;
        }

        internal sealed class Recipe
        {
            public Type? Type;              // null: the parse produced null here
            public readonly List<Op> Ops = new();
        }

        internal enum OpKind { Set, DuplicateUid, UidRef, AssetStub, Nested, PrimitiveArray, PrimitiveList, List }

        internal sealed class Op
        {
            public OpKind Kind;
            public Field Field = null!;
            public object? Value;           // Set: the value; DuplicateUid / UidRef: the UID; AssetStub: the GUID
            public Type? Type;              // AssetStub: the stub's type; PrimitiveArray: the element type; lists: the list type
            public Recipe? Nested;
            public object?[]? Values;       // PrimitiveArray / PrimitiveList
            public List<object>? Items;     // List: a Recipe or an AssetItem per element
        }

        internal sealed record AssetItem(Type Type, string Guid);

        private Recording? _recording;
        private Dictionary<object, Recipe>? _recipes;   // object the recorded parse created -> its recipe

        /// <summary>
        /// Parse like <see cref="DeserializeAll(byte[])"/> and record what that did into <paramref name="recording"/>,
        /// which it overwrites. Check <see cref="Recording.IsUsable"/> afterwards.
        /// </summary>
        public List<object> DeserializeAll(byte[] bytes, Recording recording)
        {
            recording.Documents.Clear();
            recording.DuplicateMode = DuplicateMode;
            recording.Generation = Reflector.Generation;
            recording.Failed = false;

            _recording = recording;
            _recipes = new Dictionary<object, Recipe>(ReferenceEqualityComparer.Instance);
            try
            {
                return DeserializeAll(bytes);
            }
            finally
            {
                _recording = null;
                _recipes = null;
            }
        }

        /// <summary>Do again what a recorded parse did: the objects it returned, made anew, in the same order.</summary>
        public List<object> Replay(Recording recording)
        {
            if (recording.DuplicateMode != DuplicateMode)
                throw new InvalidOperationException("A recording replays only in the DuplicateMode it was recorded in");

            var results = new List<object>(recording.Documents.Count);
            foreach (var recipe in recording.Documents)
            {
                try
                {
                    var instance = Build(recipe);
                    if (instance != null) results.Add(instance);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("YAMLSerializer", $"Replay doc #{results.Count}: {ex.GetType().Name}: {ex.Message}");
                }
            }
            return results;
        }

        private object? Build(Recipe recipe)
        {
            if (recipe.Type == null) return null;
            var instance = Activator.CreateInstance(recipe.Type)!;
            foreach (var op in recipe.Ops)
                Apply(instance, op);
            return instance;
        }

        private void Apply(object parent, Op op)
        {
            switch (op.Kind)
            {
                case OpKind.Set:
                    op.Field.SetValue(parent, op.Value);
                    break;

                case OpKind.DuplicateUid:
                    DuplicateUidMap[(ulong)op.Value!] = ((IUniqueId)parent).UID;
                    break;

                case OpKind.UidRef:
                    DeferredUniqueIdRefs.Add(new DeferredUniqueIdRef { Parent = parent, Field = op.Field, UID = (ulong)op.Value! });
                    break;

                case OpKind.AssetStub:
                {
                    var stub = (Asset)Activator.CreateInstance(op.Type!)!;
                    stub.Guid = (string)op.Value!;
                    op.Field.SetValue(parent, stub);
                    DeferredRefs.Add(new DeferredAssetRef { Parent = parent, Field = op.Field, ListIndex = -1, Guid = stub.Guid, AssetType = op.Type! });
                    break;
                }

                case OpKind.Nested:
                    op.Field.SetValue(parent, Build(op.Nested!));
                    break;

                case OpKind.PrimitiveArray:
                {
                    var values = op.Values!;
                    var array = Array.CreateInstance(op.Type!, values.Length);
                    for (int i = 0; i < values.Length; i++)
                        array.SetValue(Convert.ChangeType(values[i], op.Type!), i);
                    op.Field.SetValue(parent, array);
                    break;
                }

                case OpKind.PrimitiveList:
                {
                    var list = (op.Field.GetValue(parent) ?? Activator.CreateInstance(op.Type!)) as IList;
                    list!.Clear();
                    foreach (var value in op.Values!) list.Add(value);
                    op.Field.SetValue(parent, list);
                    break;
                }

                case OpKind.List:
                {
                    var list = (op.Field.GetValue(parent) ?? Activator.CreateInstance(op.Type!)) as IList;
                    list!.Clear();
                    foreach (var item in op.Items!)
                    {
                        if (item is Recipe recipe)
                        {
                            var value = Build(recipe);
                            if (value != null) list.Add(value);
                        }
                        else if (item is AssetItem asset)
                        {
                            var stub = (Asset)Activator.CreateInstance(asset.Type)!;
                            stub.Guid = asset.Guid;
                            int index = list.Count;
                            list.Add(stub);
                            DeferredRefs.Add(new DeferredAssetRef { Parent = list, Field = null!, ListIndex = index, Guid = asset.Guid, AssetType = asset.Type });
                        }
                    }
                    op.Field.SetValue(parent, list);
                    break;
                }
            }
        }

        // ── Recording hooks, called by the parse ─────────────────

        /// <summary>The parse made <paramref name="value"/>: give it a recipe that the ops on it go into.</summary>
        private void Track(object? value)
        {
            if (_recipes != null && value != null)
                _recipes[value] = new Recipe { Type = value.GetType() };
        }

        /// <summary>The recipe of an object the parse made (a null recipe for null); null when not recording.</summary>
        private Recipe? RecipeOf(object? value)
        {
            if (_recipes == null) return null;
            if (value == null) return new Recipe();
            if (_recipes.TryGetValue(value, out var recipe)) return recipe;

            _recording!.Failed = true;   // an object the parse did not make: the replay could not make it either
            return null;
        }

        private void Record(object parent, Op op)
        {
            if (_recipes == null) return;
            if (_recipes.TryGetValue(parent, out var recipe)) recipe.Ops.Add(op);
            else _recording!.Failed = true;
        }

        private void RecordSet(object parent, Field field, object? value)
        {
            if (_recipes == null) return;
            // Shared by every replay: only values nobody can change
            if (value != null && value is not string && !value.GetType().IsValueType)
                _recording!.Failed = true;
            Record(parent, new Op { Kind = OpKind.Set, Field = field, Value = value });
        }
    }
}
