using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Threading;
using Vortice.Mathematics;
using System.Numerics;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Graph
{
    [Serializable]
    public abstract class Node
    {
        private readonly Dictionary<int, Port> HashedPorts = new Dictionary<int, Port>();
        private readonly Dictionary<int, object> _outputCache = new Dictionary<int, object>();

        [Browsable(false)]
        public int ID;

        [Browsable(false)]
        public Vector2 Position;

        [Browsable(false)]
        public bool Expanded;

        [Browsable(false)]
        public int Seed;

        public readonly List<Port> Ports = new List<Port>();
        public readonly List<Port> Inputs = new List<Port>();
        public readonly List<Port> Outputs = new List<Port>();

        public NodeGraph Graph { get; private set; }

        public Port GetPort(string name)
        {
            int hash = Port.StableHash(name);
            HashedPorts.TryGetValue(hash, out var port);
            return port;
        }

        public Port GetPort(int hash)
        {
            if (HashedPorts.TryGetValue(hash, out Port port))
                return port;
            return null;
        }

        /// <summary>
        /// Override to produce output values. Called during graph execution
        /// in topological order. Use GetInputValue to pull upstream data
        /// and SetOutput to cache results for downstream nodes.
        /// </summary>
        public virtual void Process() { }

        /// <summary>
        /// Pull a value from a connected upstream port, or fall back to
        /// the backing field value when the port is unconnected.
        /// </summary>
        public T GetInputValue<T>(string portName)
        {
            var port = GetPort(portName);
            if (port == null)
                return default;

            var connected = port.GetConnectedPort();
            if (connected != null)
                return connected.Node.GetOutput<T>(connected);

            // Unconnected — use the backing field value on this node
            return (T)port.Field.GetValue(this);
        }

        /// <summary>
        /// Cache an output value for downstream consumers.
        /// </summary>
        public void SetOutput<T>(string portName, T value)
        {
            var port = GetPort(portName);
            if (port != null)
            {
                _outputCache[port.FieldHash] = value;
                port.Field.SetValue(this, value);
            }
        }

        /// <summary>
        /// Retrieve a cached output value. Called by downstream nodes
        /// via GetInputValue.
        /// </summary>
        internal T GetOutput<T>(Port port)
        {
            if (_outputCache.TryGetValue(port.FieldHash, out var cached))
                return (T)cached;

            // Fall back to field value (e.g. for constant/default outputs)
            return (T)port.Field.GetValue(this);
        }

        /// <summary>
        /// Legacy override point for pull-based evaluation.
        /// </summary>
        public virtual object GetValue(Port port)
        {
            return default;
        }

        /// <summary>
        /// Clear cached output values. Called before graph re-execution.
        /// </summary>
        public void ClearCache()
        {
            _outputCache.Clear();
        }

        /// <summary>
        /// Create a deterministic Random from this node's seed.
        /// </summary>
        protected Random CreateRandom()
        {
            return new Random(Seed);
        }

        public void CreatePorts(NodeGraph graph)
        {
            Graph = graph;
            Ports.Clear();
            Inputs.Clear();
            Outputs.Clear();
            HashedPorts.Clear();

            var inputs = Reflector.GetMapping<InputAttribute>(GetType());

            foreach (var field in inputs)
            {
                var att = field.GetAttribute<InputAttribute>();

                var port = new Port(this, field)
                {
                    Type = Port.InOut.Input,
                    Constraint = att.typeConstraint,
                    ConnectionType = att.connectionType,
                };

                Ports.Add(port);
                Inputs.Add(port);
                HashedPorts.Add(port.FieldHash, port);
            }

            var outputs = Reflector.GetMapping<OutputAttribute>(GetType());

            foreach (var field in outputs)
            {
                var att = field.GetAttribute<OutputAttribute>();

                var port = new Port(this, field)
                {
                    Type = Port.InOut.Output,
                    Constraint = att.typeConstraint,
                    ConnectionType = att.connectionType,
                };

                Ports.Add(port);
                Outputs.Add(port);
                HashedPorts.Add(port.FieldHash, port);
            }
        }
    }

    public enum ShowBackingValue
    {
        Never,
        Unconnected,
        Always
    }

    public enum ConnectionType
    {
        Multiple,
        Single,
    }

    public enum TypeConstraint
    {
        /// <summary> Allow all types of input</summary>
        None,

        /// <summary> Allow connections where input value type is assignable from output value type (eg. ScriptableObject --> Object)</summary>
        Inherited,

        /// <summary> Allow only similar types </summary>
        Strict,

        /// <summary> Allow connections where output value type is assignable from input value type (eg. Object --> ScriptableObject)</summary>
        InheritedInverse,

        /// <summary> Allow connections where output value type is assignable from input value or input value type is assignable from output value type</summary>
        InheritedAny
    }
}