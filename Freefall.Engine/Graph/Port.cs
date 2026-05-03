using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Freefall.Reflection;

namespace Freefall.Graph
{
    public class Port
    {
        public enum InOut
        {
            Input = 0,
            Output = 1
        }

        public InOut Type;
        public TypeConstraint Constraint;
        public ConnectionType ConnectionType;
        public Field Field;
        public int FieldHash;
        public Node Node;
        public object Tag;

        public NodeGraph Graph => Node.Graph;

        public bool IsConnected => Graph.IsConnected(this);

        /// <summary>
        /// Returns the port on the other end of an incoming connection,
        /// or null if this port is unconnected.
        /// </summary>
        public Port GetConnectedPort()
        {
            return Graph.GetConnectedPort(this);
        }

        public Port(Node node, Field field)
        {
            Node = node;
            Field = field;
            FieldHash = StableHash(field.Name);
        }

        /// <summary>
        /// Deterministic string hash (FNV-1a). Unlike string.GetHashCode(),
        /// this produces the same value across process runs.
        /// </summary>
        internal static int StableHash(string s)
        {
            unchecked
            {
                int hash = (int)2166136261;
                foreach (char c in s)
                {
                    hash ^= c;
                    hash *= 16777619;
                }
                return hash;
            }
        }
    }
}
