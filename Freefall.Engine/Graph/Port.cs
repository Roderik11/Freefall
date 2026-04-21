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
            FieldHash = field.Name.GetHashCode();
        }
    }
}
