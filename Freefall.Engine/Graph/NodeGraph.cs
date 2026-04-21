using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Freefall.Base;

namespace Freefall.Graph
{

    public class NodeGraph //: IOnDeserialize
    {
        public List<Node> Nodes = new List<Node>();
        public List<Connection> Connections = new List<Connection>();

        private readonly Dictionary<int, Node> hashedNodes = new Dictionary<int, Node>();
        private int nodeIDcounter;

        /// <summary>
        /// Master seed for deterministic graph execution.
        /// Each node receives a derived seed based on this + its ID.
        /// </summary>
        public int Seed = 42;

        public NodeGraph()
        {

        }

        public void AddNode(Node node)
        {
            node.ID = Interlocked.Increment(ref nodeIDcounter);
            node.CreatePorts(this);
            Nodes.Add(node);
            hashedNodes.Add(node.ID, node);
        }

        public void RemoveNode(Node node)
        {
            foreach(var port in node.Ports)
                ClearConnections(port);
            Nodes.Remove(node);
            hashedNodes.Remove(node.ID);
        }

        private Connection GetConnection(Port a, Port b)
        {
            foreach (var connection in Connections)
            {
                bool check1 = connection.FromNodeID == a.Node.ID && connection.ToNodeID == b.Node.ID;
                bool check2 = connection.FromFieldHash == a.FieldHash && connection.ToFieldHash == b.FieldHash;
                if (check1 && check2) return connection;

                bool check3 = connection.FromNodeID == b.Node.ID && connection.ToNodeID == a.Node.ID;
                bool check4 = connection.FromFieldHash == b.FieldHash && connection.ToFieldHash == a.FieldHash;
                if (check3 && check4) return connection;
            }

            return null;
        }

        /// <summary>
        /// Returns the port connected to the given input port (the source/upstream port).
        /// Returns null if the port is unconnected.
        /// </summary>
        public Port GetConnectedPort(Port port)
        {
            foreach (var connection in Connections)
            {
                if (connection.PortA == port) return connection.PortB;
                if (connection.PortB == port) return connection.PortA;
            }

            return null;
        }

        public bool IsConnected(Port port)
        {
            foreach (var connection in Connections)
            {
                if(connection.PortA == port) return true;
                if(connection.PortB == port) return true;
            }

            return false;
        }

        public bool IsConnected(Port a, Port b) => GetConnection(a, b) != null;

        public bool CanConnect(Port a, Port b)
        {
            if (a == b) return false;
            if (a.Node == b.Node) return false;
            if (a.Type == b.Type) return false;
            if (a.Graph != b.Graph) return false;
            if (a.Field.Type != b.Field.Type) return false;
            if (IsConnected(a, b)) return false;

            return true;
        }

        public void ClearConnections(params Port[] ports)
        {
            foreach (Port port in ports)
            {
                Connections.RemoveAll((x) => x.FromNodeID == port.Node.ID && x.FromFieldHash == port.FieldHash);
                Connections.RemoveAll((x) => x.ToNodeID == port.Node.ID && x.ToFieldHash == port.FieldHash);
            }
        }

        public void AddConnection(Port a, Port b)
        {
            if (a.ConnectionType == ConnectionType.Single)
                ClearConnections(a);

            if (b.ConnectionType == ConnectionType.Single)
                ClearConnections(b);

            Connections.Add(new Connection
            {
                PortA = a,
                PortB = b,
                FromNodeID = a.Node.ID,
                FromFieldHash = a.FieldHash,
                ToNodeID = b.Node.ID,
                ToFieldHash = b.FieldHash
            });
        }

        public void RemoveConnection(Port a, Port b)
        {
            var found = GetConnection(a, b);
            if (found != null)
                Connections.Remove(found);
        }

        /// <summary>
        /// Execute the graph: clear caches, assign deterministic seeds,
        /// topologically sort, then Process() each node in order.
        /// </summary>
        public void Execute()
        {
            // 1. Clear all caches
            foreach (var node in Nodes)
                node.ClearCache();

            // 2. Assign deterministic per-node seeds
            foreach (var node in Nodes)
                node.Seed = HashCombine(Seed, node.ID);

            // 3. Sort and execute
            var sorted = TopologicalSort();
            foreach (var node in sorted)
            {
                try
                {
                    node.Process();
                }
                catch (Exception e)
                {
                    Debug.Log($"[Graph] Node {node.GetType().Name} (ID:{node.ID}) failed: {e.Message}");
                }
            }
        }

        /// <summary>
        /// Kahn's algorithm for topological sorting.
        /// Determines execution order based on connection dependencies.
        /// </summary>
        public List<Node> TopologicalSort()
        {
            // Build adjacency: for each connection, the node with the Output port
            // feeds into the node with the Input port.
            var inDegree = new Dictionary<int, int>();
            var adjacency = new Dictionary<int, List<int>>();

            foreach (var node in Nodes)
            {
                inDegree[node.ID] = 0;
                adjacency[node.ID] = new List<int>();
            }

            foreach (var connection in Connections)
            {
                // Determine which node is the source (output) and which is the sink (input)
                int sourceID, sinkID;
                if (connection.PortA.Type == Port.InOut.Output)
                {
                    sourceID = connection.PortA.Node.ID;
                    sinkID = connection.PortB.Node.ID;
                }
                else
                {
                    sourceID = connection.PortB.Node.ID;
                    sinkID = connection.PortA.Node.ID;
                }

                adjacency[sourceID].Add(sinkID);
                inDegree[sinkID]++;
            }

            // Enqueue all nodes with no incoming edges
            var queue = new Queue<int>();
            foreach (var kvp in inDegree)
            {
                if (kvp.Value == 0)
                    queue.Enqueue(kvp.Key);
            }

            var sorted = new List<Node>();
            while (queue.Count > 0)
            {
                var id = queue.Dequeue();
                sorted.Add(hashedNodes[id]);

                foreach (var neighbor in adjacency[id])
                {
                    inDegree[neighbor]--;
                    if (inDegree[neighbor] == 0)
                        queue.Enqueue(neighbor);
                }
            }

            if (sorted.Count != Nodes.Count)
                Debug.Log("[Graph] Warning: cycle detected, some nodes were skipped.");

            return sorted;
        }

        private static int HashCombine(int h1, int h2)
        {
            return h1 ^ (h2 + unchecked((int)0x9e3779b9) + (h1 << 6) + (h1 >> 2));
        }

        //public void OnDeserialize(JSON json)
        //{
        //    int max = 0;
        //    foreach (var node in Nodes)
        //    {
        //        node.CreatePorts(this);
        //        hashedNodes.Add(node.ID, node);
        //        max = Math.Max(max, node.ID);
        //    }

        //    nodeIDcounter = max;

        //    foreach(var connection in Connections)
        //    {
        //        connection.PortA = hashedNodes[connection.FromNodeID].GetPort(connection.FromFieldHash);
        //        connection.PortB = hashedNodes[connection.ToNodeID].GetPort(connection.ToFieldHash);
        //    }
        //}
    }
}
