using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace Freefall.Graph
{
    /// <summary>
    /// Layered left-to-right auto layout for node graphs: a node sits one column right of its
    /// furthest-right input, and nodes within a column are ordered by the average row of what feeds
    /// them. Unconnected groups are laid out separately and stacked top to bottom.
    /// Good for the mostly-linear chains PCG graphs are; it does not route long wires around nodes.
    /// </summary>
    public static class GraphLayout
    {
        // Node cards as the graph editor draws them (GraphFrame)
        public const int NodeWidth = 168;
        private const int NodeBaseHeight = 40;
        private const int PortRowHeight = 20;

        private const int ColumnGap = 72;
        private const int RowGap = 32;
        private const int GroupGap = 56;
        private const int Grid = 8;

        // Where an empty graph starts: the middle of the editor's canvas
        private const float DefaultOrigin = 500000;

        public static int NodeHeight(Node node) => NodeBaseHeight + PortRowHeight * node.Ports.Count;

        /// <summary>
        /// Reposition every node. The layout keeps the top-left corner of the area the nodes occupied.
        /// </summary>
        public static void Arrange(NodeGraph graph)
        {
            var nodes = graph.Nodes;
            if (nodes.Count == 0) return;

            // Producer → consumer edges, whichever way round the connection was stored
            var inputs = nodes.ToDictionary(n => n, _ => new List<Node>());
            var neighbours = nodes.ToDictionary(n => n, _ => new List<Node>());
            foreach (var connection in graph.Connections)
            {
                if (connection.PortA?.Node == null || connection.PortB?.Node == null) continue;
                bool aIsSource = connection.PortA.Type == Port.InOut.Output;
                Node source = aIsSource ? connection.PortA.Node : connection.PortB.Node;
                Node sink = aIsSource ? connection.PortB.Node : connection.PortA.Node;
                if (source == sink || !inputs.ContainsKey(source) || !inputs.ContainsKey(sink)) continue;

                inputs[sink].Add(source);
                neighbours[source].Add(sink);
                neighbours[sink].Add(source);
            }

            // Column = longest chain of inputs behind the node
            var column = new Dictionary<Node, int>();
            var visiting = new HashSet<Node>();
            int ColumnOf(Node node)
            {
                if (column.TryGetValue(node, out int known)) return known;
                if (!visiting.Add(node)) return 0;     // cycle: break it here

                int result = 0;
                foreach (var input in inputs[node])
                    result = Math.Max(result, ColumnOf(input) + 1);

                visiting.Remove(node);
                return column[node] = result;
            }
            foreach (var node in nodes) ColumnOf(node);

            // Connected groups, in the order the user had them (top to bottom, then left to right)
            var groups = new List<List<Node>>();
            var grouped = new HashSet<Node>();
            foreach (var start in nodes.OrderBy(n => n.Position.Y).ThenBy(n => n.Position.X))
            {
                if (!grouped.Add(start)) continue;
                var group = new List<Node>();
                var pending = new Stack<Node>();
                pending.Push(start);
                while (pending.Count > 0)
                {
                    var node = pending.Pop();
                    group.Add(node);
                    foreach (var next in neighbours[node])
                        if (grouped.Add(next)) pending.Push(next);
                }
                groups.Add(group);
            }

            float originX = Snap(nodes.Min(n => n.Position.X));
            float originY = Snap(nodes.Min(n => n.Position.Y));
            if (nodes.All(n => n.Position == Vector2.Zero))
                originX = originY = DefaultOrigin;

            float top = originY;
            foreach (var group in groups)
            {
                int rowHeight = group.Max(NodeHeight) + RowGap;
                int firstColumn = group.Min(n => column[n]);

                var columns = group.GroupBy(n => column[n]).OrderBy(c => c.Key).ToList();
                int tallest = columns.Max(c => c.Count());

                // Row within the column: average row of the inputs, falling back to where the node was
                var row = new Dictionary<Node, float>();
                foreach (var members in columns)
                {
                    var ordered = members
                        .OrderBy(n => inputs[n].Where(row.ContainsKey).Select(i => row[i]).DefaultIfEmpty(float.MaxValue).Average())
                        .ThenBy(n => n.Position.Y)
                        .ToList();

                    // Shorter columns are centred against the tallest one
                    float offset = (tallest - ordered.Count) / 2f;
                    for (int i = 0; i < ordered.Count; i++)
                    {
                        row[ordered[i]] = offset + i;
                        ordered[i].Position = new Vector2(
                            originX + (members.Key - firstColumn) * (NodeWidth + ColumnGap),
                            Snap(top + (offset + i) * rowHeight));
                    }
                }

                top += tallest * rowHeight - RowGap + GroupGap;
            }
        }

        private static float Snap(float value) => MathF.Round(value / Grid) * Grid;
    }
}
