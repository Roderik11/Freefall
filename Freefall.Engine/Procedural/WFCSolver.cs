using System;
using System.Collections.Generic;

namespace Freefall.Procedural
{
    /// <summary>
    /// Generic 2D Wave Function Collapse solver.
    /// Operates on a grid of cells, each containing a set of possible tile IDs.
    /// Collapses cells by propagating adjacency constraints until all cells have a single tile.
    /// </summary>
    public class WFCSolver
    {
        /// <summary>Adjacency rules: for each tile, which tiles are allowed in each direction.</summary>
        public class RuleSet
        {
            // Direction enum: 0=Up, 1=Right, 2=Down, 3=Left
            public const int Up = 0, Right = 1, Down = 2, Left = 3;

            private readonly int _tileCount;
            // allowed[dir][tileA] = set of tileB that can be adjacent to tileA in direction dir
            private readonly HashSet<int>[,] _allowed;
            private readonly float[] _weights;

            public int TileCount => _tileCount;

            public RuleSet(int tileCount)
            {
                _tileCount = tileCount;
                _allowed = new HashSet<int>[4, tileCount];
                _weights = new float[tileCount];

                for (int d = 0; d < 4; d++)
                    for (int t = 0; t < tileCount; t++)
                        _allowed[d, t] = new HashSet<int>();

                for (int t = 0; t < tileCount; t++)
                    _weights[t] = 1f;
            }

            /// <summary>Set the selection weight for a tile (higher = more likely to be chosen).</summary>
            public void SetWeight(int tile, float weight) => _weights[tile] = weight;
            public float GetWeight(int tile) => _weights[tile];

            /// <summary>
            /// Declare that tileB can appear in direction dir from tileA.
            /// Automatically adds the reverse rule.
            /// </summary>
            public void Allow(int tileA, int dir, int tileB)
            {
                _allowed[dir, tileA].Add(tileB);
                _allowed[Opposite(dir), tileB].Add(tileA);
            }

            /// <summary>Allow tileA to be adjacent to tileB in ALL directions (symmetric).</summary>
            public void AllowAll(int tileA, int tileB)
            {
                for (int d = 0; d < 4; d++)
                    Allow(tileA, d, tileB);
            }

            /// <summary>Get allowed neighbors for tile in given direction.</summary>
            public HashSet<int> GetAllowed(int dir, int tile) => _allowed[dir, tile];

            public static int Opposite(int dir) => (dir + 2) % 4;
        }

        // Grid state
        private readonly int _width, _height;
        private readonly HashSet<int>[] _cells; // possible tiles per cell
        private readonly RuleSet _rules;
        private readonly Random _rng;

        // Neighbor offsets: Up, Right, Down, Left
        private static readonly int[] _dx = { 0, 1, 0, -1 };
        private static readonly int[] _dy = { -1, 0, 1, 0 };

        /// <summary>Width of the grid.</summary>
        public int Width => _width;
        /// <summary>Height of the grid (rows).</summary>
        public int Height => _height;

        public WFCSolver(int width, int height, RuleSet rules, int seed = 0)
        {
            _width = width;
            _height = height;
            _rules = rules;
            _rng = new Random(seed);

            _cells = new HashSet<int>[width * height];
            for (int i = 0; i < _cells.Length; i++)
            {
                _cells[i] = new HashSet<int>();
                for (int t = 0; t < rules.TileCount; t++)
                    _cells[i].Add(t);
            }
        }

        /// <summary>Get the possible tiles for a cell.</summary>
        public HashSet<int> GetCell(int x, int y) => _cells[y * _width + x];

        /// <summary>Get the resolved tile for a cell (-1 if not yet collapsed or contradiction).</summary>
        public int GetTile(int x, int y)
        {
            var cell = _cells[y * _width + x];
            if (cell.Count == 1)
            {
                using var e = cell.GetEnumerator();
                e.MoveNext();
                return e.Current;
            }
            return -1;
        }

        /// <summary>Force a cell to a specific tile and propagate constraints.</summary>
        public void Constrain(int x, int y, int tile)
        {
            var cell = _cells[y * _width + x];
            cell.Clear();
            cell.Add(tile);
            Propagate(x, y);
        }

        /// <summary>Remove a tile from a cell's possibilities and propagate.</summary>
        public void Ban(int x, int y, int tile)
        {
            var cell = _cells[y * _width + x];
            if (cell.Remove(tile))
                Propagate(x, y);
        }

        /// <summary>
        /// Run WFC to completion. Returns true if solved, false if contradiction.
        /// </summary>
        public bool Solve()
        {
            int maxIterations = _width * _height * _rules.TileCount;
            for (int iter = 0; iter < maxIterations; iter++)
            {
                // Find lowest-entropy uncollapsed cell
                int bestIdx = -1;
                int bestEntropy = int.MaxValue;

                for (int i = 0; i < _cells.Length; i++)
                {
                    int count = _cells[i].Count;
                    if (count <= 1) continue; // Already collapsed or contradiction
                    if (count < bestEntropy)
                    {
                        bestEntropy = count;
                        bestIdx = i;
                    }
                }

                if (bestIdx == -1)
                    return true; // All collapsed

                // Collapse: pick a tile weighted by selection weights
                int bx = bestIdx % _width;
                int by = bestIdx / _width;
                int chosen = PickWeighted(_cells[bestIdx]);
                _cells[bestIdx].Clear();
                _cells[bestIdx].Add(chosen);

                // Propagate constraints
                Propagate(bx, by);

                // Check for contradictions
                for (int i = 0; i < _cells.Length; i++)
                {
                    if (_cells[i].Count == 0)
                        return false; // Contradiction
                }
            }

            return true;
        }

        private int PickWeighted(HashSet<int> options)
        {
            float totalWeight = 0;
            foreach (int t in options)
                totalWeight += _rules.GetWeight(t);

            float r = (float)_rng.NextDouble() * totalWeight;
            float acc = 0;
            foreach (int t in options)
            {
                acc += _rules.GetWeight(t);
                if (r <= acc) return t;
            }

            // Fallback: return any
            using var e = options.GetEnumerator();
            e.MoveNext();
            return e.Current;
        }

        private void Propagate(int startX, int startY)
        {
            var queue = new Queue<(int x, int y)>();
            queue.Enqueue((startX, startY));

            while (queue.Count > 0)
            {
                var (cx, cy) = queue.Dequeue();
                var cell = _cells[cy * _width + cx];

                for (int dir = 0; dir < 4; dir++)
                {
                    int nx = cx + _dx[dir];
                    int ny = cy + _dy[dir];
                    if (nx < 0 || nx >= _width || ny < 0 || ny >= _height) continue;

                    var neighbor = _cells[ny * _width + nx];
                    if (neighbor.Count <= 1) continue;

                    // Compute which tiles are still valid for the neighbor
                    var validForNeighbor = new HashSet<int>();
                    foreach (int myTile in cell)
                    {
                        var allowed = _rules.GetAllowed(dir, myTile);
                        foreach (int a in allowed)
                        {
                            if (neighbor.Contains(a))
                                validForNeighbor.Add(a);
                        }
                    }

                    // Remove invalid tiles from neighbor
                    int before = neighbor.Count;
                    neighbor.IntersectWith(validForNeighbor);

                    if (neighbor.Count < before && neighbor.Count > 0)
                        queue.Enqueue((nx, ny));
                }
            }
        }
    }
}
