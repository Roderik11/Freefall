using System;
using System.Collections.Generic;
using System.Text.Json;

namespace Freefall.Editor.Tools
{
    // ═══════════════════════════════
    // ── Dwellings Data Model ──
    // ═══════════════════════════════

    /// <summary>Parsed Watabou Dwellings building data.</summary>
    public class DwellingsData
    {
        public List<DwellingsFloor> Floors = new();
        public DwellingsEdge Exit;
    }

    public class DwellingsFloor
    {
        public int Level;
        public List<DwellingsRoom> Rooms = new();
        public List<DwellingsDoor> Doors = new();
        public List<DwellingsEdge> Windows = new();
        public List<DwellingsStair> Stairs = new();
    }

    public class DwellingsRoom
    {
        public string Name;
        public List<CellCoord> Cells = new();
    }

    public class DwellingsDoor
    {
        public DwellingsEdge Edge;
        public string Type; // "REGULAR", etc.
    }

    public class DwellingsStair
    {
        public CellCoord Cell;
        public string Dir;
        public bool Up;
    }

    public class DwellingsEdge
    {
        public CellCoord Cell;
        public string Dir; // "n", "s", "e", "w"
    }

    public struct CellCoord : IEquatable<CellCoord>
    {
        public int I, J;
        public CellCoord(int i, int j) { I = i; J = j; }

        public CellCoord Neighbor(string dir) => dir switch
        {
            "n" => new CellCoord(I - 1, J),
            "s" => new CellCoord(I + 1, J),
            "e" => new CellCoord(I, J + 1),
            "w" => new CellCoord(I, J - 1),
            _ => this
        };

        public bool Equals(CellCoord other) => I == other.I && J == other.J;
        public override bool Equals(object obj) => obj is CellCoord c && Equals(c);
        public override int GetHashCode() => HashCode.Combine(I, J);
        public override string ToString() => $"({I},{J})";
    }

    // ═══════════════════════════════
    // ── Parser ──
    // ═══════════════════════════════

    /// <summary>
    /// Parses Watabou Dwellings JSON into a DwellingsData model.
    /// Format: { floors: [{ level, rooms, doors, windows, stairs }], exit }
    /// </summary>
    public static class DwellingsParser
    {
        public static DwellingsData Parse(string json)
        {
            var data = new DwellingsData();
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (root.TryGetProperty("floors", out var floors))
            {
                foreach (var floorEl in floors.EnumerateArray())
                    data.Floors.Add(ParseFloor(floorEl));
            }

            if (root.TryGetProperty("exit", out var exitEl))
                data.Exit = ParseEdge(exitEl);

            return data;
        }

        private static DwellingsFloor ParseFloor(JsonElement el)
        {
            var floor = new DwellingsFloor();

            if (el.TryGetProperty("level", out var lvl))
                floor.Level = lvl.GetInt32();

            if (el.TryGetProperty("rooms", out var rooms))
            {
                foreach (var roomEl in rooms.EnumerateArray())
                    floor.Rooms.Add(ParseRoom(roomEl));
            }

            if (el.TryGetProperty("doors", out var doors))
            {
                foreach (var doorEl in doors.EnumerateArray())
                    floor.Doors.Add(ParseDoor(doorEl));
            }

            if (el.TryGetProperty("windows", out var windows))
            {
                foreach (var winEl in windows.EnumerateArray())
                    floor.Windows.Add(ParseEdge(winEl));
            }

            if (el.TryGetProperty("stairs", out var stairs))
            {
                foreach (var stairEl in stairs.EnumerateArray())
                    floor.Stairs.Add(ParseStair(stairEl));
            }

            return floor;
        }

        private static DwellingsRoom ParseRoom(JsonElement el)
        {
            var room = new DwellingsRoom();

            if (el.TryGetProperty("name", out var name))
                room.Name = name.GetString();

            if (el.TryGetProperty("cells", out var cells))
            {
                foreach (var cellEl in cells.EnumerateArray())
                    room.Cells.Add(ReadCell(cellEl));
            }

            return room;
        }

        private static DwellingsDoor ParseDoor(JsonElement el)
        {
            var door = new DwellingsDoor();

            if (el.TryGetProperty("edge", out var edge))
                door.Edge = ParseEdge(edge);

            if (el.TryGetProperty("type", out var type))
                door.Type = type.GetString();

            return door;
        }

        private static DwellingsEdge ParseEdge(JsonElement el)
        {
            var edge = new DwellingsEdge();

            if (el.TryGetProperty("cell", out var cell))
                edge.Cell = ReadCell(cell);

            if (el.TryGetProperty("dir", out var dir))
                edge.Dir = dir.GetString();

            return edge;
        }

        private static DwellingsStair ParseStair(JsonElement el)
        {
            var stair = new DwellingsStair();

            if (el.TryGetProperty("cell", out var cell))
                stair.Cell = ReadCell(cell);

            if (el.TryGetProperty("dir", out var dir))
                stair.Dir = dir.GetString();

            if (el.TryGetProperty("up", out var up))
                stair.Up = up.GetBoolean();

            return stair;
        }

        private static CellCoord ReadCell(JsonElement el)
        {
            int i = 0, j = 0;
            if (el.TryGetProperty("i", out var iProp)) i = iProp.GetInt32();
            if (el.TryGetProperty("j", out var jProp)) j = jProp.GetInt32();
            return new CellCoord(i, j);
        }
    }
}
