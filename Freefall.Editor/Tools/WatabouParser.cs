using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text.Json;

namespace Freefall.Editor.Tools
{
    // ═══════════════════════════════
    // ── Data Model ──
    // ═══════════════════════════════

    /// <summary>Parsed Watabou village/city data.</summary>
    public class WatabouData
    {
        // Global values
        public float RoadWidth = 8;
        public float WallThickness = 7.6f;
        public float TowerRadius = 7.6f;
        public float RiverWidth = 30;

        public List<Vector2> EarthBoundary = new();
        public List<WatabouPolyline> Roads = new();
        public List<WatabouPolygon> Walls = new();
        public List<WatabouPolyline> Rivers = new();
        public List<WatabouPolyline> Planks = new();
        public List<List<Vector2>> Buildings = new();

        // Extended features
        public List<List<Vector2>> Fields = new();
        public List<List<Vector2>> Prisms = new();
        public List<List<Vector2>> Squares = new();
        public List<List<Vector2>> Greens = new();
        public List<Vector2> Trees = new();
        public List<WatabouDistrict> Districts = new();
    }

    /// <summary>A polyline with width (roads, rivers, planks).</summary>
    public class WatabouPolyline
    {
        public float Width;
        public List<Vector2> Points = new();
    }

    /// <summary>A polygon with width (walls).</summary>
    public class WatabouPolygon
    {
        public float Width;
        public List<Vector2> Points = new();
    }

    /// <summary>A named district polygon.</summary>
    public class WatabouDistrict
    {
        public string Name = "";
        public List<Vector2> Points = new();
    }

    // ═══════════════════════════════
    // ── Parser ──
    // ═══════════════════════════════

    /// <summary>
    /// Parses Watabou village/city JSON (GeoJSON FeatureCollection format).
    /// Uses System.Text.Json for zero-dependency parsing.
    /// </summary>
    public static class WatabouParser
    {
        public static WatabouData Parse(string json)
        {
            var data = new WatabouData();
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // The root is a FeatureCollection with a "features" array
            if (!root.TryGetProperty("features", out var features))
                throw new InvalidOperationException("Missing 'features' array in Watabou JSON");

            foreach (var feature in features.EnumerateArray())
            {
                string id = GetId(feature);
                string type = GetType(feature);

                switch (id)
                {
                    case "values":
                        ParseValues(feature, data);
                        break;
                    case "earth":
                        data.EarthBoundary = ParsePolygonCoords(feature);
                        break;
                    case "roads":
                        ParseGeometryCollection(feature, data.Roads, parseAsPolyline: true);
                        break;
                    case "walls":
                        ParseGeometryCollectionPolygons(feature, data.Walls);
                        break;
                    case "rivers":
                        ParseGeometryCollection(feature, data.Rivers, parseAsPolyline: true);
                        break;
                    case "planks":
                        ParseGeometryCollection(feature, data.Planks, parseAsPolyline: true);
                        break;
                    case "buildings":
                        ParseBuildings(feature, data);
                        break;
                    case "fields":
                        ParseMultiPolygon(feature, data.Fields);
                        break;
                    case "prisms":
                        ParseMultiPolygon(feature, data.Prisms);
                        break;
                    case "squares":
                        ParseMultiPolygon(feature, data.Squares);
                        break;
                    case "greens":
                        ParseMultiPolygon(feature, data.Greens);
                        break;
                    case "trees":
                        ParseMultiPoint(feature, data.Trees);
                        break;
                    case "districts":
                        ParseDistricts(feature, data.Districts);
                        break;
                }
            }

            return data;
        }

        // ── Helpers ──

        private static string GetId(JsonElement el)
        {
            if (el.TryGetProperty("id", out var id))
                return id.GetString() ?? "";
            return "";
        }

        private static string GetType(JsonElement el)
        {
            if (el.TryGetProperty("type", out var t))
                return t.GetString() ?? "";
            return "";
        }

        private static void ParseValues(JsonElement el, WatabouData data)
        {
            if (el.TryGetProperty("roadWidth", out var rw))
                data.RoadWidth = rw.GetSingle();
            if (el.TryGetProperty("wallThickness", out var wt))
                data.WallThickness = wt.GetSingle();
            if (el.TryGetProperty("towerRadius", out var tr))
                data.TowerRadius = tr.GetSingle();
            if (el.TryGetProperty("riverWidth", out var riw))
                data.RiverWidth = riw.GetSingle();
        }

        /// <summary>Parse a Polygon feature's coordinates → flat list of Vector2.</summary>
        private static List<Vector2> ParsePolygonCoords(JsonElement feature)
        {
            var result = new List<Vector2>();
            if (!feature.TryGetProperty("coordinates", out var coords))
                return result;

            // Polygon: coordinates is [ring[]] where ring is [[x,y], ...]
            foreach (var ring in coords.EnumerateArray())
            {
                foreach (var point in ring.EnumerateArray())
                {
                    result.Add(ReadPoint(point));
                }
                break; // Only take the outer ring
            }
            return result;
        }

        /// <summary>Parse a GeometryCollection of LineStrings into polylines.</summary>
        private static void ParseGeometryCollection(JsonElement feature, List<WatabouPolyline> target, bool parseAsPolyline)
        {
            if (!feature.TryGetProperty("geometries", out var geometries))
                return;

            foreach (var geom in geometries.EnumerateArray())
            {
                var line = new WatabouPolyline();

                if (geom.TryGetProperty("width", out var w))
                    line.Width = w.GetSingle();

                if (geom.TryGetProperty("coordinates", out var coords))
                {
                    foreach (var point in coords.EnumerateArray())
                        line.Points.Add(ReadPoint(point));
                }

                if (line.Points.Count > 0)
                    target.Add(line);
            }
        }

        /// <summary>Parse a GeometryCollection of Polygons into polygon list.</summary>
        private static void ParseGeometryCollectionPolygons(JsonElement feature, List<WatabouPolygon> target)
        {
            if (!feature.TryGetProperty("geometries", out var geometries))
                return;

            foreach (var geom in geometries.EnumerateArray())
            {
                var poly = new WatabouPolygon();

                if (geom.TryGetProperty("width", out var w))
                    poly.Width = w.GetSingle();

                if (geom.TryGetProperty("coordinates", out var coords))
                {
                    // Polygon: [ring[]] → take outer ring
                    foreach (var ring in coords.EnumerateArray())
                    {
                        foreach (var point in ring.EnumerateArray())
                            poly.Points.Add(ReadPoint(point));
                        break;
                    }
                }

                if (poly.Points.Count > 0)
                    target.Add(poly);
            }
        }

        /// <summary>Parse MultiPolygon buildings.</summary>
        private static void ParseBuildings(JsonElement feature, WatabouData data)
        {
            if (!feature.TryGetProperty("coordinates", out var coords))
                return;

            // MultiPolygon: [ polygon[], polygon[], ... ]
            // Each polygon: [ ring[], ... ] where ring: [ [x,y], ... ]
            foreach (var polygon in coords.EnumerateArray())
            {
                var building = new List<Vector2>();
                foreach (var ring in polygon.EnumerateArray())
                {
                    foreach (var point in ring.EnumerateArray())
                        building.Add(ReadPoint(point));
                    break; // Outer ring only
                }
                if (building.Count >= 3)
                    data.Buildings.Add(building);
            }
        }

        /// <summary>Read a [x, y] coordinate pair as Vector2 (x→X, y→Z mapped later).</summary>
        private static Vector2 ReadPoint(JsonElement point)
        {
            var enumerator = point.EnumerateArray();
            enumerator.MoveNext();
            float x = enumerator.Current.GetSingle();
            enumerator.MoveNext();
            float y = enumerator.Current.GetSingle();
            return new Vector2(x, y);
        }

        /// <summary>Parse a MultiPolygon feature into a list of polygons.</summary>
        private static void ParseMultiPolygon(JsonElement feature, List<List<Vector2>> target)
        {
            if (!feature.TryGetProperty("coordinates", out var coords))
                return;

            foreach (var polygon in coords.EnumerateArray())
            {
                var poly = new List<Vector2>();
                foreach (var ring in polygon.EnumerateArray())
                {
                    foreach (var point in ring.EnumerateArray())
                        poly.Add(ReadPoint(point));
                    break;
                }
                if (poly.Count >= 3)
                    target.Add(poly);
            }
        }

        /// <summary>Parse a MultiPoint feature into a list of points.</summary>
        private static void ParseMultiPoint(JsonElement feature, List<Vector2> target)
        {
            if (!feature.TryGetProperty("coordinates", out var coords))
                return;

            foreach (var point in coords.EnumerateArray())
                target.Add(ReadPoint(point));
        }

        /// <summary>Parse a GeometryCollection of named Polygon districts.</summary>
        private static void ParseDistricts(JsonElement feature, List<WatabouDistrict> target)
        {
            if (!feature.TryGetProperty("geometries", out var geometries))
                return;

            foreach (var geom in geometries.EnumerateArray())
            {
                var district = new WatabouDistrict();

                if (geom.TryGetProperty("name", out var name))
                    district.Name = name.GetString() ?? "";

                if (geom.TryGetProperty("coordinates", out var coords))
                {
                    foreach (var ring in coords.EnumerateArray())
                    {
                        foreach (var point in ring.EnumerateArray())
                            district.Points.Add(ReadPoint(point));
                        break;
                    }
                }

                if (district.Points.Count >= 3)
                    target.Add(district);
            }
        }
    }
}
