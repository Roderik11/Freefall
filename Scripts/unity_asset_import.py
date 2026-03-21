#!/usr/bin/env python3
"""
Unity → Freefall Asset Import Pipeline

Reusable script to discover Unity asset store packs and import them into a
Freefall project via the Editor's WebSocket command server.

Phases:
  1. discover  - Scan Unity source, build unity_lookup.json
  2. copy      - Copy textures + FBX into Freefall Assets/
  3. import    - Drive editor via WebSocket: refresh, lookup GUIDs, create materials + staticmeshes
  4. run-all   - Execute all phases sequentially

Usage:
  python unity_asset_import.py discover --source <unity_dir>
  python unity_asset_import.py copy --source <unity_dir> --target <freefall_assets_dir> --pack MedievalTown
  python unity_asset_import.py import --target <freefall_assets_dir> --pack MedievalTown
  python unity_asset_import.py run-all --source <unity_dir> --target <freefall_assets_dir> --pack MedievalTown
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
import asyncio

try:
    import websockets
except ImportError:
    websockets = None

# ---------------------------------------------------------------------------
# Phase 1: Discovery
# ---------------------------------------------------------------------------

# Unity HDRP texture slot → Freefall material slot mapping
TEXTURE_SLOT_MAP = {
    "_BaseColorMap": "Albedo",
    "_IDMap": "Albedo",       # These packs use ID maps as albedo
    "_NormalMap": "Normal",
    "_HeightMap": "HeightTex",
    # _MaskMap is packed HDRP (M/AO/Detail/Smoothness) — skip for now
}

# Texture slots we care about in Unity .mat files
UNITY_TEXTURE_SLOTS = [
    "_BaseColorMap", "_NormalMap", "_MaskMap", "_HeightMap",
    "_IDMap", "_PatternID", "_PatternNormal", "_DetailMap",
    "_EmissiveColorMap", "_MainTex",
]

# File extensions we import
TEXTURE_EXTENSIONS = {".png", ".tga", ".jpg", ".jpeg"}
CONVERT_EXTENSIONS = {".psd", ".exr"}  # Convert to PNG during copy (Freefall can't import these)
MODEL_EXTENSIONS = {".fbx", ".obj"}
SKIP_DIRECTORIES = {"Shaders", "VFX", "Scenes", "Animation"}


def parse_unity_meta(meta_path):
    """Extract guid from a Unity .meta file (always on line 2)."""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("guid:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def parse_unity_material(mat_path):
    """Extract texture references from a Unity HDRP .mat file (YAML)."""
    textures = {}
    try:
        with open(mat_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find texture entries: - _SlotName:\n    m_Texture: {fileID: X, guid: Y, type: Z}
        pattern = r"- (\w+):\s*\n\s*m_Texture:\s*\{fileID:\s*(\d+),\s*guid:\s*([a-f0-9]+)"
        for match in re.finditer(pattern, content):
            slot_name = match.group(1)
            file_id = int(match.group(2))
            guid = match.group(3)
            if file_id != 0 and slot_name in UNITY_TEXTURE_SLOTS:
                textures[slot_name] = guid
    except Exception as e:
        print(f"  Warning: Failed to parse material {mat_path}: {e}")
    return textures


def parse_unity_prefab(prefab_path):
    """Extract mesh and material references from a Unity .prefab file."""
    mesh_guid = None
    material_guids = []
    is_simple = True

    try:
        with open(prefab_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find MeshFilter → m_Mesh references
        mesh_pattern = r"MeshFilter:.*?m_Mesh:\s*\{fileID:\s*\d+,\s*guid:\s*([a-f0-9]+)"
        mesh_matches = re.findall(mesh_pattern, content, re.DOTALL)
        if mesh_matches:
            mesh_guid = mesh_matches[0]  # Use first mesh (LOD0)

        # Find MeshRenderer → m_Materials references (first renderer only for simple prefabs)
        mat_pattern = r"m_Materials:\s*\n((?:\s*-\s*\{[^\n]+\n)+)"
        mat_blocks = re.findall(mat_pattern, content)
        if mat_blocks:
            guid_pattern = r"guid:\s*([a-f0-9]+)"
            material_guids = re.findall(guid_pattern, mat_blocks[0])

        # Check if this is a nested/complex prefab (has PrefabInstance references)
        if "PrefabInstance:" in content or content.count("MeshFilter:") > 6:
            # More than 6 MeshFilters = very likely a compound assembly, not a simple LOD prefab
            is_simple = content.count("MeshFilter:") <= 6
    except Exception as e:
        print(f"  Warning: Failed to parse prefab {prefab_path}: {e}")

    return mesh_guid, material_guids, is_simple


def discover(source_dir, output_path=None):
    """Phase 1: Scan Unity source and build lookup table."""
    print(f"[Phase 1] Discovering assets in: {source_dir}")

    lookup = {
        "textures": {},
        "models": {},
        "materials": {},
        "prefabs": {},
        "source_dir": source_dir,
    }

    for root, dirs, files in os.walk(source_dir):
        # Skip non-importable directories
        rel_root = os.path.relpath(root, source_dir)
        if any(skip in rel_root.split(os.sep) for skip in SKIP_DIRECTORIES):
            continue

        for filename in files:
            if not filename.endswith(".meta"):
                continue

            meta_path = os.path.join(root, filename)
            companion = filename[:-5]  # Strip ".meta"
            companion_path = os.path.join(root, companion)
            companion_ext = os.path.splitext(companion)[1].lower()

            # Skip .meta files without companion (directory metas, etc.)
            if not os.path.isfile(companion_path):
                continue

            guid = parse_unity_meta(meta_path)
            if not guid:
                continue

            rel_path = os.path.relpath(companion_path, source_dir)
            name = os.path.splitext(companion)[0]

            if companion_ext in TEXTURE_EXTENSIONS or companion_ext in CONVERT_EXTENSIONS:
                lookup["textures"][guid] = {"path": rel_path, "name": name}

            elif companion_ext in MODEL_EXTENSIONS:
                lookup["models"][guid] = {"path": rel_path, "name": name}

            elif companion_ext == ".mat":
                textures = parse_unity_material(companion_path)
                lookup["materials"][guid] = {
                    "path": rel_path,
                    "name": name,
                    "textures": textures,
                }

            elif companion_ext == ".prefab":
                mesh_guid, mat_guids, is_simple = parse_unity_prefab(companion_path)
                lookup["prefabs"][guid] = {
                    "path": rel_path,
                    "name": name,
                    "mesh_guid": mesh_guid,
                    "material_guids": mat_guids,
                    "is_simple": is_simple,
                }

    # Summary
    print(f"  Textures:  {len(lookup['textures'])}")
    print(f"  Models:    {len(lookup['models'])}")
    print(f"  Materials: {len(lookup['materials'])}")
    print(f"  Prefabs:   {len(lookup['prefabs'])} ({sum(1 for p in lookup['prefabs'].values() if p['is_simple'])} simple)")

    if output_path is None:
        output_path = os.path.join(source_dir, "unity_lookup.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2)
    print(f"  Saved: {output_path}")

    return lookup


# ---------------------------------------------------------------------------
# Phase 2: Copy Raw Assets
# ---------------------------------------------------------------------------

def copy_assets(source_dir, target_assets_dir, pack_name, lookup=None):
    """Phase 2: Copy textures and models into Freefall Assets directory."""
    print(f"[Phase 2] Copying raw assets to: {target_assets_dir}/{pack_name}")

    if lookup is None:
        lookup_path = os.path.join(source_dir, "unity_lookup.json")
        with open(lookup_path, "r", encoding="utf-8") as f:
            lookup = json.load(f)

    copy_map = {}  # unity_relative_path → freefall_relative_path
    copied = {"textures": 0, "models": 0, "skipped": 0, "converted": 0}

    # Copy textures
    for guid, info in lookup["textures"].items():
        src = os.path.join(source_dir, info["path"])
        if not os.path.exists(src):
            copied["skipped"] += 1
            continue

        src_ext = os.path.splitext(src)[1].lower()
        needs_conversion = src_ext in CONVERT_EXTENSIONS

        # Preserve one level of category folder (e.g., Castle/, Foliage/)
        parts = info["path"].replace("\\", "/").split("/")
        # Find the category folder (one before the filename)
        if len(parts) >= 2:
            category = parts[-2]
            basename = os.path.basename(info["path"])
            if needs_conversion:
                basename = os.path.splitext(basename)[0] + ".png"
            dst_rel = os.path.join(pack_name, "Textures", category, basename)
        else:
            basename = os.path.basename(info["path"])
            if needs_conversion:
                basename = os.path.splitext(basename)[0] + ".png"
            dst_rel = os.path.join(pack_name, "Textures", basename)

        dst = os.path.join(target_assets_dir, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if not os.path.exists(dst):
            if needs_conversion:
                try:
                    from PIL import Image
                    img = Image.open(src)
                    img.save(dst, "PNG")
                    copied["converted"] += 1
                except Exception as e:
                    print(f"  Warning: Failed to convert {src}: {e}")
                    copied["skipped"] += 1
                    continue
            else:
                shutil.copy2(src, dst)
            copied["textures"] += 1

        copy_map[info["path"]] = dst_rel

    # Copy models
    for guid, info in lookup["models"].items():
        src = os.path.join(source_dir, info["path"])
        if not os.path.exists(src):
            copied["skipped"] += 1
            continue

        parts = info["path"].replace("\\", "/").split("/")
        if len(parts) >= 2:
            category = parts[-2]
            dst_rel = os.path.join(pack_name, "Meshes", category, os.path.basename(info["path"]))
        else:
            dst_rel = os.path.join(pack_name, "Meshes", os.path.basename(info["path"]))

        dst = os.path.join(target_assets_dir, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied["models"] += 1

        copy_map[info["path"]] = dst_rel

    print(f"  Copied: {copied['textures']} textures, {copied['models']} models")
    if copied["converted"]:
        print(f"  Converted: {copied['converted']} PSD/EXR → PNG")
    if copied["skipped"]:
        print(f"  Skipped: {copied['skipped']} (missing source or conversion failed)")

    # Save copy map
    map_path = os.path.join(source_dir, "copy_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(copy_map, f, indent=2)
    print(f"  Saved: {map_path}")

    return copy_map


# ---------------------------------------------------------------------------
# Phase 3-6: WebSocket-driven Import & Generation
# ---------------------------------------------------------------------------

class EditorClient:
    """WebSocket client for the Freefall Editor command server."""

    def __init__(self, url="ws://localhost:21721/ws"):
        self.url = url
        self.ws = None
        self._msg_id = 0

    async def connect(self):
        if websockets is None:
            raise RuntimeError("websockets package required: pip install websockets")
        self.ws = await websockets.connect(self.url, max_size=10 * 1024 * 1024)
        print(f"  Connected to editor: {self.url}")

    async def close(self):
        if self.ws:
            await self.ws.close()

    async def send(self, cmd, body=None, timeout=120):
        """Send a command and wait for the response."""
        self._msg_id += 1
        msg = {"id": self._msg_id, "cmd": cmd}
        if body is not None:
            msg["body"] = body

        await self.ws.send(json.dumps(msg))

        # Wait for response with matching id
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            resp = json.loads(raw)
            if resp.get("id") == self._msg_id:
                if resp.get("status", 200) >= 400:
                    raise RuntimeError(f"Editor error {resp.get('status')}: {resp.get('body')}")
                return resp.get("body")
            # Skip events
            if "event" in resp:
                continue


async def import_and_generate(source_dir, target_assets_dir, pack_name, editor_url="ws://localhost:21721/ws"):
    """Phases 3-6: Connect to editor, import, harvest GUIDs, generate materials + staticmeshes."""

    # Load lookup files
    lookup_path = os.path.join(source_dir, "unity_lookup.json")
    with open(lookup_path, "r", encoding="utf-8") as f:
        unity_lookup = json.load(f)

    map_path = os.path.join(source_dir, "copy_map.json")
    with open(map_path, "r", encoding="utf-8") as f:
        copy_map = json.load(f)

    client = EditorClient(editor_url)
    await client.connect()

    try:
        # ── Phase 3: Import raw assets ──
        print("[Phase 3] Triggering asset database refresh...")
        result = await client.send("POST /api/assets/import-refresh", timeout=300)
        print(f"  Refresh: {result.get('newAssets', 0)} new assets (total: {result.get('assetsAfter', '?')})")

        # Harvest GUIDs: batch lookup all copied paths
        print("  Looking up Freefall GUIDs for copied assets...")
        freefall_paths = list(copy_map.values())

        # Batch in chunks of 200 to avoid message size issues
        freefall_lookup = {}  # unity_guid → freefall_guid
        path_to_freefall_guid = {}  # freefall_rel_path → freefall_guid

        for i in range(0, len(freefall_paths), 200):
            chunk = freefall_paths[i:i + 200]
            result = await client.send("POST /api/assets/lookup", {"paths": chunk})
            path_to_freefall_guid.update(result.get("results", {}))
            missed = result.get("missingCount", 0)
            if missed:
                print(f"  Warning: {missed} paths not found in chunk {i // 200 + 1}")

        # Build unity_guid → freefall_guid mapping
        for unity_path, freefall_path in copy_map.items():
            freefall_guid = path_to_freefall_guid.get(freefall_path)
            if freefall_guid:
                # Find the unity guid for this path
                for guid, info in {**unity_lookup["textures"], **unity_lookup["models"]}.items():
                    if info["path"] == unity_path:
                        freefall_lookup[guid] = freefall_guid
                        break

        print(f"  Mapped {len(freefall_lookup)} unity GUIDs → freefall GUIDs")

        # ── Phase 4: Generate Materials ──
        print("[Phase 4] Creating Freefall materials...")
        material_guid_map = {}  # unity_material_guid → freefall_material_guid
        mat_created = 0
        mat_skipped = 0

        for unity_mat_guid, mat_info in unity_lookup["materials"].items():
            name = mat_info["name"]
            unity_textures = mat_info.get("textures", {})

            # Map Unity texture slots → Freefall properties
            properties = {}
            for unity_slot, unity_tex_guid in unity_textures.items():
                freefall_slot = TEXTURE_SLOT_MAP.get(unity_slot)
                freefall_tex_guid = freefall_lookup.get(unity_tex_guid)

                if freefall_slot and freefall_tex_guid:
                    # Don't overwrite Albedo if already set (BaseColorMap takes priority over IDMap)
                    if freefall_slot == "Albedo" and "Albedo" in properties and unity_slot == "_IDMap":
                        continue
                    properties[freefall_slot] = freefall_tex_guid

            if not properties:
                mat_skipped += 1
                continue

            mat_path = f"{pack_name}/Materials/{name}.mat"
            try:
                result = await client.send("POST /api/asset/create", {
                    "type": "Material",
                    "name": name,
                    "path": mat_path,
                    "properties": properties,
                })
                material_guid_map[unity_mat_guid] = result.get("guid")
                mat_created += 1
            except Exception as e:
                print(f"  Error creating material {name}: {e}")

        print(f"  Created: {mat_created} materials (skipped {mat_skipped} with no resolvable textures)")

        # Refresh so materials get their GUIDs registered
        if mat_created > 0:
            await client.send("POST /api/assets/import-refresh")

        # ── Phase 5: Generate StaticMeshes ──
        print("[Phase 5] Creating Freefall staticmeshes...")
        sm_created = 0
        sm_skipped = 0

        for unity_prefab_guid, prefab_info in unity_lookup["prefabs"].items():
            if not prefab_info.get("is_simple", False):
                sm_skipped += 1
                continue

            name = prefab_info["name"]
            mesh_unity_guid = prefab_info.get("mesh_guid")
            mat_unity_guids = prefab_info.get("material_guids", [])

            # Resolve mesh GUID
            mesh_freefall_guid = freefall_lookup.get(mesh_unity_guid)
            if not mesh_freefall_guid:
                sm_skipped += 1
                continue

            # Resolve material GUIDs
            mesh_parts = []
            for i, mat_unity_guid in enumerate(mat_unity_guids):
                mat_freefall_guid = material_guid_map.get(mat_unity_guid)
                if mat_freefall_guid:
                    mesh_parts.append({
                        "material": mat_freefall_guid,
                        "index": i,
                        "collision": False,
                    })

            if not mesh_parts:
                sm_skipped += 1
                continue

            sm_path = f"{pack_name}/Prefabs/{name}.staticmesh"
            try:
                await client.send("POST /api/asset/create", {
                    "type": "StaticMesh",
                    "name": name,
                    "path": sm_path,
                    "properties": {
                        "mesh": mesh_freefall_guid,
                        "meshParts": mesh_parts,
                    },
                })
                sm_created += 1
            except Exception as e:
                print(f"  Error creating staticmesh {name}: {e}")

        print(f"  Created: {sm_created} staticmeshes (skipped {sm_skipped})")

        # ── Phase 6: Final Import ──
        if sm_created > 0:
            print("[Phase 6] Final import refresh...")
            result = await client.send("POST /api/assets/import-refresh", timeout=300)
            print(f"  Done: {result.get('assetsAfter', '?')} total assets")

        # Save the GUID mapping for reference
        guid_map_path = os.path.join(source_dir, "freefall_lookup.json")
        with open(guid_map_path, "w", encoding="utf-8") as f:
            json.dump({
                "unity_to_freefall": freefall_lookup,
                "material_map": material_guid_map,
                "stats": {
                    "textures_mapped": len([g for g in freefall_lookup.values()]),
                    "materials_created": mat_created,
                    "staticmeshes_created": sm_created,
                }
            }, f, indent=2)
        print(f"  Saved: {guid_map_path}")

    finally:
        await client.close()

    print("\n[Done] Import pipeline complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unity → Freefall Asset Import Pipeline")
    sub = parser.add_subparsers(dest="command")

    # discover
    p_disc = sub.add_parser("discover", help="Scan Unity source and build lookup table")
    p_disc.add_argument("--source", required=True, help="Unity source directory")
    p_disc.add_argument("--output", help="Output path for unity_lookup.json")

    # copy
    p_copy = sub.add_parser("copy", help="Copy raw assets into Freefall project")
    p_copy.add_argument("--source", required=True, help="Unity source directory")
    p_copy.add_argument("--target", required=True, help="Freefall Assets/ directory")
    p_copy.add_argument("--pack", required=True, help="Pack name (subfolder in Assets/)")

    # import (via editor WebSocket)
    p_imp = sub.add_parser("import", help="Drive editor to import and generate assets")
    p_imp.add_argument("--source", required=True, help="Unity source directory (for lookup files)")
    p_imp.add_argument("--target", required=True, help="Freefall Assets/ directory")
    p_imp.add_argument("--pack", required=True, help="Pack name")
    p_imp.add_argument("--editor-url", default="ws://localhost:21721/ws", help="Editor WS URL")

    # run-all
    p_all = sub.add_parser("run-all", help="Execute all phases")
    p_all.add_argument("--source", required=True, help="Unity source directory")
    p_all.add_argument("--target", required=True, help="Freefall Assets/ directory")
    p_all.add_argument("--pack", required=True, help="Pack name")
    p_all.add_argument("--editor-url", default="ws://localhost:21721/ws", help="Editor WS URL")

    args = parser.parse_args()

    if args.command == "discover":
        discover(args.source, args.output)

    elif args.command == "copy":
        discover(args.source)  # Ensure lookup exists
        copy_assets(args.source, args.target, args.pack)

    elif args.command == "import":
        asyncio.run(import_and_generate(args.source, args.target, args.pack, args.editor_url))

    elif args.command == "run-all":
        print("=" * 60)
        print("Unity → Freefall Asset Import Pipeline")
        print("=" * 60)
        lookup = discover(args.source)
        print()
        copy_map = copy_assets(args.source, args.target, args.pack, lookup)
        print()
        print("Connecting to editor for import phases...")
        asyncio.run(import_and_generate(args.source, args.target, args.pack, args.editor_url))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
