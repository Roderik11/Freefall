# Freefall — Project Roadmap

> **Spiritual successor to Darkfall**, with the physics combat of **Dark Messiah of Might and Magic**.
> Open-world sandbox PvP MMO. Physics-driven combat. High skill ceiling. Custom D3D12 engine in C#.

---

## Vision

A seamless open-world MMO where every fight is shaped by physics, positioning, and terrain.
Knockbacks, knockups, pulls, ragdolls — on players AND mobs. No scripted encounters, just emergent combat.

---

## Current State ✅

- [x] **D3D12 GPU-driven rendering** — indirect draws, instanced batching, persistent transform buffers
- [x] **Cascaded Shadow Mapping** — PSSM with stabilized cascades
- [x] **Hi-Z occlusion culling** — GPU-driven with feedback loop fix
- [x] **Terrain rendering** — GPU quadtree tessellation with splatmap layers
- [x] **Skeletal animation** — FBX import, bone weights, skinned mesh rendering
- [x] **PhysX integration** — capsule controller, precooked collision meshes, terrain collision
- [x] **Asset pipeline** — import → cache → load with GUID tracking, meta files, compound assets
- [x] **Scene loading** — YAML serialization, streaming entity spawn
- [x] **Third-person character controller** — movement, camera, foot IK
- [x] **Audio system** — 3D positional audio sources
- [x] **Editor (WIP)** — WinForms, scene hierarchy, basic inspector, viewport

---

## Phase 1: Editor & Content Pipeline
*Build the tools to build the game*

- [x] **Asset Inspector** — view/edit any asset type, importer-driven inspection targets
- [x] **Material Inspector** — custom inspector showing effect name, texture slots, reflected properties
- [x] **StaticMesh Inspector** — GenericInspector with LODs, mesh parts, LODGroup, collision settings
- [x] **ModelImporter config** — all Assimp flags as editable fields, compound import with GUID migration
- [x] **Asset Browser labels** — CleanImporterType-based display, subasset cards for compound assets
- [x] **ValueProviders** — MeshPartProvider dropdown for mesh part indices → names
- [ ] **Asset Previews** — thumbnail/live preview for textures, materials, meshes in the inspector
- [ ] **Asset creation** — create new `.mat`, `.staticmesh` files from editor
- [ ] **Asset saving** — serialize edited assets back to disk, trigger reimport
- [ ] **Scene saving** — save entity transforms, component changes
- [ ] **Single-pass shadow cascades** — 4 cascades in one draw call (performance)

---

## Phase 2: Animation System
*The backbone of combat feel*

- [ ] **Animation Editor** — preview clips, scrub timeline, set events
- [ ] **Mass animation import** — hundreds of locomotion + combat animations
- [ ] **Animation State Machine** — states, transitions, conditions
- [ ] **Blend Trees** — 1D/2D blending (walk/run speed, direction)
- [ ] **Animation Layers & Masking** — upper/lower body independence
  - Lower body: locomotion (run, strafe, jump)
  - Upper body: combat (swing, block, cast, aim)
- [ ] **Animation Events** — trigger hitboxes, VFX, SFX at specific frames
- [ ] **Root Motion** — optional root motion for specific attacks/dodges
- [ ] **IK refinements** — weapon grip IK, look-at IK, foot IK improvements

---

## Phase 3: Physics Combat
*Darkfall meets Dark Messiah*

- [ ] **Ragdoll system** — PhysX joint constraints on skeleton, anim→ragdoll transitions
- [ ] **Partial ragdoll** — upper body ragdoll while legs keep momentum
- [ ] **Force-based abilities** — knockback, knockup, pull, push impulse vectors
- [ ] **Bunny hopping** — momentum-based movement, air control, skill-expressive traversal
- [ ] **Hitbox system** — per-bone hitboxes, damage zones (head, torso, limbs)
- [ ] **Melee combat** — directional swings, blocking, parrying, weapon arcs
- [ ] **Ranged combat** — projectile physics, arrow drop, spell travel
- [ ] **Environmental interaction** — kick into spikes, knock off ledges, ice on floors
- [ ] **Physics on mobs** — all knockback/ragdoll mechanics work on NPCs too

---

## Phase 4: Game Mechanics
*From tech demo to playable game*

- [ ] **Character stats & skills** — attribute system, skill progression
- [ ] **Inventory & equipment** — gear slots, weapon types, armor
- [ ] **Mob AI** — basic combat AI, aggro, pathfinding, group behavior
- [ ] **Loot system** — full-loot PvP drops, mob loot tables
- [ ] **Spell system** — schools of magic, mana, cooldowns, projectile spells
- [ ] **Death & respawn** — ragdoll on death, lootable corpse, respawn mechanics
- [ ] **Basic UI/HUD** — health, mana, hotbar, target info

---

## Phase 5: World & Multiplayer
*Making it an MMO*

- [ ] **Networking foundation** — [LiteNetLib](https://github.com/RevenantX/LiteNetLib) (UDP, multi-channel, .NET 8 optimized)
  - Server-authoritative physics (PhysX on server, client-side prediction with reconciliation)
  - Multi-channel strategy: unreliable (position), reliable ordered (combat), reliable unordered (inventory/loot)
  - Delta compression, position quantization, automated replication system
- [ ] **World streaming** — seamless zone loading, LOD management at scale
- [ ] **Player-to-player combat** — networked physics, server-validated hit detection
- [ ] **Clan system** — guilds, territory, politics
- [ ] **Building/siege mechanics** — placeable structures, destructible walls
- [ ] **World building** — terrain tools, prop placement, dungeon design
- [ ] **NPC population** — towns, vendors, quest givers, patrol routes

---

## Vertical Slice Target 🎯

A playable demo showcasing:
- Third-person character on the island
- Physics-based melee + ranged combat vs mobs
- Knockback, ragdoll, environmental kills
- Multiple weapon types with distinct animations
- Basic loot from defeated mobs
- Polish: lighting, shadows, VFX, sound

---

*Last updated: 2026-02-27*
