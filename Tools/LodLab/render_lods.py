# Visual check of generated LODs: render the OBJs LodLab writes with --obj.
# blender -b --python render_lods.py -- <obj_dir> <mesh> <out_dir> <albedo> <normal> <screen sizes LOD1,LOD2,...>
# (screen sizes as LodLab prints them; albedo/normal = the mesh's textures, so it looks as in game)
# Renders every <mesh>_LOD<k>.obj:
#   <mesh>_LOD<k>_tex.png   512 px, textured (albedo + normal map), EEVEE
#   <mesh>_LOD<k>_wire.png  512 px, clay + wireframe, Workbench
#   <mesh>_LOD<k>_at<j>.png at the pixel size the mesh has on a 1080p screen where LOD j switches in
#                           (bounding sphere diameter = screen size * 1080 px), for k = j-1 and j
import bpy, sys, os, math
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
obj_dir, mesh, out_dir, albedo, normal = argv[:5]
sizes = [float(x) for x in argv[5].split(",")] if len(argv) > 5 and argv[5] else []
os.makedirs(out_dir, exist_ok=True)

lods = []
k = 0
while os.path.exists(os.path.join(obj_dir, f"{mesh}_LOD{k}.obj")):
    lods.append(os.path.join(obj_dir, f"{mesh}_LOD{k}.obj"))
    k += 1

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.view_settings.view_transform = 'Standard'

world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.32, 0.33, 0.35, 1.0)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.8

sun_data = bpy.data.lights.new("Sun", 'SUN')
sun_data.energy = 3.5
sun = bpy.data.objects.new("Sun", sun_data)
scene.collection.objects.link(sun)
sun.rotation_euler = (math.radians(50), 0, math.radians(-40))

mat = bpy.data.materials.new("Rock")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
bsdf.inputs["Roughness"].default_value = 0.85
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load(albedo)
nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
ntex = nt.nodes.new("ShaderNodeTexImage")
ntex.image = bpy.data.images.load(normal)
ntex.image.colorspace_settings.name = 'Non-Color'
nmap = nt.nodes.new("ShaderNodeNormalMap")
nt.links.new(ntex.outputs["Color"], nmap.inputs["Color"])
nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

clay = bpy.data.materials.new("Clay")
clay.diffuse_color = (0.6, 0.6, 0.6, 1)

cam_data = bpy.data.cameras.new("Cam")
cam_data.type = 'ORTHO'
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam


def import_obj(path):
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path, forward_axis='NEGATIVE_Z', up_axis='Y')
    return [o for o in bpy.data.objects if o not in before and o.type == 'MESH']


# Frame on LOD0's bounding sphere (box half-diagonal, as the engine measures screen size)
lod0 = import_obj(lods[0])
lo = Vector((1e9, 1e9, 1e9)); hi = Vector((-1e9, -1e9, -1e9))
for o in lod0:
    for v in o.data.vertices:
        w = o.matrix_world @ v.co
        lo = Vector(map(min, lo, w)); hi = Vector(map(max, hi, w))
for o in lod0:
    bpy.data.objects.remove(o)
center = (lo + hi) * 0.5
radius = (hi - center).length

az, el = math.radians(35), math.radians(18)
direction = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
cam.location = center + direction * radius * 4
cam.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()
cam_data.ortho_scale = radius * 2
cam_data.clip_end = radius * 10


def render(objs, path, px, engine):
    scene.render.engine = engine
    scene.render.resolution_x = scene.render.resolution_y = px
    scene.render.resolution_percentage = 100
    scene.render.filepath = path
    if engine == 'BLENDER_EEVEE':
        scene.eevee.taa_render_samples = 32
    bpy.ops.render.render(write_still=True)


for k, path in enumerate(lods):
    objs = import_obj(path)
    for o in objs:
        o.data.materials.clear()
        o.data.materials.append(mat)

    render(objs, os.path.join(out_dir, f"{mesh}_LOD{k}_tex.png"), 512, 'BLENDER_EEVEE')
    for j, s in enumerate(sizes, start=1):
        if k == j - 1 or k == j:
            render(objs, os.path.join(out_dir, f"{mesh}_LOD{k}_at{j}.png"), max(16, round(s * 1080)), 'BLENDER_EEVEE')

    # Clay + wireframe
    wires = []
    for o in objs:
        o.data.materials.clear()
        o.data.materials.append(clay)
        o.color = (0.62, 0.62, 0.6, 1)
        w = o.copy()
        w.data = o.data.copy()
        scene.collection.objects.link(w)
        mod = w.modifiers.new("Wire", 'WIREFRAME')
        mod.thickness = radius * 0.004
        mod.use_relative_offset = False
        mod.use_even_offset = False  # even offset spikes on the packs' degenerate triangles
        w.color = (0.08, 0.08, 0.1, 1)
        wires.append(w)
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'OBJECT'
    render(objs, os.path.join(out_dir, f"{mesh}_LOD{k}_wire.png"), 512, 'BLENDER_WORKBENCH')

    for o in objs + wires:
        bpy.data.objects.remove(o)

print("RENDERED", mesh, len(lods), "LODs")
