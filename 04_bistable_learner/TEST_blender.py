import bpy

# deletes everything
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for m in bpy.data.materials:
    bpy.data.materials.remove(m)
for l in bpy.data.lights:
    bpy.data.lights.remove(l)
for c in bpy.data.cameras:
    bpy.data.cameras.remove(c)


# turn on bloom
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.eevee.use_bloom = True

# add camera
bpy.ops.object.camera_add(location=(5, -5, 3), rotation=(1.32573, 0.014212, 1.0179))
bpy.context.scene.camera = bpy.data.objects['Camera']

# add material
mat_name = 'EmiMat'
bpy.ops.material.new()
bpy.data.materials[-1].name = mat_name
bpy.data.materials[mat_name].node_tree.nodes.clear()
bpy.data.materials[mat_name].node_tree.nodes.new("ShaderNodeEmission")
bpy.data.materials[mat_name].node_tree.nodes["Emission"].inputs["Color"].default_value = (0.151736, 0.0997155, 1, 1)
bpy.data.materials[mat_name].node_tree.nodes["Emission"].inputs["Strength"].default_value = 5.
bpy.data.materials[mat_name].node_tree.nodes.new("ShaderNodeOutputMaterial")

# link nodes
links = bpy.data.materials[mat_name].node_tree.links
links.new(
    bpy.data.materials[mat_name].node_tree.nodes["Emission"].outputs[0], 
    bpy.data.materials[mat_name].node_tree.nodes["Material Output"].inputs[0])

# add sphere
bpy.ops.mesh.primitive_uv_sphere_add(location=(0,0,0))
bpy.ops.object.material_slot_add()
mat = bpy.data.materials.get(mat_name)
ob = bpy.context.active_object
ob.data.materials[0] = mat


bpy.data.scenes['Scene'].render.filepath = '/home/miotto/Desktop/blentest.jpg'
bpy.ops.render.render( write_still=True ) 

##run with: blender --background --python TEST_blender.py