
import pickle, os, sys
import bpy
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)
from blender_neuron import Neuron

# turn on bloom
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.eevee.use_bloom = True

# deletes lights
for l in bpy.data.lights:
    bpy.data.lights.remove(l)

frames = pickle.load(open('../../results/blendernet/animation.data', 'rb'))

for f, frame in enumerate(frames):
    frame_str = 'frame_' + str(f).rjust(6, '0')
    print(frame_str)

    # delete everything
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for m in bpy.data.materials:
        bpy.data.materials.remove(m)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)

    # add camera
    bpy.ops.object.camera_add(location=(0., -67., 34.), rotation=(1.1075, 0., 0.))
    bpy.context.scene.camera = bpy.data.objects['Camera']

    for n_count, (n_id, neuron) in enumerate(frame.items()):
        neuron_str = 'neuron_' + str(n_id).rjust(4, '0')
        print(neuron_str)

        #DEBUG:
        #if n_count > 10:
        #    break

        # add material
        mat_name = frame_str + '_' + neuron_str
        bpy.ops.material.new()
        bpy.data.materials[-1].name = mat_name
        bpy.data.materials[mat_name].node_tree.nodes.clear()
        bpy.data.materials[mat_name].node_tree.nodes.new("ShaderNodeEmission")
        bpy.data.materials[mat_name].node_tree.nodes["Emission"].inputs["Color"].default_value = neuron.color
        bpy.data.materials[mat_name].node_tree.nodes["Emission"].inputs["Strength"].default_value = neuron.strength
        bpy.data.materials[mat_name].node_tree.nodes.new("ShaderNodeOutputMaterial")
        links = bpy.data.materials[mat_name].node_tree.links
        links.new(
        bpy.data.materials[mat_name].node_tree.nodes["Emission"].outputs[0], 
        bpy.data.materials[mat_name].node_tree.nodes["Material Output"].inputs[0])

        # add sphere
        bpy.ops.mesh.primitive_uv_sphere_add(radius=neuron.radius, location=neuron.position)
        bpy.ops.object.material_slot_add()
        mat = bpy.data.materials.get(mat_name)
        ob = bpy.context.active_object
        ob.data.materials[0] = mat

    print('Rendering')
    bpy.data.scenes['Scene'].render.filepath = '/home/miotto/Dropbox/UniFreiburg/Thesis/code/results/blendernet/' + frame_str + '.jpg'
    bpy.ops.render.render( write_still=True ) 

##run with: blender --background --python TEST_blender.py