import os
import math
import bpy
import numpy as np
from pathlib import Path
import json 
import time

from utils.utils import enable_cuda_devices
from utils.utils import get_3x4_RT_matrix_from_blender
from utils.supress_logging import stdout_redirected

from utils.light_enum import SunPosition,SpotlightsPosition
import utils.blender_utils as blender_utils


SUN_FIXED_POSITION = (0,3,10)

def render(obj_path: str | Path,
           root_output_folder:str | Path,
           resolution = 400,
           scale = 1,
           num_views = 5,
           sun_position = SunPosition.FIXED,
           light_situtation = SpotlightsPosition.THREESPOTLIGHTS_SAMPLED,
           render_depth:bool = True,
           render_albedo: bool = True,
           render_normal:bool = True,
           export_blender_scene:bool = True
           ) -> None:
    
    scene_reconstruction_dict = {
        'obj_path' : obj_path,
        'root_output_folder' : root_output_folder,
        'resolution' : resolution,
        'scale' : scale,
    }

    start_t = time.time()
    img_folder = os.path.join(os.path.abspath(root_output_folder))

    os.makedirs(img_folder, exist_ok=True)

    # Create a new scene
    scene = bpy.data.scenes.new(name="MyNewScene")

    # Link the new scene to the context
    bpy.context.window.scene = scene

    # Set up rendering in the new scene
    render = scene.render

    render.engine = "CYCLES"
    render.image_settings.color_mode = 'RGBA'
    render.image_settings.file_format = "PNG"
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100
    scene.cycles.filter_width = 0.01
    scene.render.film_transparent = True

    scene.cycles.device = 'GPU'
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True

    enable_cuda_devices()


    color_depth = '16' # Important for albedo and depth

    if render_depth or render_albedo or render_normal:
        scene.use_nodes = True
        active_view_layer = bpy.context.view_layer
        if not active_view_layer:
            print("View Layer not found.")
            raise Exception("View layer not found, neither depth, albdedo nor normal pass can be enabled")

        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links

        # Clear default nodes
        for n in nodes:
            nodes.remove(n)

        # Create input render layer node
        render_layers = nodes.new('CompositorNodeRLayers')

    if render_depth:
        active_view_layer.use_pass_z = True
        print("Depth pass enabled")
        # Create depth output nodes
        depth_file_output = nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.label = 'Depth Output'
        depth_file_output.base_path = '/'
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = "PNG"
        depth_file_output.format.color_depth = color_depth
        depth_file_output.format.color_mode = "BW"

        # Remap as other types can not represent the full range of depth.
        map = nodes.new(type="CompositorNodeMapValue")

        # These values parametrize the resulting depth map
        """
        Quick explanation of the individual values:
        Offset = -1 says that only objects that are at least 1 blender unit away from the camera are recognoized for the depth. Every
        closer object is ignored

        Size = 0.7 specifies the slope of the linear function that represents the conversion from blender depth units to depth map units.

        In the case of size = 1, it takes a depth difference of 1 blender unit to go from 0 to 255 (or 65.536, depending on 8 or 16 bit) in the depth map.
        So in the case of offset = -1 and size = 1, the depth map will be completely black for everything closer than 1. Objects that are
        between 1 and 2 blender units away will be represented by a gradient from black to white. Everything further away than 2 blender units will be
        completely white.

        In the case of size = 0.5, it takes a depth difference of 2 blender units to go from 0 to 255 (or 65.536, depending on 8 or 16 bit) in the depth map.
        So in the case of offset = -1 and size = 2, the depth map will be completely black for everything closer than 1. Objects that are
        between 1 and 3 blender units away will be represented by a gradient from black to white. Everything further away than 3 blender units will be
        completely white.

        The parameter min lets you specify if you want another distance-delay in the range where the depth map is actually neither black nor white.
        """
        
        map.offset = [-1]
        map.size = [0.7]
        map.use_min = True
        map.min = [0]

        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])
    
    if render_normal:
        active_view_layer.use_pass_normal = True
        print("Normal pass enabled.")
        # Create normal output nodes
        scale_node = nodes.new(type="CompositorNodeMixRGB")
        scale_node.blend_type = 'MULTIPLY'
        scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
        links.new(render_layers.outputs['Normal'], scale_node.inputs[1])

        bias_node = nodes.new(type="CompositorNodeMixRGB")
        bias_node.blend_type = 'ADD'
        bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
        links.new(scale_node.outputs[0], bias_node.inputs[1])

        normal_file_output = nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        normal_file_output.base_path = '/'
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = "PNG"
        links.new(bias_node.outputs[0], normal_file_output.inputs[0])
        
    if render_albedo:
        active_view_layer.use_pass_diffuse_color = True
        print("Albedo pass enabled")
        # Create albedo output nodes
        alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])

        albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
        albedo_file_output.label = 'Albedo Output'
        albedo_file_output.base_path = '/'
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = "PNG"
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = color_depth
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])


    time_import = time.time()
    with stdout_redirected():
        bpy.ops.import_scene.obj(filepath=obj_path, use_edges=False, use_smooth_groups=False, split_mode='OFF')
    print("importing took {}s".format(time.time()-time_import))

    # Get the last added object (assuming the new object is the most recently added one)
    new_object = bpy.context.scene.objects[-1]

    # Set the name for the newly imported object
    new_object.name = "shape"

    bpy.context.view_layer.objects.active = new_object

    # normalize bounding box
    # Get the dimensions of the bounding box
    bbox_dimensions = new_object.dimensions

    # Find the maximum dimension
    max_dimension = max(bbox_dimensions)

    # Calculate the scale factor to fit within a 1x1x1 cube -> scale such that proportions stay the same
    scale_factor = 1.0 / (max_dimension)

    # # Scale the object uniformly
    new_object.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bbox_dimensions = new_object.dimensions
    max_dimension = max(bbox_dimensions)

    # # Set the object's origin to its center to center the object in the scene-> This is buggy and might not work for every object
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
    new_object.location = (0, 0, 0)
    bpy.context.view_layer.update()

    obj = new_object
    material_slot = obj.active_material

    
    # make object shiny to allow for non-Lambertian effects
    # Create a new material if none exists
    material_slot = obj.material_slots[0] if obj.material_slots else None

    if material_slot is None:
        # Create a material
        material = bpy.data.materials.new(name="MyMaterial")

        # Assign the material to the object
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
    else:
        material = material_slot.material

    # set up material to make it shiny
    material.use_nodes = False  # Disable node-based material for simplicity

    # Set specular reflection intensity
    material.specular_intensity = 1.0  # Adjust as needed

    # Set glossiness (shininess)
    material.roughness = 0.2  # Adjust this value as needed

    # Optionally, set the material to use the Principled BSDF shader for more control
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs["Specular"].default_value = 1.0 # 1
    material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.1  # Adjust as needed , 0.1


    # set up lightning
    sun_lamp = bpy.data.lights.new(name="Sun", type='SUN')
    sun_obj = bpy.data.objects.new(name="SunObject", object_data=sun_lamp)
    bpy.context.scene.collection.objects.link(sun_obj)

    # Set the strength of the Sun lamp
    sun_lamp.energy = 30.0  # Adjust the strength as needed
    scene_reconstruction_dict['sun_energy'] = sun_lamp.energy

    # Set the location of the Sun lamp
    if sun_position == SunPosition.SAMPLE_ONCE:
        sun_obj.rotation_euler = blender_utils.carthesian_to_euler(blender_utils.sample_hemisphere_around_object(new_object))
    elif sun_position == SunPosition.FIXED:
        sun_obj.rotation_euler = blender_utils.carthesian_to_euler(SUN_FIXED_POSITION)
        # sun_obj.location = SUN_FIXED_POSITION
    scene_reconstruction_dict['sun_rotation_euler'] = tuple(sun_obj.rotation_euler)

    # important if we want to use ambient light
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new(name="World")

    bpy.context.scene.world.use_nodes = False  # Disable nodes for simplicity
    if light_situtation == SpotlightsPosition.AMBIENT_LIGHT:
        # ambient light and ONE ADDITIONAL spotlight
        spot_object = blender_utils.create_new_spot_light()
        blender_utils.reposition_light_source(spot_object,[2,2,2])
        bpy.context.scene.world.color = (0.35, 0.35, 0.35)  # Adjust the RGB values as needed
        scene_reconstruction_dict = blender_utils.put_spotlight_info_in_dict(scene_reconstruction_dict,"spot_in_ambient_scene",spot_object)
        scene_reconstruction_dict['ambient_color'] = bpy.context.scene.world.color
    else:
        pass
        # bpy.context.scene.world.color = (0.1, 0.1, 0.1)  # Adjust the RGB values as needed
    bpy.context.scene.world.color = (0, 0, 0)  # No ambient lightning, adjust the RGB values as needed
    if light_situtation == SpotlightsPosition.THREESPOTLIGHTS:
        spot_objects = []
        spot_object = blender_utils.create_new_spot_light()
        blender_utils.reposition_light_source(spot_object,[-2,2,3])
        spot_objects.append(spot_object)

        spot_object = blender_utils.create_new_spot_light()
        blender_utils.reposition_light_source(spot_object,[2,2,3])
        spot_objects.append(spot_object)

        spot_object = blender_utils.create_new_spot_light()
        blender_utils.reposition_light_source(spot_object,[0,-2,3])
        spot_objects.append(spot_object)
    
    if light_situtation == SpotlightsPosition.THREESPOTLIGHTS_SAMPLED:
        spot_objects = []
        spot_object = blender_utils.create_new_spot_light()
        theta,phi,new_location = list(blender_utils.sample_hemisphere_around_object(new_object))
        blender_utils.reposition_light_source(spot_object,new_location)
        spot_objects.append(spot_object)
        scene_reconstruction_dict = blender_utils.put_spotlight_info_in_dict(scene_reconstruction_dict,"spotsampled1",spot_object)

        spot_object = blender_utils.create_new_spot_light()
        theta2,phi2 = theta,phi
        while blender_utils.angular_distance(phi,theta,phi2,theta2) < 60:
            theta2,phi2,new_location2 = list(blender_utils.sample_hemisphere_around_object(new_object))
        blender_utils.reposition_light_source(spot_object,new_location2)
        spot_objects.append(spot_object)
        scene_reconstruction_dict = blender_utils.put_spotlight_info_in_dict(scene_reconstruction_dict,"spotsampled2",spot_object)

        spot_object = blender_utils.create_new_spot_light()
        theta3,phi3 = theta,phi
        while blender_utils.angular_distance(phi,theta,phi3,theta3) < 30 or blender_utils.angular_distance(phi2,theta2,phi3,theta3) < 30:
            theta3,phi3,new_location3 = list(blender_utils.sample_hemisphere_around_object(new_object))
            
        blender_utils.reposition_light_source(spot_object,new_location3)
        spot_objects.append(spot_object)
        scene_reconstruction_dict = blender_utils.put_spotlight_info_in_dict(scene_reconstruction_dict,"spotsampled3",spot_object)
    
    # Place camera
    cam_data = bpy.data.cameras.new(name="mycam")
    cam = bpy.data.objects.new(name="MyCameraObject", object_data=cam_data)
    cam.location = (0,2.3,0) # THIS CONTROLS DISTANCE, we later only rotate this cam but the distance stays the same
    scene_reconstruction_dict['cam_loc'] = tuple(cam.location)
    cam.data.lens = 35
    scene_reconstruction_dict['cam_lens'] = cam.data.lens
    cam.data.sensor_width = 32
    scene_reconstruction_dict['cam_sensor_width'] = cam.data.sensor_width

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'

    cam_empty = bpy.data.objects.new("Empty", None)

    cam_empty.location = (0, 0, 0)
    cam.parent = cam_empty # this means the camera will follow the movements and transformations of the cam_empty object


    scene.collection.objects.link(cam_empty) # cam_empty becomes visible as it is added to this scenes collection
    bpy.context.view_layer.objects.active = cam_empty # make it the active object, the object certain operations and modifications will apply to
    cam_constraint.target = cam_empty
    scene.camera = cam

    rotation_angle_list = np.linspace(0,360,num_views) # one rotations
    elevation_angle_list = np.linspace(0,60,num_views)

    aabb = [[-scale/2,-scale/2,-scale/2],
                [scale/2,scale/2,scale/2]]

    
    # creation of the transform.json
    to_export = {
        'camera_angle_x': bpy.data.cameras[1].angle_x,
        'sensor_width' : cam.data.sensor_width,
        "aabb": aabb,
    }

    # render the views
    frames = []
    for i in range(num_views):
        cam_empty.rotation_euler[2] = math.radians(rotation_angle_list[i])
        cam_empty.rotation_euler[0] = math.radians(elevation_angle_list[i])

        if sun_position == SunPosition.SAMPLE_EVERY_VIEW:
            sun_obj.rotation_euler = blender_utils.carthesian_to_euler(blender_utils.sample_hemisphere_around_object(new_object))
        
        render_file_path = os.path.join(img_folder,'%04d' % (i))

        scene.render.filepath = render_file_path+".png"
        # print(f"Rendered to {render_file_path+'.png'}")
        if render_depth:
            depth_file_output.file_slots[0].path = render_file_path + "_depth"
        if render_normal:
            normal_file_output.file_slots[0].path = render_file_path + "_normal"
        if render_albedo:
            albedo_file_output.file_slots[0].path = render_file_path + "_albedo"
            
        with stdout_redirected():
            bpy.ops.render.render(write_still=True)
        # might not need it, but just in case cam is not updated correctly
        bpy.context.view_layer.update()


        # save camera location information
        rt = get_3x4_RT_matrix_from_blender(cam)
        pos, rt, scale = cam.matrix_world.decompose() # matrix_world is the world matrix of the camera. The world matrix is a transformation matrix
                                                        # that transforms coordinates from the camera's local space into world space
                                                        # decompose returns a tuple of location, rotation, and scale 
                                                        # matrix refers to transformation matrix of a camera object in world space. Represents cameras position
                                                        # orientation and scale in 3D space of blender scene
                                                        # contains all the information necessary to position and orient the camera
        

        rt = rt.to_matrix()
        matrix = []
        for ii in range(3):
            a = []
            for jj in range(3):
                a.append(rt[ii][jj])
            a.append(pos[ii])
            matrix.append(a)
        matrix.append([0,0,0,1])

        to_add = {\
            "file_path":f'./{str(i).zfill(4)}',
            "transform_matrix":matrix
        }
        frames.append(to_add)

    if len(frames) > 0:
        to_export['frames'] = frames
        
        with open(f'{img_folder}/transforms.json', 'w') as f:
            json.dump(to_export, f,indent=4)    
    else:
        # print("len of frames is zero, no json dumping")
        pass
    
    with open(f'{img_folder}/scene_info.json', 'w') as f:
        json.dump(scene_reconstruction_dict, f,indent=4)    
    if export_blender_scene:
        if Path(img_folder,"scene.blend").exists():
            os.remove(Path(img_folder,"scene.blend"))
        bpy.ops.wm.save_as_mainfile(filepath=f"{img_folder}/scene.blend")

    print(f"{time.time()-start_t}s to load and render {obj_path}")
    
# render green airplane
if __name__ == "__main__":
    render("green_airplane_model/model_normalized.obj",
            'example_render') 
    