import bpy
import math
import uuid
from mathutils import Vector
import random

def angular_distance(phi1, theta1, phi2, theta2):
    # Calculate angular distance in radians
    cos_angle = math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(theta1 - theta2)
    cos_angle = max(min(cos_angle, 1), -1)
    return math.degrees(math.acos(cos_angle))

def euclidian_dist(point1,point2):
    squared_distance = sum((x - y) ** 2 for x, y in zip(point1, point2))
    distance = math.sqrt(squared_distance)
    return distance

def put_spotlight_info_in_dict(thisdict,spotname,spot_object):
    thisdict[f"{spotname}_rotation_euler"] = tuple(spot_object.rotation_euler)
    thisdict[f"{spotname}_location"] = tuple(spot_object.location)
    thisdict[f"{spotname}_energy"] = spot_object.data.energy
    thisdict[f"{spotname}_distance"] = spot_object.data.distance
    thisdict[f"{spotname}_spot_size"] = spot_object.data.spot_size
    thisdict[f"{spotname}_spot_blend"] = spot_object.data.spot_blend
    return thisdict


def carthesian_to_euler(direction):
    # Normalize the direction vector
    direction = Vector(direction)
    direction.normalize()

    # Calculate azimuth and elevation angles
    azimuth = math.atan2(direction.y, direction.x)
    elevation = math.asin(direction.z)
    return (elevation,0,-azimuth)

def sample_point_on_hemisphere(center_point,hemisphere_radius):
    # Sample points on the hemisphere
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, 0.5 * math.pi)  # Restrict phi to [0, pi/2] for upper hemisphere

    # Convert spherical coordinates to Cartesian coordinates
    x = center_point[0] + hemisphere_radius * math.sin(phi) * math.cos(theta)
    y = center_point[1] + hemisphere_radius * math.sin(phi) * math.sin(theta)
    z = center_point[2] + hemisphere_radius * math.cos(phi)

    sampled_point = (x, y, z)

    return theta,phi,sampled_point

def sample_hemisphere_around_object(obj):
    # Get the object's bounding box vertices
    bounding_box_vertices = obj.bound_box
    bounding_box_vectors = [Vector(vertex) for vertex in bounding_box_vertices]

    # Calculate the current center of the object
    current_center = sum(bounding_box_vectors, Vector()) / 8

    # Calculate the radius of the hemisphere based on the bounding box size
    bounding_box_size = Vector((
        max(vertex[0] for vertex in bounding_box_vertices) - min(vertex[0] for vertex in bounding_box_vertices),
        max(vertex[1] for vertex in bounding_box_vertices) - min(vertex[1] for vertex in bounding_box_vertices),
        max(vertex[2] for vertex in bounding_box_vertices) - min(vertex[2] for vertex in bounding_box_vertices),
    ))
    hemisphere_radius = max(bounding_box_size) * 3
    return sample_point_on_hemisphere(current_center,hemisphere_radius)


def calc_rotation_shine_origin(location):
    """
    Given a location, calculate the rotation necessary to make the light shine to the origin
    """
    rotation = [0,0,0]
    
    distance_to_origin = math.sqrt(location[0]**2+location[1]**2)
    if distance_to_origin == 0:
        distance_to_origin = 1e-4
    angle_elevation = math.atan(location[2]/distance_to_origin)
    rotation[0] = math.pi * 3/2 + angle_elevation

    if location[0] == 0:
        location[0] = 1e-4
    if location[1] == 0:
        location[1] = 1e-4
    rotation_angle = calculate_angle(location[0],location[1])

    rotation[2] = rotation_angle

    return rotation

def calculate_angle(x, y):
    """
    calc angle between two vectors
    """
    # Calculate the angle in radians
    angle_rad = math.atan2(x, y)
    return -angle_rad

def create_new_spot_light():
    spot_data = bpy.data.lights.new(name="spot_light_{}".format(str(uuid.uuid4())), type='SPOT')
    spot_object = bpy.data.objects.new(name="spot_light_object_{}".format(str(uuid.uuid4())), object_data=spot_data)
    bpy.context.collection.objects.link(spot_object)

    # Set the location and rotation of the spot light
    spot_object.location = (0, 2, 2)  # Adjust the location as needed
    spot_object.rotation_euler = (4.71,0,0) # 4.71,0,0 shine straight # (-0.8, 0, 0)  # Adjust the rotation as needed

    # Set up spot light parameters
    spot_data.energy = 60000.0  # Adjust the intensity of the light
    spot_data.distance = 2
    spot_data.spot_size = 0.1  # Adjust the size of the spot cone
    spot_data.spot_blend = 0.2  # Adjust the blend between the spot cone and the background
    spot_data.show_cone = False
    # Optionally, configure other spot light properties
    spot_data.use_shadow = True
    spot_data.shadow_soft_size = 2.0

    return spot_object

def reposition_light_source(light_object,location):
    light_object.location = location
    rotation = calc_rotation_shine_origin(location)
    light_object.rotation_euler = rotation
