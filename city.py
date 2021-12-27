import cairo
import random
import os
import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPoint
from shapely.ops import unary_union
import colorsys
from params import *
from tqdm import tqdm
import trimesh
import pyrender

size_half = (size[0] / 2, size[1] / 2)

surface = cairo.ImageSurface(cairo.FORMAT_RGB24, *size)
ctx = cairo.Context(surface)
ctx.translate(*size_half)
ctx.scale(*size_half)

def deg_to_rad(deg):
    return deg * math.pi / 180.0

def branch(position, angle, type, depth, hsv, max_depth, length_range, thickness, angle_range, paths):

    angle_rad = deg_to_rad(angle)
    length = random.uniform(length_range[0], length_range[1])
    destination = (position[0] + math.cos( angle_rad ) * length,
                   position[1] + math.sin( angle_rad ) * length)
    ctx.move_to(*position)
    ctx.line_to(*destination)
    rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1] / np.sqrt(depth), hsv[2] / np.sqrt(depth))
    ctx.set_source_rgb(rgb[0], rgb[1], rgb[2])
    ctx.set_line_width(thickness)
    ctx.stroke()

    # add the line segment to our paths
    paths.append( LineString( [Point(position), Point(destination)]))

    if depth < max_depth:
        angle_left = random.uniform(-angle_range[1], angle_range[0])
        angle_right = random.uniform(angle_range[0], angle_range[1])

        branch(destination, angle + angle_left, type, depth + 1, hsv, max_depth, length_range,
                   thickness - thickness_branch_reduction , angle_range, paths)

        branch(destination, angle + angle_right, type, depth + 1, hsv, max_depth, length_range,
                   thickness - thickness_branch_reduction , angle_range, paths)


def perform_union(paths):
    # now we union the lines
    print('performing union')
    union = unary_union(paths)
    print(' done')
    # grab the end points to determine the intersections
    nodes = [c for l in union for c in l.coords]
    nodes = np.array(nodes)
    # now we remove the duplicate points and retrieve the counts
    # the counts are the degrees of each node
    nodes, counts = np.unique(nodes, axis=0, return_counts=True)
    return nodes, counts

def render_building(x, y, max_radius, height_offset, building_geoms):
    dist = np.sqrt(x ** 2 + y ** 2)
    angle = math.atan2(x,y)
    xn, yn = x / dist, y / dist
    # grab the orthogonal direction
    xo, yo = yn, -xn
    random_rate = dist * 0.004 + 0.002
    random_rate_half = random_rate / 2
    forw_random = random.random() * random_rate - random_rate_half
    perp_random = random.random() * random_rate - random_rate_half
    forw_distance = 0.005 + forw_random
    perp_distance = 0.005 + perp_random

    x1 = x - xn * forw_distance - xo * perp_distance
    y1 = y - yn * forw_distance - yo * perp_distance
    x2 = x - xn * forw_distance + xo * perp_distance
    y2 = y - yn * forw_distance + yo * perp_distance
    x3 = x + xn * forw_distance + xo * perp_distance
    y3 = y + yn * forw_distance + yo * perp_distance
    x4 = x + xn * forw_distance - xo * perp_distance
    y4 = y + yn * forw_distance - yo * perp_distance

    ctx.move_to(x1, y1)
    ctx.line_to(x2, y2)
    ctx.set_line_width(0.001)

    hue = random.random() / 7.0 + 0.5 * angle / np.pi
    sat = 1.0 - (random.random() / 7.0 + 1.1 * dist)
    val = random.random() / 2.0 + 0.5
    rgb = colorsys.hsv_to_rgb(hue, sat, val)

    ctx.set_source_rgb(*rgb)
    ctx.stroke()
    ctx.move_to(x2, y2)
    ctx.line_to(x3, y3)
    ctx.stroke()
    ctx.move_to(x3, y3)
    ctx.line_to(x4, y4)
    ctx.stroke()
    ctx.move_to(x4, y4)
    ctx.line_to(x1, y1)
    ctx.stroke()

    coords = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    poly = trimesh.path.polygons.edges_to_polygons(edges, np.array(coords))[0]
    mesh = trimesh.creation.extrude_polygon(poly, (1.0 - dist) * building_height_rate + random.random() * 0.02 - 0.01)
    # translate the mesh to lay on the terrain
    base_height = terrain_get_base_height(max_radius, height_offset) + terrain_get_extrude_height(max_radius, dist)
    mesh.vertices[:, 2] += base_height
    mesh.visual.mesh.visual.face_colors = [255, 0, 64, 200]
    building_geoms.append(mesh)

def terrain_get_base_height(island_max_radius, island_height_offset):
    return 0.8 * (1.0 - island_max_radius) * terrain_height_rate + island_height_offset

def terrain_get_extrude_height(island_max_radius, radius):
    radius_difference = island_max_radius - radius
    return radius_difference * terrain_height_rate + 0.001

# render the branch starting at the root
# our path list
paths = []
for angle in range(0, 359, branch_angle_increment):
    hue = angle / 360.0
    saturation = 1.0
    value = 1.0
    max_depth = random.randint(3, 9)
    length = random.random()/20.0 + 0.05
    thickness = 0.002
    branch((0.0, 0.0), angle + random.random() * 10.0 - 5.0, 1, 1, (hue, saturation, value),
           max_depth, [length - 0.02, length + 0.06], thickness, [15.0, 25.0], paths)

nodes, counts = perform_union(paths)

circles = []
for node in nodes:
    x, y = node
    distance = np.sqrt(x ** 2 + y ** 2) + 0.000001
    radius = influence_radius_multiplier / distance
    radius = min(radius, influence_radius_max)
    radius = max(radius, influence_radius_min)
    point = Point(x,y)
    circle = point.buffer(radius, resolution=influence_circle_resolution)
    coords = circle.exterior.coords[:]
    noise_factor = (inner_circle_noise_factor) * distance
    coords2 = coords + np.random.random( (len(coords), 2)) * noise_factor
    coords = list(map(tuple, coords2))
    circle = Polygon( coords ).buffer(0)
    circles.append(circle)

print ('performing circle union')
circle_union = unary_union(circles)
print ('done')

# let's determine the number of islands, and for each island several properties
islands = list(circle_union.geoms)
island_height_offsets = []
for island in islands:
    island_height_offset = random.random() * 0.1
    island_height_offsets.append(island_height_offset)

# generate the inner circles
inner_circles = []
meshes = []
island_max_radius = np.zeros(len(islands))

print ('generating inner ring geometries')
for radius in tqdm(np.arange(inner_circle_max, inner_circle_min, -inner_circle_increment)):

    point = Point(0,0)
    circle = point.buffer(radius, resolution = 8)
    circle_inner = point.buffer(radius - inner_circle_increment, resolution = 8)
    ring = circle - circle_inner

    inner_circle = circle_union.intersection(ring)
    inner_circles.append(inner_circle)

    lines = []
    boundary = inner_circle.boundary
    if boundary.type == 'MultiLineString':
        for line in boundary:
            lines.append(line)
    else:
        lines.append(boundary)

    for line in lines:
        coords = line.coords
        for i in range(len(coords) - 1):
            ctx.move_to(coords[i][0], coords[i][1])
            ctx.line_to(coords[i + 1][0], coords[i + 1][1])
            ctx.set_line_width(0.001)
            ctx.set_source_rgb(radius, radius, 1.0)
            ctx.stroke()

        for island_index in range(len(islands)):
            if islands[island_index].intersects(line):
                break

        if island_max_radius[island_index] < radius:
            island_max_radius[island_index] = radius
        # create the ring geometries
        edges = np.array([np.arange(0, len(coords)-1), np.arange(1, len(coords))]).T
        edges[-1, 1] = 0
        poly = trimesh.path.polygons.edges_to_polygons(edges, np.array(coords))[0]

        base_height = terrain_get_base_height(island_max_radius[island_index], island_height_offsets[island_index])
        extrude_height = terrain_get_extrude_height(island_max_radius[island_index], radius)

        mesh = trimesh.creation.extrude_polygon(poly, extrude_height)
        mesh.vertices[:, 2] += base_height
        mesh.visual.mesh.visual.face_colors = [222, 222, 222, 255]
        meshes.append(mesh)


# generate the building candidates
print ('\ndone')
print ('generating building candidates')
building_points = []
for inner_radius in np.arange(inner_circle_max, inner_circle_min, -inner_circle_increment):

    number_of_buildings = int(inner_radius * building_rate)
    number_of_buildings += random.randint(0, int(number_of_buildings * building_rate_upper_factor))
    angle_offset = random.random() * angle_offset_factor
    angle_inc = 2.0 * np.pi / number_of_buildings

    for angle_original in np.arange(0.0, np.pi * 2.0 - 0.0001, angle_inc):
        angle = angle_original + angle_offset

        if random.random() > 0.65 + inner_radius:
            outer_radius = inner_radius + 0.02
            x1 = np.cos(angle) * inner_radius
            y1 = np.sin(angle) * inner_radius
            x2 = np.cos(angle) * outer_radius
            y2 = np.sin(angle) * outer_radius
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
            ctx.set_line_width(0.001)
            ctx.set_source_rgb(1.0,1.0,1.0)
            ctx.stroke()

        elif random.random() > inner_radius * 0.6:
            x = np.cos(angle) * (inner_radius + 0.0125)
            y = np.sin(angle) * (inner_radius + 0.0125)
            point = Point(x, y)
            building_points.append(point)

print('\ndone')

# intersect the building candidates with the islands
building_geoms = []
building_points = MultiPoint(building_points)
e = 0
print ('intersecting islands with building candidates')
for island in tqdm(islands):
    #print (f'intersecting island {e + 1} with building candidates')
    intersection = island.intersection(building_points)
    if intersection:
        max_radius = island_max_radius[e]
        height_offset = island_height_offsets[e]
        # loop through the intersected buildings and add them to our geometries
        if isinstance(intersection, MultiPoint):
            point_list = intersection.geoms
        else:
            point_list = MultiPoint([intersection])

        for i in point_list:
            x,y = i.coords[0]
            render_building(x,y, max_radius, height_offset, building_geoms)
    e+=1

for island in islands:
    x, y = island.centroid.coords[0]
    ctx.arc(x,y, 0.007, 0.0, math.pi*2.0)
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.set_line_width(0.003)
    ctx.stroke()

for node in nodes:
    x, y = node
    distance = np.sqrt(x ** 2 + y ** 2) + 0.000001
    radius = influence_radius_multiplier / distance
    radius = min(radius, influence_radius_max)
    radius = max(radius, influence_radius_min)
    ctx.arc(x,y, radius, 0.0, math.pi * 2.0)
    ctx.set_source_rgb(0.9, 0.4, 0.4)
    ctx.set_line_width(0.0001)
    ctx.stroke()


surface.write_to_png(out_image)
os.system(out_image)
terrain_concat = trimesh.util.concatenate(meshes)
building_concat = trimesh.util.concatenate(building_geoms)

#scene = trimesh.scene.Scene(geometry=[terrain_concat, building_concat])
#trimesh.exchange.export.export_scene(scene, 'out/out.glb', file_type='glb')
pyrender_terrain = pyrender.Mesh.from_trimesh(terrain_concat, smooth=False)
pyrender_building = pyrender.Mesh.from_trimesh(building_concat, smooth=False)

scene = pyrender.Scene()
scene.add(pyrender_terrain)
scene.add(pyrender_building)
pyrender.Viewer(scene, viewport_size=(1000, 1000),
                use_raymond_lighting=True,
                shadows=True,
                window_title='City Art')


