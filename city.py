import cairo
import random
import os
import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
import colorsys
from params import *
from tqdm import tqdm
import trimesh
import pyrender
import sys

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

def render_building(angle1, angle2, radius, outer_radius, buildings, height_offset):

    x1 = np.cos(angle1) * radius
    y1 = np.sin(angle1) * radius
    x2 = np.cos(angle2) * radius
    y2 = np.sin(angle2) * radius

    x3 = np.cos(angle2) * outer_radius
    y3 = np.sin(angle2) * outer_radius
    x4 = np.cos(angle1) * outer_radius
    y4 = np.sin(angle1) * outer_radius

    ctx.move_to(x1, y1)
    ctx.line_to(x2, y2)
    ctx.set_line_width(0.001)
    hue = random.random() / 7.0 + 0.5 * angle1 / np.pi
    sat = 1.0 - (random.random() / 7.0 + 1.1 * radius)
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
    mesh = trimesh.creation.extrude_polygon(poly, (1.0 - radius) * building_height_rate + random.random() * 0.02)
    # translate the mesh to lay on the terrain
    mesh.vertices[:, 2] += (1.0 - radius) * terrain_height_rate - 0.008 + height_offset
    mesh.visual.mesh.visual.face_colors = [255, 0, 64, 200]
    buildings.append(mesh)


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
    print (island.centroid, island.area)
    island_height_offset = random.random() * 0.1
    island_height_offsets.append(island_height_offset)

# generate the inner circles
inner_circles = []
meshes = []
print ('generating inner ring geometries')
for radius in tqdm(np.arange(inner_circle_min, inner_circle_max, inner_circle_increment)):

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

        for island_index in range(len(island_height_offsets)):
            if islands[island_index].intersects(line):
                break


        # create the ring geometries
        edges = np.array([np.arange(0, len(coords)-1), np.arange(1, len(coords))]).T
        edges[-1, 1] = 0
        poly = trimesh.path.polygons.edges_to_polygons(edges, np.array(coords))[0]
        mesh = trimesh.creation.extrude_polygon(poly, 0.5 * (1.0 - radius) * terrain_height_rate)
        mesh.vertices[:, 2] += 0.5 * (1.0 - radius) * terrain_height_rate - 0.001 + island_height_offsets[island_index]
        mesh.visual.mesh.visual.face_colors = [64, 128, 64, 200]
        meshes.append(mesh)


# generate the building rings
circle_index = 1
print ('\ndone')
print ('generating the building rings')
buildings = []
for inner_radius in tqdm(np.arange(inner_circle_min, inner_circle_max - inner_circle_increment, inner_circle_increment)):

    #print (f'generating building ring {inner_radius}')
    number_of_buildings = int(inner_radius * building_rate)
    number_of_buildings += random.randint(0, int(number_of_buildings * building_rate_upper_factor))
    angle_offset = random.random() * angle_offset_factor

    angle_inc = 2.0 * np.pi / number_of_buildings
    building_half_angle = angle_inc * building_width_max
    inner_circle = inner_circles[circle_index]
    for angle_original in np.arange(0.0, np.pi * 2.0 - 0.0001, angle_inc):
        angle = angle_original + angle_offset
        x = np.cos(angle) * (inner_radius + 0.0125)
        y = np.sin(angle) * (inner_radius + 0.0125)
        point = Point(x,y)
        if inner_circle.contains(point):

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

                # determine the island offset of this point
                for island_index in range(len(islands)):
                    if islands[island_index].contains(point):
                        break

                # render a building
                outer_radius = inner_radius + 0.017 - random.random()*.005
                radius = inner_radius + 0.003 + random.random()*.005
                angle1 = angle - building_half_angle * (random.random()*0.5 + 0.5)
                angle2 = angle + building_half_angle * (random.random()*0.5 + 0.5)
                height_offset = island_height_offsets[island_index]
                render_building(angle1, angle2, radius, outer_radius, buildings, height_offset)

    circle_index+=1
print('\ndone')

for island in islands:
    x, y = island.centroid.coords[0]
    ctx.arc(x,y, 0.001, 0.0, math.pi*2.0)
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.set_line_width(0.007)
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
building_concat = trimesh.util.concatenate(buildings)

scene = trimesh.scene.Scene(geometry=[terrain_concat, building_concat])
trimesh.exchange.export.export_scene(scene, 'out/out.glb', file_type='glb')
pyrender_terrain = pyrender.Mesh.from_trimesh(terrain_concat, smooth=False)
pyrender_building = pyrender.Mesh.from_trimesh(building_concat, smooth=False)

scene = pyrender.Scene()
scene.add(pyrender_terrain)
scene.add(pyrender_building)
pyrender.Viewer(scene, viewport_size=(1000, 1000),
                use_raymond_lighting=True,
                shadows=True,
                window_title='City Art')


