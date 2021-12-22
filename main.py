import cairo
import random
import os
import math
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import colorsys
#import trimesh
import pyrender

size = (3000, 3000)
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
                   thickness - 0.0001 , angle_range, paths)

        branch(destination, angle + angle_right, type, depth + 1, hsv, max_depth, length_range,
                   thickness - 0.0001 , angle_range, paths)


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

# render the branch starting at the root
# our path list
paths = []
for angle in range(0, 359, 45):
    hue = angle / 360.0
    saturation = 1.0
    value = 1.0
    max_depth = random.randint(3, 9)
    length = 0.05
    thickness = 0.001
    branch((0.0, 0.0), angle + random.random() * 10.0 - 5.0, 1, 1, (hue, saturation, value),
           max_depth, [length - 0.02, length + 0.06], thickness, [15.0, 25.0], paths)

nodes, counts = perform_union(paths)

circles = []
for node in nodes:
    x, y = node
    distance = np.sqrt(x ** 2 + y ** 2) + 0.000001
    radius = 0.004 / distance
    radius = min(radius, 0.12)
    ctx.arc(x,y, radius, 0.0, math.pi * 2.0)
    ctx.set_source_rgb(0.5, 0.5, 0.5)
    ctx.set_line_width(0.0003)
    ctx.stroke()
    point = Point(x,y)
    circle = point.buffer(radius, resolution=4)
    circles.append(circle)

print ('performing circle union')
circle_union = unary_union(circles)
print ('done')

print ('building lines')


# center = Point(0,0)
# for pol in circle_union:
#     lines = []
#
#     boundary = pol.boundary
#     if boundary.type == 'MultiLineString':
#         for line in boundary:
#             lines.append(line)
#     else:
#         lines.append(boundary)
#
#     for line in lines:
#         coords = line.coords
#         for i in range(len(coords) - 1):
#             ctx.move_to(coords[i][0], coords[i][1])
#             ctx.line_to(coords[i + 1][0], coords[i + 1][1])
#             ctx.set_source_rgb(1.0, 0.0, 0.0)
#             ctx.stroke()

# generate the inner circles
for radius in np.arange(0.025, 0.8, 0.02):
    print (f'generating circle of radius {radius}')

    point = Point(0,0)
    circle = point.buffer(radius, resolution = 4)
    circle_inner = point.buffer(radius - 0.02, resolution = 4)
    ring = circle - circle_inner

    inner_circle = circle_union.intersection(ring)
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
            ctx.set_source_rgb(1.0, 0.0, 0.0)
            ctx.stroke()

# generate the inner roads
for inner_radius in np.arange(0.025, 0.112, 0.02):
    for angle in np.arange(0.01, np.pi * 2.0, 0.15):
        if random.random() > 0.7:
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


#    angle = random.uniform(0.0, 360.0)
#    max_depth = 1.1 / distance
#    max_depth = min(max_depth, 6.0)
#    length = max(0.001 / distance, 0.002)
#    length = min(length, 0.01)
#    branch((x,y), angle, 1, 1, (0.0, 0.0, 1.0), max_depth, [length, length*2], 0.001, [85.0, 95.0], secondary_paths)

surface.write_to_png('out/test1.png')
os.system(r'out\test1.png')


