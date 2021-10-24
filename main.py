import cairo
import random
import os
import math
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

size = (2000, 2000)
size_half = (size[0] / 2, size[1] / 2)

surface = cairo.ImageSurface(cairo.FORMAT_RGB24, *size)
ctx = cairo.Context(surface)
ctx.translate(*size_half)
ctx.scale(*size_half)

paths = []

def deg_to_rad(deg):
    return deg * math.pi / 180.0


def tree(position, angle, depth, rgb):

    length = random.random() * 0.2 + 0.01
    angle_rad = deg_to_rad(angle)
    destination = (position[0] + math.cos( angle_rad ) * length,
                   position[1] + math.sin( angle_rad ) * length)
    ctx.move_to(*position)
    ctx.line_to(*destination)
    ctx.set_source_rgb(rgb[0], rgb[1] * math.sqrt(1.0/math.sqrt(depth)), rgb[2] * math.sqrt(1.0/depth))
    ctx.set_line_width(0.005 - 0.0002 * depth)
    ctx.stroke()

    # add the line segment to our paths
    paths.append( LineString( [Point(position), Point(destination)]))

    if depth < 7:
        angle_left = -15 - random.random() * 10.0
        angle_right = 15 + random.random() * 10.0
        tree(destination, angle + angle_left, depth + 1, rgb)
        tree(destination, angle + angle_right, depth + 1, rgb)

for angle in range(0, 350, 30):
    tree((0.0,0.0), angle + random.random() * 10.0 - 5.0, 1, np.random.random((3)))

for i in np.linspace(-1, 1, 100):
    if random.random() > 0.6:
        ctx.move_to( random.random() * 2.0 - 1.0, i)
        ctx.line_to( random.random() * 2.0 - 1.0, i)
        ctx.set_source_rgba(0.3, 0.3, 0.3, 0.4)
        ctx.stroke()

        ctx.move_to(i, random.random() * 2.0 - 1.0)
        ctx.line_to(i, random.random() * 2.0 - 1.0)
        ctx.set_source_rgba(0.3, 0.3, 0.3, 0.4)
        ctx.stroke()

# now we union the lines
print ('performing union')
union = unary_union(paths)
print (' done')

# grab the end points to determine the intersections
nodes = [c for l in union for c in l.coords]
nodes = np.array(nodes)

# now we remove the duplicate points and retrieve the counts
# the counts are the degrees of each node
nodes, counts = np.unique(nodes, axis=0, return_counts=True)

rgb  = np.random.random((3))

for coord, count in zip(nodes, counts):
    x = coord[0]
    y = coord[1]

    ctx.arc(x,y,0.0024*count,0,math.pi*2)
    ctx.set_source_rgba(*rgb, 0.5)
    ctx.fill()

surface.write_to_png('out/test1.png')
os.system(r'out\test1.png')

# generate the street lights
