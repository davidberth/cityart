lux.importFile("c:/harvard/cityart/test.glb")

## polygons

# if path_dist > 0.0001:
#    dists = math.sqrt(xdir**2 + ydir**2)
#    coord = np.zeros((4,2))

#    coord[0, 0] = orig[0] + xdir_side
#    coord[0, 1] = orig[1] + ydir_side
#    coord[1, 0] = orig[0] - xdir_side
#    coord[1, 1] = orig[1] - ydir_side

#    coord[3, 0] = orig[0] + xdir + xdir_side
#    coord[3, 1] = orig[1] + ydir + ydir_side
#    coord[2, 0] = orig[0] + xdir - xdir_side
#    coord[2, 1] = orig[1] + ydir - ydir_side

#    edges = np.array([np.arange(0, 4), np.arange(1, 5)]).T
#    edges[-1, 1] = 0

# poly = trimesh.path.polygons.edges_to_polygons(edges, coord)[0]

# mesh = trimesh.creation.extrude_polygon(poly, 0.001)
# meshes.append(mesh)


## street lights

#meshes = []
# generate the street lights
# for path in union:
#     coords = path.coords
#     coord1 = coords[0]
#     coord2 = coords[1]
#     dist1 = math.sqrt(coord1[0] ** 2 + coord1[1] ** 2)
#     dist2 = math.sqrt(coord2[0] ** 2 + coord2[1] ** 2)
#     dista = (dist1 + dist2) / 2.0
#     if dist2 > dist1:
#         dest = coord2
#         orig = coord1
#     else:
#         dest = coord1
#         orig = coord2
#
#     xdir = dest[0] - orig[0]
#     ydir = dest[1] - orig[1]
#     ang = math.atan2(ydir, xdir)
#     angs = ang + math.pi / 2
#     xdir_side = math.cos(angs) / 320.0
#     ydir_side = math.sin(angs) / 320.0
#     path_dist = np.abs(dist2 - dist1)
#
#
#     for i in np.arange(0, 1, 0.02 / path_dist ):
#         x1 = orig[0] + xdir * i + xdir_side
#         y1 = orig[1] + ydir * i + ydir_side
#         x2 = orig[0] + xdir * i - xdir_side
#         y2 = orig[1] + ydir * i - ydir_side
#
#         ctx.arc(x1, y1, 0.0024, 0, math.pi * 2)
#         ctx.set_source_rgba(*rgb, 0.9 / (dist1 * 4.0 + 0.000001))
#         ctx.fill()
#
#         ctx.arc(x2, y2, 0.0024, 0, math.pi * 2)
#         ctx.set_source_rgba(*rgb, 0.9 / (dist1 * 4.0 + 0.000001))
#         ctx.fill()

## render polygons

#mesh = trimesh.util.concatenate(meshes)
# mesh.show()

#scene = trimesh.scene.Scene(geometry=mesh)
#trimesh.exchange.export.export_scene(scene, 'test.glb', file_type='glb')
#pyrenderMesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
#scene = pyrender.Scene()
#scene.add(pyrenderMesh)
#pyrender.Viewer(scene, viewport_size=(1000, 1000),
#                use_raymond_lighting=True,
#                shadows=True,
#                window_title='City Art')

#os.system(r'"c:\program files\KeyShot10\bin\keyshot.exe" -script keyshot.py')


## background grid
# render the grid
# for i in np.linspace(-1, 1, 100):
#     if random.random() > 0.6:
#         ctx.move_to( random.random() * 2.0 - 1.0, i)
#         ctx.line_to( random.random() * 2.0 - 1.0, i)
#         ctx.set_source_rgba(0.3, 0.3, 0.3, 0.4)
#         ctx.stroke()
#
#         ctx.move_to(i, random.random() * 2.0 - 1.0)
#         ctx.line_to(i, random.random() * 2.0 - 1.0)
#         ctx.set_source_rgba(0.3, 0.3, 0.3, 0.4)
#         ctx.stroke()

## render hubs
# rgb  = np.random.random((3))
#
# for coord, count in zip(nodes, counts):
#     x = coord[0]
#     y = coord[1]
#
#     ctx.arc(x,y,0.0024*count,0,math.pi*2)
#     ctx.set_source_rgba(*rgb, 0.5)
#     ctx.fill()



