import trimesh
import numpy as np
import pyrender
import cairo
from PIL import Image

def generate_window_texture():

    # generate the emmisive texture
    size = (512, 512)
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, *size)
    ctx = cairo.Context(surface)
    ctx.scale(*size)


    ctx.set_source_rgb(1.0,1.0,1.0)
    num_windows_x = 9
    num_windows_y = 15

    window_spacing_x = 1.0 / num_windows_x
    window_size_x = 0.5 / num_windows_x
    window_spacing_y = 1.0 / num_windows_y
    window_size_y = 0.5 / num_windows_y


    for x in np.arange(window_size_x/2, 1.0, window_spacing_x):
        for y in np.arange(window_size_y/2, 1.0, window_spacing_y):
            ctx.rectangle(x,y,window_size_x,window_size_y)
            ctx.fill()

    surface.write_to_png('out/building_texture.png')

    window_texture = Image.open('out/building_texture.png')
    return window_texture


if __name__ == '__main__':

    window_texture = generate_window_texture()
    x1, y1 = 0,0
    x2, y2 = 1,0
    x3, y3 = 1,1
    x4, y4 = 0,1

    coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    poly = trimesh.path.polygons.edges_to_polygons(edges, np.array(coords))[0]
    mesh = trimesh.creation.extrude_polygon(poly, 3.0)
    uvs = [(0,0), (0,1), (1,0), (1,1), (1,0), (1,1), (0,0), (0,1)]

    material = trimesh.visual.material.PBRMaterial(name = 'building', baseColorFactor=[255,0,0], metallicFactor=0.6,
                                                   roughnessFactor=0.5, emissiveTexture=window_texture, emissiveFactor = [1.0,1.0,1.0])
    mesh.visual = trimesh.visual.TextureVisuals(material=material, uv = uvs)

    building = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    scene = pyrender.Scene()
    scene.add(building)
    pyrender.Viewer(scene, viewport_size=(3000, 1400),
                    use_raymond_lighting=False,
                    shadows=True,
                    window_title='Building Demo')



