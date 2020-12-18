import taichi as ti
import tina

ti.init(ti.cpu)

#obj = tina.readobj('assets/monkey.obj')
#verts = tina.objverts(obj)
#coors = tina.objcoors(obj)

mesh = tina.MeshGrid(3)

engine = tina.Engine(texturing=True)
lighting = tina.Lighting()
material = tina.CookTorrance(basecolor=tina.Texture('assets/cloth.jpg'))

img = ti.Vector.field(3, float, engine.res)
shader = tina.Shader(img, lighting, material)

lighting.add_light(dir=[1, 1, 1])

gui = ti.GUI('test')
control = tina.Control(gui)

while gui.running:
    control.get_camera(engine)

    img.fill(0)
    engine.clear_depth()

    #engine.set_face_verts(verts)
    #engine.set_face_coors(coors)
    engine.set_mesh(mesh)
    engine.render(shader)

    gui.set_image(img)
    gui.show()
