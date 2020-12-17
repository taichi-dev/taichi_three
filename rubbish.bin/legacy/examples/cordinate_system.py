import taichi_three as t3

scene = t3.Scene()
camera = t3.Camera()
scene.add_camera(camera)

light = t3.Light(dir=[-0.2, -0.6, -1.0])
scene.add_light(light)

mesh = t3.Mesh.from_obj(t3.Geometry.cube())
xplus = t3.Model(mesh)
xplus.material = t3.Material(t3.CookTorrance(
    color=t3.Constant(t3.RGB(1, 0, 0)),
))
scene.add_model(xplus)
yplus = t3.Model(mesh)
yplus.material = t3.Material(t3.CookTorrance(
    color=t3.Constant(t3.RGB(0, 1, 0)),
))
scene.add_model(yplus)
zplus = t3.Model(mesh)
zplus.material = t3.Material(t3.CookTorrance(
    color=t3.Constant(t3.RGB(0, 0, 1)),
))
scene.add_model(zplus)
center = t3.Model(mesh)
center.material = t3.Material(t3.CookTorrance(
    color=t3.Constant(t3.RGB(1, 1, 1)),
))
scene.add_model(center)

xplus.L2W[None] = t3.translate(1, 0, 0) @ t3.scale(0.1)
yplus.L2W[None] = t3.translate(0, 1, 0) @ t3.scale(0.1)
zplus.L2W[None] = t3.translate(0, 0, 1) @ t3.scale(0.1)
center.L2W[None] = t3.scale(0.1)

gui = t3.GUI('Coordinate system')
while gui.running:
    gui.get_event(None)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()