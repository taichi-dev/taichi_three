import taichi as ti
import numpy as np
import taichi_inject
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True, texturing=True)
if 0:
    scene.load_gltf('assets/sphere.gltf')
else:
    roughness = tina.Param(float)
    #material = tina.CookTorrance(metallic=1.0, roughness=roughness)
    material = tina.Phong(color=[1, 0, 0])
    mesh = tina.MeshModel('assets/sphere.obj')
    scene.add_object(mesh, material)

if 0 and isinstance(scene, tina.PTScene):
    scene.lighting.set_lights(np.array([
        [0, 3, 0],
    ], dtype=np.float32))
    scene.lighting.set_light_radii(np.array([
        0.25,
    ], dtype=np.float32))
    scene.lighting.set_light_colors(np.array([
        [12.0, 12.0, 12.0],
    ], dtype=np.float32))

if isinstance(scene, tina.PTScene):
    scene.update()

gui = ti.GUI('path', scene.res)
if 'roughness' in globals():
    roughness.make_slider(gui, 'roughness')

while gui.running:
    if scene.input(gui):
        scene.clear()
    if gui.frame <= 4000:
        scene.render(nsteps=5)
    gui.set_image(scene.img)
    gui.show()
