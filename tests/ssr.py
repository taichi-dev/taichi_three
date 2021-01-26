import taichi as ti
import tina

ti.init(ti.gpu)


scene = tina.Scene((640, 480), smoothing=True, ssr=True, taa=True)
monkey_material = tina.PBR(metallic=0.0, roughness=0.4)
monkey = tina.MeshModel('assets/monkey.obj')
scene.add_object(monkey, monkey_material)

param_metallic = tina.Param()
param_roughness = tina.Param()
plane_material = tina.PBR(metallic=param_metallic, roughness=param_roughness)
plane = tina.MeshTransform(tina.MeshGrid(32),
    tina.scale(2) @ tina.eularXYZ([-ti.pi / 2, 0, 0]))
scene.add_object(plane, plane_material)

gui = ti.GUI(res=scene.res)
nsteps = gui.slider('nsteps', 1, 128, 1)
nsamples = gui.slider('nsamples', 1, 128, 1)
stepsize = gui.slider('stepsize', 0, 32, 0.1)
tolerance = gui.slider('tolerance', 0, 64, 0.1)
blurring = gui.slider('blurring', 1, 8, 1)
metallic = gui.slider('metallic', 0, 1, 0.01)
roughness = gui.slider('roughness', 0, 1, 0.01)
nsteps.value = 64
nsamples.value = 12
blurring.value = 4
stepsize.value = 2
tolerance.value = 15
metallic.value = 1.0
roughness.value = 0.0

while gui.running:
    scene.ssr.nsteps[None] = int(nsteps.value)
    scene.ssr.nsamples[None] = int(nsamples.value)
    scene.ssr.blurring[None] = int(blurring.value)
    scene.ssr.stepsize[None] = stepsize.value
    scene.ssr.tolerance[None] = tolerance.value
    param_metallic.value[None] = metallic.value
    param_roughness.value[None] = roughness.value
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
