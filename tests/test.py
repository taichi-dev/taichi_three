import taichi as ti
import tina

ti.init(ti.cuda)

scene = tina.Scene()
#model = tina.MeshModel('assets/monkey.obj')
model = tina.MeshModel('/home/bate/Documents/testssao.obj')
scene.add_object(model)

#normal = tina.Normal2DShader(ti.Vector.field(2, float, scene.res))
#scene.pre_shaders.append(normal)
ssao = tina.SSAOShader(ti.field(float, scene.res))
scene.post_shaders.append(ssao)

denoise = tina.Denoise(scene.res)
denoise.src = ssao.img
denoise.dst = ti.field(float, scene.res)

scene.lighting.clear_lights()
scene.lighting.set_ambient_light([1, 1, 1])

gui = ti.GUI()
while gui.running:
    scene.input(gui)
    scene.render()
    denoise.knn()
    gui.set_image(denoise.dst)
    gui.show()
