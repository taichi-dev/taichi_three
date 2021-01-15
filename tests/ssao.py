import taichi as ti
import tina

ti.init(ti.cuda)

scene = tina.Scene()
model = tina.MeshModel('assets/monkey.obj')
#model = tina.MeshModel('/home/bate/Documents/testssao.obj')
scene.add_object(model)

vnorm = tina.NormalShader(ti.Vector.field(3, float, scene.res))
scene.pre_shaders.append(vnorm)
ssao = tina.SSAOShader(scene.res, vnorm.img)
scene.post_shaders.append(ssao)
scene.post_pp = ssao.apply

scene.lighting.clear_lights()
scene.lighting.set_ambient_light([1, 1, 1])

gui = ti.GUI(res=scene.res)

radius = gui.slider('radius', 0, 2, 0.01)
thresh = gui.slider('thresh', 0, 1, 0.01)
factor = gui.slider('factor', 0, 2, 0.01)
radius.value = 0.2
thresh.value = 0.0
factor.value = 1.0

while gui.running:
    ssao.radius[None] = radius.value
    ssao.thresh[None] = thresh.value
    ssao.factor[None] = factor.value

    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
