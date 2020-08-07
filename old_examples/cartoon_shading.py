import taichi as ti
import taichi_glsl as ts
import taichi_three as t3
ti.init(ti.opengl)

scene = t3.SceneRT()
pos = ti.Vector(3, ti.f32, 3)
radius = ti.var(ti.f32, 3)

scene.add_ball(pos, radius)
scene.set_light_dir([2, 1, -2])

@ti.func
def my_render_func(pos, normal, dir, light_dir):
    n = 4  # cartoon level
    #refl_dir = ts.reflect(light_dir, normal)
    refl_dir = ts.mix(light_dir, -dir, 0.5)
    NoL = pow(ts.dot(normal, refl_dir), 12)
    NoL = ts.mix(NoL, max(0, ts.dot(normal, light_dir)), 0.6)
    strength = 0.2
    if any(normal):
        strength = ti.floor(max(0, NoL * n + 0.5)) / n
    return ts.vec3(strength)

scene.opt.render_func = my_render_func

radius[0] = 0.3
radius[1] = 0.2
radius[2] = 0.2
pos[1] = [-0.3, 0.3, 0]
pos[2] = [+0.3, 0.3, 0]

gui = ti.GUI('Mickey')
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    scene.camera.from_mouse(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
