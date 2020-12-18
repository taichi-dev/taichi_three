from .common import *


@ti.data_oriented
class Scene:
    def __init__(self, res=512, **options):
        self.engine = tina.Engine(res, **options)
        self.res = self.engine.res

        self.img = ti.Vector.field(3, float, self.res)
        self.lighting = tina.Lighting()
        self.default_material = tina.Lambert()
        self.shaders = {}
        self.objects = {}

        @ti.materialize_callback
        def init_light():
            self.lighting.add_light(dir=[0, 1, 1], color=[0.9, 0.9, 0.9])
            self.lighting.set_ambient_light([0.1, 0.1, 0.1])

    def _ensure_material_shader(self, material):
        if material not in self.shaders:
            shader = tina.Shader(self.img, self.lighting, material)
            self.shaders[material] = shader

    def add_object(self, mesh, material=None):
        assert mesh not in self.objects
        if material is None:
            material = self.default_material

        self._ensure_material_shader(material)
        model = tina.Transform(mesh)

        self.objects[mesh] = namespace(model=model, material=material)

    def set_object_transform(self, mesh, trans):
        self.objects[mesh].model.set_transform(trans)

    def render(self):
        self.img.fill(0)
        self.engine.clear_depth()

        for mesh, object in self.objects.items():
            shader = self.shaders[object.material]
            self.engine.set_mesh(mesh)
            self.engine.render(shader)

    def input(self, gui):
        if not hasattr(self, 'control'):
            self.control = tina.Control(gui)
        self.control.get_camera(self.engine)


if __name__ == '__main__':
    scene = Scene(smoothing=True)

    mesh = tina.NoCulling(tina.MeshGrid(64))
    scene.add_object(mesh)

    @ti.kernel
    def deform_mesh(t: float):
        for i, j in mesh.pos:
            xy = mesh.pos[i, j].xy
            z = 0.1 * ti.sin(10 * xy.norm() - ti.tau * t)
            mesh.pos[i, j].z = z

    gui = ti.GUI('scene', scene.res)
    while gui.running:
        scene.input(gui)
        deform_mesh(gui.frame * 0.03)
        scene.render()
        gui.set_image(scene.img)
        gui.show()
