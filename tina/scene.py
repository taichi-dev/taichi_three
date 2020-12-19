from .common import *


@ti.data_oriented
class Scene:
    def __init__(self, res=512, taa=False, **options):
        self.engine = tina.Engine(res)
        self.res = self.engine.res

        self.triangle_raster = tina.TriangleRaster(self.engine, **options)
        self.particle_raster = tina.ParticleRaster(self.engine, **options)

        self.image = ti.Vector.field(3, float, self.res)
        self.lighting = tina.Lighting()
        self.default_material = tina.Lambert()
        self.shaders = {}
        self.objects = {}

        self.taa = taa
        if self.taa:
            self.accum = tina.Accumator(self.res)

        @ti.materialize_callback
        def init_light():
            self.lighting.add_light(dir=[0, 1, 1], color=[0.9, 0.9, 0.9])
            self.lighting.set_ambient_light([0.1, 0.1, 0.1])

    def _ensure_material_shader(self, material):
        if material not in self.shaders:
            shader = tina.Shader(self.image, self.lighting, material)
            self.shaders[material] = shader

    def add_object(self, object, material=None):
        assert object not in self.objects
        if material is None:
            material = self.default_material

        self._ensure_material_shader(material)

        self.objects[object] = material

    def init_control(self, gui, center=None, theta=None, phi=None, radius=None):
        self.control = tina.Control(gui)
        if center is not None:
            self.control.center[:] = center
        if theta is not None:
            self.control.theta = theta
        if phi is not None:
            self.control.phi = phi
        if radius is not None:
            self.control.radius = radius

    def render(self):
        if self.taa:
            self.engine.randomize_bias(self.accum.count[None] == 0)

        self.image.fill(0)
        self.engine.clear_depth()

        for object, material in self.objects.items():
            shader = self.shaders[material]
            if hasattr(object, 'get_nfaces'):
                raster = self.triangle_raster
            elif hasattr(object, 'get_npars'):
                raster = self.particle_raster
            else:
                raise ValueError(f'cannot determine raster type of object: {object}')
            raster.set_object(object)
            raster.render(shader)

        if self.taa:
            self.accum.update(self.image)

    @property
    def img(self):
        if self.taa:
            return self.accum.img
        return self.image

    def input(self, gui):
        if not hasattr(self, 'control'):
            self.control = tina.Control(gui)
        changed = self.control.get_camera(self.engine)
        if changed and self.taa:
            self.accum.clear()
        return changed
