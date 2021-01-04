from .common import *


@ti.data_oriented
class Scene:
    def __init__(self, res=512, **options):
        '''
        :param res: (int | tuple) resolution of screen
        :param options: options for the rasterizers

        Creates a Tina scene, the top level structure to manage everything in your scene.
        '''

        self.engine = tina.Engine(res)
        self.res = self.engine.res
        self.options = options
        self.rtx = options.get('rtx', False)
        self.taa = options.get('taa', False)
        self.pp = options.get('pp', True)

        self.image = ti.Vector.field(3, float, self.res)
        self.lighting = tina.Lighting() if not self.rtx else tina.RTXLighting()
        self.default_material = tina.Lambert()
        self.shaders = {}
        self.objects = {}

        if self.taa:
            self.accum = tina.Accumator(self.res)

        if self.rtx:
            self.stack = tina.Stack()
            self.tracer = tina.TriangleTracer(**options, multimtl=False)
            self.tree = tina.BVHTree(self.tracer)

        if self.pp:
            self.postp = tina.PostProcessor(self.raw_image, self.res)

        @ti.materialize_callback
        def init_light():
            if not self.rtx:
                self.lighting.add_light(dir=[1, 2, 3], color=[0.9, 0.9, 0.9])
                self.lighting.set_ambient_light([0.1, 0.1, 0.1])
            else:
                pass
                #self.lighting.set_lights(np.array([[0, 0, 2]]))

    def update(self):
        if self.rtx:
            self.tracer.clear_objects()
            for object in self.objects:
                self.tracer.add_object(object, 0)
            self.tracer.build(self.tree)

    def _ensure_material_shader(self, material):
        if material not in self.shaders:
            if not self.rtx:
                shader = tina.Shader(self.image, self.lighting, material)
            else:
                shader = tina.RTXShader(self.image, self.lighting, self.tree, material)
            self.shaders[material] = shader

    def add_object(self, object, material=None, raster=None):
        '''
        :param object: (Mesh | Pars | Voxl) object to add into the scene
        :param material: (Material) specify material for shading the object, self.default_material by default
        :param raster: (Rasterizer) specify the rasterizer for this object, automatically guess if not specified
        '''

        assert object not in self.objects
        if material is None:
            material = self.default_material

        if raster is None:
            if hasattr(object, 'get_nfaces'):
                if hasattr(object, 'get_npolygon') and object.get_npolygon() == 2:
                    if not hasattr(self, 'wireframe_raster'):
                        self.wireframe_raster = tina.WireframeRaster(self.engine, **self.options)
                    raster = self.wireframe_raster
                else:
                    if not hasattr(self, 'triangle_raster'):
                        self.triangle_raster = tina.TriangleRaster(self.engine, **self.options)
                    raster = self.triangle_raster
            elif hasattr(object, 'get_npars'):
                if not hasattr(self, 'particle_raster'):
                    self.particle_raster = tina.ParticleRaster(self.engine, **self.options)
                raster = self.particle_raster
            elif hasattr(object, 'sample_volume'):
                if not hasattr(self, 'volume_raster'):
                    self.volume_raster = tina.VolumeRaster(self.engine, **self.options)
                raster = self.volume_raster
            else:
                raise ValueError(f'cannot determine raster type of object: {object}')
            if self.rtx:
                raster.stack = self.stack

        self._ensure_material_shader(material)

        self.objects[object] = namespace(material=material, raster=raster)

    def init_control(self, gui, center=None, theta=None, phi=None, radius=None,
                     fov=60, blendish=False):
        '''
        :param gui: (GUI) the GUI to bind with
        :param center: (3 * [float]) the target (lookat) position
        :param theta: (float) the altitude of camera
        :param phi: (float) the longitude of camera
        :param radius: (float) the distance from camera to target
        :param blendish: (bool) whether to use blender key bindings
        :param fov: (bool) the initial field of view of the camera
        '''

        self.control = tina.Control(gui, fov, blendish)
        if center is not None:
            self.control.center[:] = center
        if theta is not None:
            self.control.theta = theta
        if phi is not None:
            self.control.phi = phi
        if radius is not None:
            self.control.radius = radius

    def render(self):
        '''
        Render the image to field self.img
        '''

        if self.taa:
            self.engine.randomize_bias(self.accum.count[None] == 0)

        self.image.fill(0)
        self.engine.clear_depth()

        for object, o in self.objects.items():
            shader = self.shaders[o.material]
            o.raster.set_object(object)
            o.raster.render(shader)

        if self.taa:
            self.accum.update(self.image)
        if self.pp:
            self.postp.process()

    @property
    def raw_image(self):
        return self.accum.img if self.taa else self.image

    @property
    def img(self):
        '''
        The image to be displayed in GUI
        '''
        return self.postp.out if self.pp else self.raw_image

    def input(self, gui):
        '''
        :param gui: (GUI) GUI to recieve event from

        Feed inputs from the mouse drag events on GUI to control the camera
        '''

        if not hasattr(self, 'control'):
            self.control = tina.Control(gui)
        changed = self.control.get_camera(self.engine)
        if changed:
            self.clear()
        return changed

    def clear(self):
        if hasattr(self, 'accum'):
            self.accum.clear()

    def load_gltf(self, path):
        '''
        :param path: (str | readable-stream) path to the gltf file

        Load the scene from a GLTF format file
        '''

        from .assimp.gltf import readgltf
        return readgltf(path).extract(self)


# noinspection PyMissingConstructor
class PTScene(Scene):
    def __init__(self, res=512, **options):
        self.tracer = tina.TriangleTracer(**options)
        self.mtltab = tina.MaterialTable()
        self.lighting = tina.RTXLighting()
        self.tree = tina.BVHTree(self.tracer)
        self.engine = tina.PathEngine(self.tree, self.lighting, self.mtltab, res=res)
        self.res = self.engine.res
        self.options = options

        self.default_material = tina.Lambert()
        self.materials = []
        self.objects = []

    def add_object(self, object, material=None, tracer=None):
        if material is None:
            material = self.default_material
        # TODO: multiple tracer

        if material not in self.materials:
            self.materials.append(material)
        mtlid = self.materials.index(material)
        self.objects.append((object, mtlid))

    def clear(self):
        self.engine.clear_image()

    def update(self):
        self.engine.clear_image()
        self.mtltab.clear_materials()
        self.tracer.clear_objects()
        for material in self.materials:
            self.mtltab.add_material(material)
        for object, mtlid in self.objects:
            self.tracer.add_object(object, mtlid)
        self.tracer.build(self.tree)

    def render(self, nsteps=4, strict=True):
        self.engine.load_rays()
        for step in range(nsteps):
            self.engine.step_rays()
        self.engine.update_image(strict)

    @property
    def img(self):  # TODO: use postp for this too
        return self.engine.get_image()

    @property
    def raw_image(self):
        return self.engine.get_image(lambda x: x)
