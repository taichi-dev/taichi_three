from ..advans import *


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
        self.pp = options.get('pp', True)
        self.taa = options.get('taa', False)
        self.ibl = options.get('ibl', False)
        self.bgcolor = options.get('bgcolor', 0)

        if not self.ibl:
            self.lighting = tina.Lighting()
        else:
            skybox = tina.Atomsphere()
            skybox = tina.Skybox(skybox.resolution).cook_from(skybox)
            #skybox = tina.Skybox('assets/skybox.jpg', cubic=True)
            self.lighting = tina.SkyboxLighting(skybox)

        self.image = ti.Vector.field(3, float, self.res)
        self.default_material = tina.Diffuse()
        self.post_shaders = []
        self.pre_shaders = []
        self.shaders = {}
        self.objects = {}

        if self.pp:
            self.postp = tina.ToneMapping(self.image, self.res)

        if self.taa:
            self.accum = tina.Accumator(self.res)

        if self.ibl:
            self.background_shader = tina.BackgroundShader(self.image, self.lighting)

        if not self.ibl:
            @ti.materialize_callback
            def add_default_lights():
                    self.lighting.add_light(dir=[1, 2, 3], color=[0.9, 0.9, 0.9])
                    self.lighting.set_ambient_light([0.1, 0.1, 0.1])

    def _ensure_material_shader(self, material):
        if material not in self.shaders:
            shader = tina.Shader(self.image, self.lighting, material)
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

        self._ensure_material_shader(material)

        self.objects[object] = namespace(material=material, raster=raster)

    def init_control(self, gui, center=None, theta=None, phi=None, radius=None,
                     fov=60, is_ortho=False, blendish=True):
        '''
        :param gui: (GUI) the GUI to bind with
        :param center: (3 * [float]) the target (lookat) position
        :param theta: (float) the altitude of camera
        :param phi: (float) the longitude of camera
        :param radius: (float) the distance from camera to target
        :param is_ortho: (bool) whether to use orthogonal mode for camera
        :param blendish: (bool) whether to use blender key bindings
        :param fov: (bool) the initial field of view of the camera
        '''

        self.control = tina.Control(gui, fov=fov, is_ortho=is_ortho, blendish=blendish)
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

        self.image.fill(self.bgcolor)
        self.engine.clear_depth()
        for s in self.pre_shaders:
            s.clear_buffer()
        for s in self.post_shaders:
            s.clear_buffer()

        for object, oinfo in self.objects.items():
            shader = self.shaders[oinfo.material]
            oinfo.raster.set_object(object)
            oinfo.raster.render_occup()
            for s in self.pre_shaders:
                oinfo.raster.render_color(s)
            oinfo.raster.render_color(shader)
            for s in self.post_shaders:
                oinfo.raster.render_color(s)

        if hasattr(self, 'background_shader'):
            self.engine.render_background(self.background_shader)

        if self.pp:
            self.postp.process()
        if self.taa:
            self.accum.update(self.pp_img)

    @property
    def img(self):
        '''
        The final image to be displayed in GUI
        '''
        return self.accum.img if self.taa else self.pp_img

    @property
    def pp_img(self):
        return self.postp.out if self.pp else self.image

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

        return tina.readgltf(path).extract(self)
