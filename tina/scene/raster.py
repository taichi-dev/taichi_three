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
        self.taa = options.get('taa', False)
        self.ibl = options.get('ibl', False)
        self.ssr = options.get('ssr', False)
        self.ssao = options.get('ssao', False)
        self.fxaa = options.get('fxaa', False)
        self.tonemap = options.get('tonemap', True)
        self.blooming = options.get('blooming', False)
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
        self.materials = []
        self.shaders = {}
        self.objects = {}

        if self.ssr:
            self.mtltab = tina.MaterialTable()

            @ti.materialize_callback
            def init_mtltab():
                self.mtltab.clear_materials()
                for material in self.materials:
                    self.mtltab.add_material(material)

        if self.ssao or self.ssr:
            self.norm_buffer = ti.Vector.field(3, float, self.res)
            self.norm_shader = tina.NormalShader(self.norm_buffer)
            self.pre_shaders.append(self.norm_shader)

        if self.ssr:
            self.mtlid_buffer = ti.field(int, self.res)
            if 'texturing' in options:
                self.coor_buffer = ti.Vector.field(2, float, self.res)
                self.coor_shader = tina.TexcoordShader(self.coor_buffer)
                self.pre_shaders.append(self.coor_shader)
            else:
                self.coor_buffer = ti.Vector.field(2, float, (1, 1))

        if self.ssao:
            self.ssao = tina.SSAO(self.res, self.norm_buffer, taa=self.taa)

        if self.ssr:
            self.ssr = tina.SSR(self.res, self.norm_buffer,
                    self.coor_buffer, self.mtlid_buffer, self.mtltab, taa=self.taa)

        if self.blooming:
            self.blooming = tina.Blooming(self.res)

        self.pp_img = self.image

        if self.tonemap:
            self.tonemap = tina.ToneMapping(self.res)

        if self.fxaa:
            self.fxaa = tina.FXAA(self.res)

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
        if material in self.materials:
            return

        shader = tina.Shader(self.image, self.lighting, material)

        base_shaders = [shader]
        if self.ssr:
            mtlid = len(self.materials)
            mtlid_shader = tina.ConstShader(self.mtlid_buffer, mtlid)
            base_shaders.append(mtlid_shader)
        shader = tina.ShaderGroup(self.pre_shaders
                + base_shaders + self.post_shaders)

        self.materials.append(material)
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
            oinfo.raster.render_color(shader)

        if self.ssao:
            self.ssao.render(self.engine)
            self.ssao.apply(self.image)

        if self.ssr:
            self.ssr.render(self.engine, self.image)
            self.ssr.apply(self.image)

        if hasattr(self, 'background_shader'):
            self.engine.render_background(self.background_shader)

        if self.blooming:
            self.blooming.apply(self.image)
        if self.tonemap:
            self.tonemap.apply(self.image)
        if self.fxaa:
            self.fxaa.apply(self.image)
        if self.taa:
            self.accum.update(self.pp_img)

    @property
    def img(self):
        '''
        The final image to be displayed in GUI
        '''
        return self.accum.img if self.taa else self.pp_img

    def input(self, gui):
        '''
        :param gui: (GUI) GUI to recieve event from

        Feed inputs from the mouse drag events on GUI to control the camera
        '''

        if not hasattr(self, 'control'):
            self.control = tina.Control(gui)
        changed = self.control.apply_camera(self.engine)
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

    @ti.kernel
    def _fast_export_image(self, out: ti.ext_arr()):
        for x, y in ti.grouped(self.img):
            base = (y * self.res.x + x) * 3
            r, g, b = self.img[x, y]
            out[base + 0] = r
            out[base + 1] = g
            out[base + 2] = b

    def visualize(self):
        with ti.GUI() as gui:
            while gui.running:
                self.input(gui)
                self.render()
                gui.set_image(self.img)
                gui.show()
