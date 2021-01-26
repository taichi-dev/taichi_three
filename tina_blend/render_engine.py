import bpy
import bgl

import taichi as ti
import numpy as np
import tina


def calc_camera_matrices(depsgraph):
    camera = depsgraph.scene.camera
    render = depsgraph.scene.render
    scale = render.resolution_percentage / 100.0
    proj = np.array(camera.calc_matrix_camera(depsgraph,
        x=render.resolution_x * scale, y=render.resolution_y * scale,
        scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y))
    view = np.linalg.inv(np.array(camera.matrix_world))
    return view, proj


def bmesh_verts_to_numpy(bm):
    arr = [x.co for x in bm.verts]
    if len(arr) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(arr, dtype=np.float32)


def bmesh_faces_to_numpy(bm):
    arr = [[e.index for e in f.verts] for f in bm.faces]
    if len(arr) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(arr, dtype=np.int32)


def bmesh_face_norms_to_numpy(bm):
    vnorms = [x.normal for x in bm.verts]
    if len(vnorms) == 0:
        vnorms = np.zeros((0, 3), dtype=np.float32)
    else:
        vnorms = np.array(vnorms)
    norms = [
        [vnorms[e.index] for e in f.verts]
        if f.smooth else [f.normal for e in f.verts]
        for f in bm.faces]
    if len(norms) == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)
    return np.array(norms, dtype=np.float32)


def bmesh_face_coors_to_numpy(bm):
    uv_lay = bm.loops.layers.uv.active
    if uv_lay is None:
        return np.zeros((len(bm.faces), 3, 2), dtype=np.float32)
    coors = [[l[uv_lay].uv for l in f.loops] for f in bm.faces]
    if len(coors) == 0:
        return np.zeros((0, 3, 2), dtype=np.float32)
    return np.array(coors, dtype=np.float32)


def blender_get_object_mesh(object, depsgraph=None):
    import bmesh
    bm = bmesh.new()
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    object_eval = object.evaluated_get(depsgraph)
    bm.from_object(object_eval, depsgraph)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    verts = bmesh_verts_to_numpy(bm)[bmesh_faces_to_numpy(bm)]
    norms = bmesh_face_norms_to_numpy(bm)
    coors = bmesh_face_coors_to_numpy(bm)
    return verts, norms, coors


class TinaMaterialPanel(bpy.types.Panel):
    '''Tina material options'''

    bl_label = 'Tina Material'
    bl_idname = 'MATERIAL_PT_tina'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'material'

    def draw(self, context):
        layout = self.layout
        object = context.object

        layout.prop_search(object, 'tina_material', bpy.data, 'node_groups')
        layout.operator('scene.tina_reset')


class TinaLightPanel(bpy.types.Panel):
    '''Tina light options'''

    bl_label = 'Tina Light'
    bl_idname = 'DATA_PT_tina'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'data'

    def draw(self, context):
        layout = self.layout
        object = context.object

        if object.type == 'LIGHT':
            layout.prop(object.data, 'tina_color')
            layout.prop(object.data, 'tina_strength')
            layout.operator('scene.tina_reset')


class TinaWorldPanel(bpy.types.Panel):
    '''Tina world options'''

    bl_label = 'Tina World'
    bl_idname = 'WORLD_PT_tina'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'world'

    def draw(self, context):
        layout = self.layout
        world = context.scene.world

        layout.prop(world, 'tina_color')
        layout.prop(world, 'tina_strength')
        layout.operator('scene.tina_reset')


class TinaRenderPanel(bpy.types.Panel):
    '''Tina render options'''

    bl_label = 'Tina Render'
    bl_idname = 'RENDER_PT_tina'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'

    def draw(self, context):
        layout = self.layout
        options = context.scene.tina_render

        row = layout.row()
        row.prop(options, 'taa')
        row.prop(options, 'fxaa')
        row.prop(options, 'ssr')
        row.prop(options, 'ssao')
        row = layout.row()
        row.prop(options, 'blooming')
        row.prop(options, 'smoothing')
        row.prop(options, 'texturing')
        layout.prop(options, 'path_tracing')
        layout.operator('scene.tina_reset')
        # import code; code.interact(local=locals())


class TinaRenderEngine(bpy.types.RenderEngine):
    # These three members are used by blender to set up the
    # RenderEngine; define its internal name, visible name and capabilities.
    bl_idname = "TINA"
    bl_label = "Tina"
    bl_use_preview = True
    instances = []

    # Init is called whenever a new render engine instance is created. Multiple
    # instances may exist at the same time, for example for a viewport and final
    # render.
    def __init__(self):
        self.scene_data = None
        self.draw_data = None
        self.is_pt = False
        self.instances.append(self)

        self.object_to_mesh = {}
        self.light_to_index = {}

    # When the render engine instance is destroy, this is called. Clean up any
    # render engine data here, for example stopping running render threads.
    def __del__(self):
        if self in self.instances:
            self.instances.remove(self)


    def __setup_light_object(self, object, depsgraph, nlights):
        light = object.data
        idx = nlights

        if light.type == 'POINT':
            print('adding point light', object.name)

            color = (light.tina_strength * np.array(light.tina_color)).tolist()
            pos = np.array(object.matrix_world) @ np.array([0, 0, 0, 1])

            @ti.materialize_callback
            def init_light():
                self.scene.lighting.set_light(idx, pos=pos, color=color)

        elif light.type == 'SUN':
            print('adding sun light', object.name)

            color = (light.tina_strength * np.array(light.tina_color)).tolist()
            dir = np.array(object.matrix_world) @ np.array([0, 0, 1, 0])

            @ti.materialize_callback
            def init_light():
                self.scene.lighting.set_light(idx, dir=dir, color=color)

        self.light_to_index[light] = idx

    def __update_light_object(self, object, depsgraph):
        light = object.data
        idx = self.light_to_index[light]

        if light.type == 'POINT':
            print('updating point light', object.name)

            color = (light.tina_strength * np.array(light.tina_color)).tolist()
            pos = np.array(object.matrix_world) @ np.array([0, 0, 0, 1])

            self.scene.lighting.set_light(idx, pos=pos, color=color)

        elif light.type == 'SUN':
            print('updating sun light', object.name)

            color = (light.tina_strength * np.array(light.tina_color)).tolist()
            dir = np.array(object.matrix_world) @ np.array([0, 0, 1, 0])

            self.scene.lighting.set_light(idx, dir=dir, color=color)

    def __setup_mesh_object(self, object, depsgraph, materials):
        print('adding mesh object', object.name)

        verts, norms, coors = blender_get_object_mesh(object, depsgraph)
        world = np.array(object.matrix_world)

        mesh = tina.MeshTransform(tina.SimpleMesh())

        @ti.materialize_callback
        def init_mesh():
            mesh.set_transform(world)
            mesh.set_face_verts(verts)
            mesh.set_face_norms(norms)
            mesh.set_face_coors(coors)

        if not object.tina_material:
            matr = tina.PBR()
        else:
            if object.tina_material not in materials:
                tree = bpy.data.node_groups[object.tina_material]
                from .node_system import construct_material_output
                matr = construct_material_output(tree)
                materials[object.tina_material] = matr
            matr = materials[object.tina_material]
        self.scene.add_object(mesh, matr)

        self.object_to_mesh[object] = mesh

    def __update_mesh_object(self, object, depsgraph):
        print('updating mesh object', object.name)

        verts, norms, coors = blender_get_object_mesh(object, depsgraph)
        world = np.array(object.matrix_world)

        mesh = self.object_to_mesh[object]
        mesh.set_transform(world)
        mesh.set_face_verts(verts)
        mesh.set_face_norms(norms)
        mesh.set_face_coors(coors)

    def __setup_scene(self, depsgraph):
        scene = depsgraph.scene
        self.is_pt = scene.tina_render.path_tracing
        if self.is_pt:
            options = scene.tina_render
            self.scene = tina.PTScene(
                    (self.size_x, self.size_y),
                    smoothing=options.smoothing,
                    texturing=options.texturing)
            self.scene.lighting = tina.Lighting()
        else:
            options = scene.tina_render
            self.scene = tina.Scene((self.size_x, self.size_y),
                    bgcolor=(np.array(scene.world.tina_color)
                        * scene.world.tina_strength).tolist(),
                    taa=options.taa,
                    fxaa=options.fxaa,
                    ssr=options.ssr,
                    ssao=options.ssao,
                    blooming=options.blooming,
                    smoothing=options.smoothing,
                    texturing=options.texturing,
                    tonemap=False)

        materials = {}

        nlights = 0
        for object in depsgraph.ids:
            if isinstance(object, bpy.types.Object):
                if object.type == 'MESH':
                    self.__setup_mesh_object(object, depsgraph, materials)
                elif object.type == 'LIGHT':
                    self.__setup_light_object(object, depsgraph, nlights)
                    nlights += 1

        @ti.materialize_callback
        def set_nlights():
            self.scene.lighting.nlights[None] = nlights
            self.scene.lighting.set_ambient_light(
                    (np.array(scene.world.tina_color)
                        * scene.world.tina_strength).tolist())

    def __update_scene(self, depsgraph):
        for update in depsgraph.updates:
            object = update.id
            if isinstance(object, bpy.types.Object):
                if object.type == 'MESH':
                    self.__update_mesh_object(object, depsgraph)
                elif object.type == 'LIGHT':
                    self.__update_light_object(object, depsgraph)

    # This is the method called by Blender for both final renders (F12) and
    # small preview for materials, world and lights.
    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)
        view, proj = calc_camera_matrices(depsgraph)

        self.__setup(depsgraph, proj @ view)

        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, self.size_x, self.size_y)

        nsamples = 32
        for samp in range(nsamples):
            self.update_stats('Rendering', f'{samp}/{nsamples} Samples')
            self.update_progress((samp + .5) / nsamples)
            if self.test_break():
                break
            self.scene.render()
            if self.is_pt:
                img = self.scene.raw_img**2.2
            else:
                img = np.concatenate([self.scene.img.to_numpy(),
                    np.ones((self.size_x, self.size_y, 1))], axis=2)
            img = np.ascontiguousarray(img.swapaxes(0, 1))
            rect = img.reshape(self.size_x * self.size_y, 4).tolist()
            layer = result.layers[0].passes["Combined"]
            layer.rect = rect
            self.update_result(result)
        else:
            self.update_progress(1.0)

        self.end_result(result)

    def __setup(self, depsgraph, perspective):
        ti.init(ti.gpu)

        self.update_stats('Initializing', 'Loading scene')
        self.__setup_scene(depsgraph)

        if self.is_pt:
            self.update_stats('Initializing', 'Constructing tree')
            self.scene.update()
        else:
            self.update_stats('Initializing', 'Materializing layout')
        self.scene.engine.set_camera(np.eye(4), np.array(perspective))

    def __update_camera(self, perspective):
        self.scene.engine.set_camera(np.eye(4), np.array(perspective))

    # For viewport renders, this method gets called once at the start and
    # whenever the scene or 3D viewport changes. This method is where data
    # should be read from Blender in the same thread. Typically a render
    # thread will be started to do the work while keeping Blender responsive.
    def view_update(self, context, depsgraph):
        print('view_update')

        region = context.region
        region3d = context.region_data
        view3d = context.space_data
        scene = depsgraph.scene

        # Get viewport dimensions
        dimensions = region.width, region.height
        perspective = region3d.perspective_matrix.to_4x4()
        self.size_x, self.size_y = dimensions

        if not self.scene_data:
            # First time initialization
            self.scene_data = True
            first_time = True

            # Loop over all datablocks used in the scene.
            print('setup scene')
            self.__setup(depsgraph, perspective)
        else:
            first_time = False

            print('update scene')
            # Test which datablocks changed
            for update in depsgraph.updates:
                print("Datablock updated:", update.id.name)

            self.__update_scene(depsgraph)

            # Test if any material was added, removed or changed.
            if depsgraph.id_type_updated('MATERIAL'):
                print("Materials updated")

        # Loop over all object instances in the scene.
        if first_time or depsgraph.id_type_updated('OBJECT'):
            for instance in depsgraph.object_instances:
                pass

        self.draw_data = None

    # For viewport renders, this method is called whenever Blender redraws
    # the 3D viewport. The renderer is expected to quickly draw the render
    # with OpenGL, and not perform other expensive work.
    # Blender will draw overlays for selection and editing on top of the
    # rendered image automatically.
    def view_draw(self, context, depsgraph):
        print('view_draw')

        region = context.region
        region3d = context.region_data
        scene = depsgraph.scene

        # Get viewport dimensions
        dimensions = region.width, region.height
        perspective = region3d.perspective_matrix.to_4x4()

        # Bind shader that converts from scene linear to display space,
        bgl.glEnable(bgl.GL_BLEND)
        bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE_MINUS_SRC_ALPHA)
        self.bind_display_space_shader(scene)

        if not self.draw_data or self.draw_data.dimensions != dimensions \
                or self.draw_data.perspective != perspective:
            self.__update_camera(perspective)
            self.scene.clear()
            self.scene.render()
            self.draw_data = TinaDrawData(self.scene, dimensions, perspective)
            self.update_stats('Rendering', 'Done')

        self.draw_data.draw()

        self.unbind_display_space_shader()
        bgl.glDisable(bgl.GL_BLEND)


class TinaDrawData:
    def __init__(self, scene, dimensions, perspective):
        print('redraw!')
        # Generate dummy float image buffer
        self.dimensions = dimensions
        self.perspective = perspective
        width, height = dimensions

        resx, resy = scene.res

        pixels = np.empty(resx * resy * 3, np.float32)
        scene._fast_export_image(pixels)
        self.pixels = bgl.Buffer(bgl.GL_FLOAT, resx * resy * 3, pixels)

        # Generate texture
        self.texture = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenTextures(1, self.texture)
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture[0])
        bgl.glTexImage2D(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGB16F, resx, resy, 0, bgl.GL_RGB, bgl.GL_FLOAT, self.pixels)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

        # Bind shader that converts from scene linear to display space,
        # use the scene's color management settings.
        shader_program = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGetIntegerv(bgl.GL_CURRENT_PROGRAM, shader_program)

        # Generate vertex array
        self.vertex_array = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenVertexArrays(1, self.vertex_array)
        bgl.glBindVertexArray(self.vertex_array[0])

        texturecoord_location = bgl.glGetAttribLocation(shader_program[0], "texCoord")
        position_location = bgl.glGetAttribLocation(shader_program[0], "pos")

        bgl.glEnableVertexAttribArray(texturecoord_location)
        bgl.glEnableVertexAttribArray(position_location)

        # Generate geometry buffers for drawing textured quad
        position = [0.0, 0.0, width, 0.0, width, height, 0.0, height]
        position = bgl.Buffer(bgl.GL_FLOAT, len(position), position)
        texcoord = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        texcoord = bgl.Buffer(bgl.GL_FLOAT, len(texcoord), texcoord)

        self.vertex_buffer = bgl.Buffer(bgl.GL_INT, 2)

        bgl.glGenBuffers(2, self.vertex_buffer)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, self.vertex_buffer[0])
        bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, position, bgl.GL_STATIC_DRAW)
        bgl.glVertexAttribPointer(position_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)

        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, self.vertex_buffer[1])
        bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, texcoord, bgl.GL_STATIC_DRAW)
        bgl.glVertexAttribPointer(texturecoord_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)

        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, 0)
        bgl.glBindVertexArray(0)

    def __del__(self):
        bgl.glDeleteBuffers(2, self.vertex_buffer)
        bgl.glDeleteVertexArrays(1, self.vertex_array)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)
        bgl.glDeleteTextures(1, self.texture)

    def draw(self):
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture[0])
        bgl.glBindVertexArray(self.vertex_array[0])
        bgl.glDrawArrays(bgl.GL_TRIANGLE_FAN, 0, 4)
        bgl.glBindVertexArray(0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)


# RenderEngines also need to tell UI Panels that they are compatible with.
# We recommend to enable all panels marked as BLENDER_RENDER, and then
# exclude any panels that are replaced by custom panels registered by the
# render engine, or that are not supported.
def get_panels():
    exclude_panels = {
        'VIEWLAYER_PT_filter',
        'VIEWLAYER_PT_layer_passes',
    }

    panels = []
    for panel in bpy.types.Panel.__subclasses__():
        if hasattr(panel, 'COMPAT_ENGINES') and 'BLENDER_RENDER' in panel.COMPAT_ENGINES:
            if panel.__name__ not in exclude_panels:
                panels.append(panel)

    return panels


class TinaRenderProperties(bpy.types.PropertyGroup):
    taa: bpy.props.BoolProperty(name='TAA', default=False)
    ssr: bpy.props.BoolProperty(name='SSR', default=False)
    ssao: bpy.props.BoolProperty(name='SSAO', default=False)
    fxaa: bpy.props.BoolProperty(name='FXAA', default=False)
    blooming: bpy.props.BoolProperty(name='Blooming', default=False)
    smoothing: bpy.props.BoolProperty(name='Smoothing', default=True)
    texturing: bpy.props.BoolProperty(name='Texturing', default=True)
    path_tracing: bpy.props.BoolProperty(name='Path Tracing', default=True)


class TinaResetOperator(bpy.types.Operator):
    '''Reset Tina Renderer'''

    bl_idname = "scene.tina_reset"
    bl_label = "Reset Tina"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        for inst in TinaRenderEngine.instances:
            inst.scene_data = None
            inst.tag_update()
            inst.tag_redraw()
        return {'FINISHED'}


def register():
    bpy.utils.register_class(TinaRenderProperties)

    bpy.types.Object.tina_material = bpy.props.StringProperty(name='Material')
    bpy.types.Light.tina_color = bpy.props.FloatVectorProperty(name='Color', subtype='COLOR', min=0, max=1, default=(1, 1, 1))
    bpy.types.Light.tina_strength = bpy.props.FloatProperty(name='Strength', min=0, default=16)
    bpy.types.World.tina_color = bpy.props.FloatVectorProperty(name='Color', subtype='COLOR', min=0, max=1, default=(0.04, 0.04, 0.04))
    bpy.types.World.tina_strength = bpy.props.FloatProperty(name='Strength', min=0, default=1)
    bpy.types.World.tina_strength = bpy.props.FloatProperty(name='Strength', min=0, default=1)
    bpy.types.Scene.tina_render = bpy.props.PointerProperty(name='tina', type=TinaRenderProperties)

    bpy.utils.register_class(TinaRenderEngine)
    bpy.utils.register_class(TinaResetOperator)
    bpy.utils.register_class(TinaMaterialPanel)
    bpy.utils.register_class(TinaLightPanel)
    bpy.utils.register_class(TinaWorldPanel)
    bpy.utils.register_class(TinaRenderPanel)

    for panel in get_panels():
        panel.COMPAT_ENGINES.add('TINA')


def unregister():
    bpy.utils.unregister_class(TinaRenderEngine)
    bpy.utils.unregister_class(TinaResetOperator)
    bpy.utils.unregister_class(TinaMaterialPanel)
    bpy.utils.unregister_class(TinaLightPanel)
    bpy.utils.unregister_class(TinaWorldPanel)
    bpy.utils.unregister_class(TinaRenderPanel)

    for panel in get_panels():
        if 'TINA' in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove('TINA')

    del bpy.types.Object.tina_material
    del bpy.types.Light.tina_color
    del bpy.types.Light.tina_strength
    del bpy.types.World.tina_color
    del bpy.types.World.tina_strength
    del bpy.types.Scene.tina_render

    bpy.utils.unregister_class(TinaRenderProperties)
