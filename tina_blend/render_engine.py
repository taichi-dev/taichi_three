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


class TinaRenderEngine(bpy.types.RenderEngine):
    # These three members are used by blender to set up the
    # RenderEngine; define its internal name, visible name and capabilities.
    bl_idname = "TINA"
    bl_label = "Tina"
    bl_use_preview = True

    # Init is called whenever a new render engine instance is created. Multiple
    # instances may exist at the same time, for example for a viewport and final
    # render.
    def __init__(self):
        self.scene_data = None
        self.draw_data = None

    # When the render engine instance is destroy, this is called. Clean up any
    # render engine data here, for example stopping running render threads.
    def __del__(self):
        pass

    def __update_light_object(self, s, object, depsgraph):
        light = object.data
        if light.type == 'POINT':
            print('adding point light', object.name)

            color = (light.tina_strength * np.array(light.tina_color)).tolist()
            pos = np.array(object.matrix_world) @ np.array([0, 0, 1, 0])

            @ti.materialize_callback
            def init_light():
                s.lighting.add_light(pos=pos, color=color)

        elif light.type == 'SUN':
            print('adding sun light', object.name)

            color = (light.tina_strength * np.array(light.tina_color)).tolist()
            dir = np.array(object.matrix_world) @ np.array([0, 0, 0, 1])

            @ti.materialize_callback
            def init_light():
                s.lighting.add_light(dir=dir, color=color)

    def __update_mesh_object(self, s, object, depsgraph):
        print('adding mesh object', object.name)

        mesh = tina.MeshTransform(tina.SimpleMesh())
        verts, norms, coors = blender_get_object_mesh(object, depsgraph)
        world = np.array(object.matrix_world)

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
        s.add_object(mesh, matr)

    def __update_scene(self, s, depsgraph):
        materials = {}
        # import code; code.interact(local=locals())

        @ti.materialize_callback
        def clear_lights():
            s.lighting.clear_lights()

        for object in depsgraph.ids:
            if isinstance(object, bpy.types.Object):
                if object.type == 'MESH':
                    self.__update_mesh_object(s, object, depsgraph)
                elif object.type == 'LIGHT':
                    self.__update_light_object(s, object, depsgraph)

    # This is the method called by Blender for both final renders (F12) and
    # small preview for materials, world and lights.
    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)
        view, proj = calc_camera_matrices(depsgraph)

        is_pt = False

        ti.init(ti.cpu)

        self.update_stats('Initializing', 'Loading scene')
        if is_pt:
            s = tina.PTScene((self.size_x, self.size_y))
        else:
            options = scene.tina_render
            s = tina.Scene((self.size_x, self.size_y),
                    bgcolor=(np.array(scene.world.tina_color)
                        * scene.world.tina_strength).tolist(),
                    taa=options.taa,
                    fxaa=options.fxaa,
                    ssr=options.ssr,
                    ssao=options.ssao,
                    blooming=options.blooming,
                    smoothing=options.smoothing,
                    texturing=options.texturing,
                    )
        self.__update_scene(s, depsgraph)

        if is_pt:
            self.update_stats('Initializing', 'Constructing tree')
            s.update()
        else:
            self.update_stats('Initializing', 'Materializing layout')
        s.engine.set_camera(view, proj)

        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, self.size_x, self.size_y)

        nsamples = 1
        for samp in range(nsamples):
            self.update_stats('Rendering', f'{samp}/{nsamples} Samples')
            self.update_progress((samp + .5) / nsamples)
            if self.test_break():
                break
            s.render()
            if is_pt:
                img = s.raw_img**2.2
            else:
                img = np.concatenate([s.img.to_numpy(),
                    np.ones((self.size_x, self.size_y, 1))], axis=2)
            img = np.ascontiguousarray(img.swapaxes(0, 1))
            rect = img.reshape(self.size_x * self.size_y, 4).tolist()
            layer = result.layers[0].passes["Combined"]
            layer.rect = rect
            self.update_result(result)
        else:
            self.update_stats('Done', f'{nsamples} Samples')
            self.update_progress(1.0)

        self.end_result(result)

    # For viewport renders, this method gets called once at the start and
    # whenever the scene or 3D viewport changes. This method is where data
    # should be read from Blender in the same thread. Typically a render
    # thread will be started to do the work while keeping Blender responsive.
    def view_update(self, context, depsgraph):
        region = context.region
        view3d = context.space_data
        scene = depsgraph.scene

        # Get viewport dimensions
        dimensions = region.width, region.height

        if not self.scene_data:
            # First time initialization
            self.scene_data = []
            first_time = True

            # Loop over all datablocks used in the scene.
            for datablock in depsgraph.ids:
                pass
        else:
            first_time = False

            # Test which datablocks changed
            for update in depsgraph.updates:
                print("Datablock updated: ", update.id.name)

            # Test if any material was added, removed or changed.
            if depsgraph.id_type_updated('MATERIAL'):
                print("Materials updated")

        # Loop over all object instances in the scene.
        if first_time or depsgraph.id_type_updated('OBJECT'):
            for instance in depsgraph.object_instances:
                pass

    # For viewport renders, this method is called whenever Blender redraws
    # the 3D viewport. The renderer is expected to quickly draw the render
    # with OpenGL, and not perform other expensive work.
    # Blender will draw overlays for selection and editing on top of the
    # rendered image automatically.
    def view_draw(self, context, depsgraph):
        region = context.region
        scene = depsgraph.scene

        # Get viewport dimensions
        dimensions = region.width, region.height

        # Bind shader that converts from scene linear to display space,
        bgl.glEnable(bgl.GL_BLEND)
        bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE_MINUS_SRC_ALPHA)
        self.bind_display_space_shader(scene)

        if not self.draw_data or self.draw_data.dimensions != dimensions:
            self.draw_data = TinaDrawData(dimensions)

        self.draw_data.draw()

        self.unbind_display_space_shader()
        bgl.glDisable(bgl.GL_BLEND)


class TinaDrawData:
    def __init__(self, dimensions):
        # Generate dummy float image buffer
        self.dimensions = dimensions
        width, height = dimensions

        pixels = [0.1, 0.2, 0.1, 1.0] * width * height
        pixels = bgl.Buffer(bgl.GL_FLOAT, width * height * 4, pixels)

        # Generate texture
        self.texture = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenTextures(1, self.texture)
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture[0])
        bgl.glTexImage2D(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGBA16F, width, height, 0, bgl.GL_RGBA, bgl.GL_FLOAT, pixels)
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
    taa: bpy.props.BoolProperty(name='TAA', default=True)
    ssr: bpy.props.BoolProperty(name='SSR', default=False)
    ssao: bpy.props.BoolProperty(name='SSAO', default=False)
    fxaa: bpy.props.BoolProperty(name='FXAA', default=False)
    blooming: bpy.props.BoolProperty(name='Blooming', default=False)
    smoothing: bpy.props.BoolProperty(name='Smoothing', default=True)
    texturing: bpy.props.BoolProperty(name='Texturing', default=True)


def register():
    bpy.utils.register_class(TinaRenderProperties)

    bpy.types.Object.tina_material = bpy.props.StringProperty(name='Material')
    bpy.types.Light.tina_color = bpy.props.FloatVectorProperty(name='Color', subtype='COLOR', min=0, max=1, default=(1, 1, 1))
    bpy.types.Light.tina_strength = bpy.props.FloatProperty(name='Strength', min=0, default=1)
    bpy.types.World.tina_color = bpy.props.FloatVectorProperty(name='Color', subtype='COLOR', min=0, max=1, default=(0.04, 0.04, 0.04))
    bpy.types.World.tina_strength = bpy.props.FloatProperty(name='Strength', min=0, default=1)
    bpy.types.World.tina_strength = bpy.props.FloatProperty(name='Strength', min=0, default=1)
    bpy.types.Scene.tina_render = bpy.props.PointerProperty(name='tina', type=TinaRenderProperties)

    bpy.utils.register_class(TinaRenderEngine)
    bpy.utils.register_class(TinaMaterialPanel)
    bpy.utils.register_class(TinaLightPanel)
    bpy.utils.register_class(TinaWorldPanel)
    bpy.utils.register_class(TinaRenderPanel)

    for panel in get_panels():
        panel.COMPAT_ENGINES.add('TINA')


def unregister():
    bpy.utils.unregister_class(TinaRenderEngine)
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


'''''
'''''
