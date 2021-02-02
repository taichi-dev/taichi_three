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

        layout.prop(options, 'render_samples')
        layout.prop(options, 'viewport_samples')
        layout.prop(options, 'start_pixel_size')
        row = layout.row()
        row.prop(options, 'smoothing')
        row.prop(options, 'texturing')
        layout.operator('scene.tina_reset')


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

        self.object_to_mesh = {}
        self.material_to_id = {}
        self.nblocks = 0
        self.nsamples = 0
        self.viewport_samples = 16

    # When the render engine instance is destroy, this is called. Clean up any
    # render engine data here, for example stopping running render threads.
    def __del__(self):
        pass

    def __setup_mesh_object(self, object, depsgraph):
        print('adding mesh object', object.name, object.tina_material)

        verts, norms, coors = blender_get_object_mesh(object, depsgraph)
        world = np.array(object.matrix_world)

        if not object.tina_material:
            matr, mtlid = self.scene.materials[0], 0
        else:
            if object.tina_material not in self.material_to_id:
                tree = bpy.data.node_groups[object.tina_material]
                from .node_system import construct_material_output
                matr = construct_material_output(tree)
                self.material_to_id[object.tina_material] = len(self.scene.materials)
                self.scene.materials.append(matr)
            mtlid = self.material_to_id[object.tina_material]

        self.object_to_mesh[object] = world, verts, norms, coors, mtlid

    def __update_mesh_object(self, object, depsgraph):
        print('updating mesh object', object.name)

        verts, norms, coors = blender_get_object_mesh(object, depsgraph)
        world = np.array(object.matrix_world)

        if not object.tina_material:
            matr, mtlid = self.scene.materials[0], 0
        else:
            if object.tina_material not in self.material_to_id:
                tree = bpy.data.node_groups[object.tina_material]
                from .node_system import construct_material_output
                matr = construct_material_output(tree)
                self.material_to_id[object.tina_material] = len(self.scene.materials)
                self.scene.materials.append(matr)
            mtlid = self.material_to_id[object.tina_material]

        self.object_to_mesh[object] = world, verts, norms, coors, mtlid

    def __setup_scene(self, depsgraph):
        scene = depsgraph.scene
        options = scene.tina_render
        self.scene = tina.PTScene(
                (self.size_x, self.size_y),
                smoothing=options.smoothing,
                texturing=options.texturing)
        self.scene.lighting = tina.Lighting()

        for object in depsgraph.ids:
            if isinstance(object, bpy.types.Object):
                if object.type == 'MESH':
                    self.__setup_mesh_object(object, depsgraph)

        @ti.materialize_callback
        def init_scene():
            for world, verts, norms, coors, mtlid in self.object_to_mesh.values():
                self.scene.add_mesh(world, verts, norms, coors, mtlid)

    def __update_scene(self, depsgraph):
        need_update = False
        for update in depsgraph.updates:
            object = update.id
            if isinstance(object, bpy.types.Scene):
                obj_to_del = []
                for obj in self.object_to_mesh:
                    if obj.name not in object.objects:
                        # this object was deleted
                        print('delete object', obj)
                        obj_to_del.append(obj)
                for obj in obj_to_del:
                    del self.object_to_mesh[obj]
                    need_update = True
            if isinstance(object, bpy.types.Object):
                if object.type == 'MESH':
                    self.__update_mesh_object(object, depsgraph)
                    need_update = True
        if need_update:
            self.scene.clear_objects()
            for world, verts, norms, coors, mtlid in self.object_to_mesh.values():
                self.scene.add_mesh(world, verts, norms, coors, mtlid)
            self.scene.update()
            self.__reset_samples(depsgraph.scene)

    def __reset_samples(self, scene):
        self.nsamples = 0
        self.nblocks = scene.tina_render.start_pixel_size

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

        nsamples = scene.tina_render.render_samples
        for samp in range(nsamples):
            self.update_stats('Rendering', f'{samp}/{nsamples} Samples')
            self.update_progress((samp + .5) / nsamples)
            if self.test_break():
                break
            self.scene.render()
            img = self.scene.raw_img
            #img = np.ones((self.size_x, self.size_y, 4))

            img = np.ascontiguousarray(img.swapaxes(0, 1))
            rect = img.reshape(self.size_x * self.size_y, 4).tolist()
            # import code; code.interact(local=locals())
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

        self.update_stats('Initializing', 'Constructing tree')
        self.scene.update()
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
        max_samples = scene.tina_render.viewport_samples

        # Get viewport dimensions
        dimensions = region.width, region.height
        perspective = region3d.perspective_matrix.to_4x4()

        # Bind shader that converts from scene linear to display space,
        bgl.glEnable(bgl.GL_BLEND)
        bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE_MINUS_SRC_ALPHA)
        self.bind_display_space_shader(scene)

        if not self.draw_data or self.draw_data.dimensions != dimensions \
                or self.draw_data.perspective != perspective:
            self.__reset_samples(scene)
            self.__update_camera(perspective)

        if self.nsamples < max_samples:
            if self.nblocks > 1:
                self.nsamples = 0
                self.scene.clear()
            else:
                if self.nblocks == 1:
                    self.scene.clear()
                self.nsamples += 1
            self.scene.render(blocksize=self.nblocks)
            self.draw_data = TinaDrawData(self.scene, dimensions, perspective,
                    self.nblocks)
            self.update_stats('Rendering', f'{self.nsamples}/{max_samples} Samples')

            if self.nsamples < max_samples or self.nblocks != 0:
                self.tag_redraw()

            self.nblocks //= 2

        self.draw_data.draw()

        self.unbind_display_space_shader()
        bgl.glDisable(bgl.GL_BLEND)


class TinaDrawData:
    def __init__(self, scene, dimensions, perspective, blocksize):
        print('redraw!')
        # Generate dummy float image buffer
        self.dimensions = dimensions
        self.perspective = perspective
        width, height = dimensions

        resx, resy = scene.res
        if blocksize != 0:
            resx //= blocksize
            resy //= blocksize

        pixels = np.empty(resx * resy * 3, np.float32)
        scene._fast_export_image(pixels, blocksize)
        self.pixels = bgl.Buffer(bgl.GL_FLOAT, resx * resy * 3, pixels)

        # Generate texture
        self.texture = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenTextures(1, self.texture)
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture[0])
        bgl.glTexImage2D(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGB16F, resx, resy, 0, bgl.GL_RGB, bgl.GL_FLOAT, self.pixels)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_NEAREST)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_NEAREST)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, bgl.GL_CLAMP_TO_EDGE)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, bgl.GL_CLAMP_TO_EDGE)
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
    render_samples: bpy.props.IntProperty(name='Render Samples', min=1, default=128)
    viewport_samples: bpy.props.IntProperty(name='Viewport Samples', min=1, default=32)
    start_pixel_size: bpy.props.IntProperty(name='Start Pixel Size', min=1, default=8)
    smoothing: bpy.props.BoolProperty(name='Smoothing', default=True)
    texturing: bpy.props.BoolProperty(name='Texturing', default=True)


class TinaResetOperator(bpy.types.Operator):
    '''Reset Tina Renderer'''

    bl_idname = "scene.tina_reset"
    bl_label = "Reset Tina"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        bpy.a
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
