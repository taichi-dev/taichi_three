import bpy

from . import worker


class TinaRenderPanel(bpy.types.Panel):
    '''Tina render options'''

    bl_label = 'Tina'
    bl_idname = 'RENDER_PT_tina'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, 'tina_backend')
        layout.prop(scene, 'tina_resolution_x')
        layout.prop(scene, 'tina_resolution_y')
        layout.prop(scene, 'tina_max_faces')
        layout.prop(scene, 'tina_max_lights')
        layout.prop(scene, 'tina_smoothing')
        layout.prop(scene, 'tina_texturing')
        layout.prop(scene, 'tina_culling')
        layout.prop(scene, 'tina_clipping')
        layout.prop(scene, 'tina_viewport_samples')
        layout.prop(scene, 'tina_render_samples')


class TinaMaterialPanel(bpy.types.Panel):
    '''Tina material options'''

    bl_label = 'Tina'
    bl_idname = 'MATERIAL_PT_tina'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'material'

    def draw(self, context):
        layout = self.layout
        object = context.object

        layout.prop_search(object, 'tina_material_nodes', bpy.data, 'node_groups')


def on_param_update(self, context):
    worker.start_main()


def register():
    bpy.types.Object.tina_material_nodes = bpy.props.StringProperty(name='Material', update=on_param_update)
    bpy.types.Scene.tina_backend = bpy.props.EnumProperty(name='Backend', items=[(item.upper(), item, '') for item in ['CPU', 'GPU', 'CUDA', 'OpenGL', 'Metal', 'CC']], update=on_param_update)
    bpy.types.Scene.tina_resolution_x = bpy.props.IntProperty(name='Resolution X', min=1, soft_min=1, subtype='PIXEL', default=1024, update=on_param_update)
    bpy.types.Scene.tina_resolution_y = bpy.props.IntProperty(name='Resolution Y', min=1, soft_min=1, subtype='PIXEL', default=768, update=on_param_update)
    bpy.types.Scene.tina_max_faces = bpy.props.IntProperty(name='Max Faces Count', min=1, soft_min=1, default=65536, update=on_param_update)
    bpy.types.Scene.tina_max_lights = bpy.props.IntProperty(name='Max Lights Count', min=1, soft_min=1, default=8, update=on_param_update)
    bpy.types.Scene.tina_smoothing = bpy.props.BoolProperty(name='Smooth shading', default=True, update=on_param_update)
    bpy.types.Scene.tina_texturing = bpy.props.BoolProperty(name='Using textures', default=True, update=on_param_update)
    bpy.types.Scene.tina_culling = bpy.props.BoolProperty(name='Back face culling', default=True, update=on_param_update)
    bpy.types.Scene.tina_clipping = bpy.props.BoolProperty(name='Viewbox clipping', default=False, update=on_param_update)
    bpy.types.Scene.tina_viewport_samples = bpy.props.IntProperty(name='Viewport Samples', default=1)
    bpy.types.Scene.tina_render_samples = bpy.props.IntProperty(name='Render Samples', default=32)

    bpy.utils.register_class(TinaMaterialPanel)
    bpy.utils.register_class(TinaRenderPanel)


def unregister():
    bpy.utils.unregister_class(TinaRenderPanel)
    bpy.utils.unregister_class(TinaMaterialPanel)

    del bpy.types.Object.tina_material_nodes
    del bpy.types.Scene.tina_backend
    del bpy.types.Scene.tina_resolution_x
    del bpy.types.Scene.tina_resolution_y
    del bpy.types.Scene.tina_max_faces
    del bpy.types.Scene.tina_max_lights
    del bpy.types.Scene.tina_smoothing
    del bpy.types.Scene.tina_texturing
    del bpy.types.Scene.tina_culling
    del bpy.types.Scene.tina_clipping
    del bpy.types.Scene.tina_viewport_samples
    del bpy.types.Scene.tina_render_samples
