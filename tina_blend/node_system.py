import bpy
import nodeitems_utils
import tina


class TinaNodeTree(bpy.types.NodeTree):
    bl_idname = 'tina_node_tree'
    bl_label = 'Tina Node Tree'
    bl_icon = 'MATERIAL'

    @classmethod
    def poll(cls, context):
        return True


sockets = []


class TinaBaseSocket(bpy.types.NodeSocket):
    def dfs(self, cache):
        if not self.links:
            return self.value
        else:
            assert len(self.links) == 1
            link = self.links[0]
            if link.from_node not in cache:
                res = link.from_node.construct(cache)
                cache[link.from_node] = res
            res = cache[link.from_node]
            cache[link.from_socket] = res
            return res


@sockets.append
class TinaValueSocket(TinaBaseSocket):
    bl_idname = 'tina_value_socket'

    value : bpy.props.FloatProperty(name='value', soft_min=0, precision=3)

    def draw(self, context, layout, node, text):
        if not self.links:
            layout.prop(self, 'value', text=text)
        else:
            layout.label(text=text)

    def draw_color(self, context, node):
        return (0.63, 0.63, 0.63, 1.0)


@sockets.append
class TinaClampedValueSocket(TinaBaseSocket):
    bl_idname = 'tina_clamped_value_socket'

    value : bpy.props.FloatProperty(name='value', min=0, max=1, precision=3)

    def draw(self, context, layout, node, text):
        if not self.links:
            layout.prop(self, 'value', text=text)
        else:
            layout.label(text=text)

    def draw_color(self, context, node):
        return (0.63, 0.63, 0.63, 1.0)


@sockets.append
class TinaColorSocket(TinaBaseSocket):
    bl_idname = 'tina_color_socket'

    value : bpy.props.FloatVectorProperty(name='value', subtype='COLOR', size=3, min=0, max=1, default=(1, 1, 1))

    def draw(self, context, layout, node, text):
        if not self.links:
            layout.prop(self, 'value', text=text)
        else:
            layout.label(text=text)

    def draw_color(self, context, node):
        return (0.78, 0.78, 0.16, 1.0)


@sockets.append
class TinaVectorSocket(TinaBaseSocket):
    bl_idname = 'tina_vector_socket'

    value : bpy.props.FloatVectorProperty(name='value', size=3)

    def draw(self, context, layout, node, text):
        if not self.links:
            layout.prop(self, 'value', text=text)
        else:
            layout.label(text=text)

    def draw_color(self, context, node):
        return (0.39, 0.39, 0.78, 1.0)


@sockets.append
class TinaMaterialSocket(TinaBaseSocket):
    bl_idname = 'tina_material_socket'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (0.39, 0.78, 0.39, 1.0)


class TinaBaseCategory(nodeitems_utils.NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'tina_node_tree'


categories_map = {}
nodes = []

def register_node(cls):
    nodes.append(cls)
    categories_map.setdefault(cls.category, [])
    categories_map[cls.category].append(cls.bl_idname)


class TinaBaseNode(bpy.types.Node):
    @classmethod
    def poll(cls, node_tree):
        return node_tree.bl_idname == 'tina_node_tree'

    def lut(self, cache, input_name):
        socket = self.inputs[input_name]
        if socket not in cache:
            cache[socket] = socket.dfs(cache)
        return cache[socket]

    def construct(self, cache):
        raise NotImplementedError


@register_node
class TinaPBRNode(TinaBaseNode):
    bl_idname = 'tina_pbr_node'
    bl_label = 'PBR'
    category = 'Material'

    def init(self, context):
        self.width = 150
        self.inputs.new('tina_color_socket', 'basecolor')
        self.inputs.new('tina_clamped_value_socket', 'roughness')
        self.inputs.new('tina_clamped_value_socket', 'metallic')
        self.inputs.new('tina_clamped_value_socket', 'specular')
        self.outputs.new('tina_material_socket', 'material')

    def construct(self, cache):
        return tina.PBR(basecolor=list(self.lut(cache, 'basecolor')),
                roughness=self.lut(cache, 'roughness'),
                metallic=self.lut(cache, 'metallic'),
                specular=self.lut(cache, 'specular'))


@register_node
class TinaDiffuseNode(TinaBaseNode):
    bl_idname = 'tina_diffuse_node'
    bl_label = 'Diffuse'
    category = 'Material'

    def init(self, context):
        self.width = 150
        self.inputs.new('tina_color_socket', 'color')
        self.outputs.new('tina_material_socket', 'material')

    def construct(self, cache):
        return tina.Diffuse(color=list(self.lut(cache, 'color')))


@register_node
class TinaEmissionNode(TinaBaseNode):
    bl_idname = 'tina_emission_node'
    bl_label = 'Emission'
    category = 'Material'

    def init(self, context):
        self.width = 150
        self.inputs.new('tina_color_socket', 'color')
        self.inputs.new('tina_value_socket', 'strength')
        self.outputs.new('tina_material_socket', 'material')

    def construct(self, cache):
        return tina.Emission() * list(self.lut(cache, 'color'))


@register_node
class TinaGlassNode(TinaBaseNode):
    bl_idname = 'tina_glass_node'
    bl_label = 'Glass'
    category = 'Material'

    def init(self, context):
        self.width = 150
        self.inputs.new('tina_color_socket', 'color')
        self.inputs.new('tina_value_socket', 'ior')
        self.outputs.new('tina_material_socket', 'material')

    def construct(self, cache):
        return tina.Glass(
                color=list(self.lut(cache, 'color')),
                ior=self.lut(cache, 'ior'))


@register_node
class TinaMaterialOutputNode(TinaBaseNode):
    bl_idname = 'tina_material_output_node'
    bl_label = 'Material Output'
    category = 'Output'

    def init(self, context):
        self.width = 100
        self.inputs.new('tina_material_socket', 'material')

    def construct(self, cache):
        return self.lut(cache, 'material')


def construct_node_tree(tree):
    cache = {}
    material = tree.nodes['Material Output'].construct(cache)
    return material


def register():
    bpy.utils.register_class(TinaNodeTree)
    for socket in sockets:
        bpy.utils.register_class(socket)

    for node in nodes:
        bpy.utils.register_class(node)

    categories = []
    for category_name, node_names in categories_map.items():
        items = []
        for node_name in node_names:
            items.append(nodeitems_utils.NodeItem(node_name))
        category = TinaBaseCategory(category_name, category_name.capitalize(),
                items=items)
        categories.append(category)
    nodeitems_utils.register_node_categories('tina_node_tree', categories)


def unregister():
    for socket in reversed(sockets):
        bpy.utils.unregister_class(socket)

    for node in reversed(nodes):
        bpy.utils.unregister_class(node)

    bpy.utils.unregister_class(TinaNodeTree)
    nodeitems_utils.unregister_node_categories('tina_node_tree')


'''''
'''''
