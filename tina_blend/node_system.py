import bpy
import nodeitems_utils


class TinaNodeTree(bpy.types.NodeTree):
    bl_idname = 'tina_node_tree'
    bl_label = 'Tina Node Tree'
    bl_icon = 'MATERIAL'

    @classmethod
    def poll(cls, context):
        return True


sockets = []


@sockets.append
class TinaValueSocket(bpy.types.NodeSocket):
    bl_idname = 'tina_value_socket'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (0.63, 0.63, 0.63, 1.0)


@sockets.append
class TinaColorSocket(bpy.types.NodeSocket):
    bl_idname = 'tina_color_socket'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (0.78, 0.78, 0.16, 1.0)


@sockets.append
class TinaVectorSocket(bpy.types.NodeSocket):
    bl_idname = 'tina_vector_socket'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (0.39, 0.39, 0.78, 1.0)


@sockets.append
class TinaMaterialSocket(bpy.types.NodeSocket):
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


@register_node
class TinaTestNode(TinaBaseNode):
    bl_idname = 'tina_test_node'
    bl_label = 'Test'
    category = 'Misc'

    def init(self, context):
        self.width = 200
        self.inputs.new('tina_color_socket', 'basecolor')
        self.inputs.new('tina_value_socket', 'roughness')
        self.inputs.new('tina_value_socket', 'metallic')
        self.inputs.new('tina_value_socket', 'specular')
        self.outputs.new('tina_material_socket', 'material')


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
