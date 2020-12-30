import json
import base64
import numpy as np
import tina

def readgltf(path):
    component_types = 'bBhHiIf234d'
    vector_types = {
        'SCALAR': '',
        'VEC2': '2',
        'VEC3': '3',
        'VEC4': '4',
    }

    if isinstance(path, str):
        with open(path, 'rb') as f:
            root = json.load(f)
    elif isinstance(path, dict):
        root = path
    else:
        root = json.load(f)

    class Primitive:
        def __init__(self):
            self.attrs = {}
            self.indices = None

        def set_indices(self, array):
            self.indices = array

        def add_attribute(self, name, array):
            self.attrs[name] = array

        def __repr__(self):
            return 'Primitive(' + repr(self.indices) + ', ' + repr(self.attrs) + ')'

        def to_obj_mesh(self):
            obj = {}
            if self.indices is not None:
                obj['f'] = self.indices.reshape(len(self.indices) // 3, 3)
            for attr_name, array in self.attrs.items():
                if attr_name == 'POSITION':
                    obj['v'] = array
                if attr_name == 'NORMAL':
                    obj['vn'] = array
                if attr_name.startswith('TEXCOORD') and 'vt' not in obj:
                    obj['vt'] = array
            return obj

        def extract(self, scene, trans):
            obj = self.to_obj_mesh()
            mesh = tina.MeshModel(obj)
            mesh = tina.MeshTransform(mesh, trans)
            scene.add_object(mesh)

    class Node:
        def __init__(self, name):
            self.name = name
            self.mesh = None
            self.trans = None

        def set_mesh(self, mesh):
            self.mesh = mesh

        def set_transform(self, *trans):
            self.trans = trans

        def __repr__(self):
            return 'Node(' + repr(self.name) + ', ' + repr(self.mesh) + ', ' + repr(self.trans) + ')'

        def extract(self, scene):
            trans = tina.identity()
            if self.trans is not None:
                offset, rotation, scale = self.trans
                if scale is not None:
                    trans = tina.scale(scale) @ trans
                if rotation is not None:
                    trans = tina.quaternion(rotation) @ trans
                if offset is not None:
                    trans = tina.translate(offset) @ trans

            if self.mesh is None:
                return
            for primitive in self.mesh:
                primitive.extract(scene, trans)

    class Scene:
        def __init__(self, name):
            self.name = name
            self.nodes = []

        def add_node(self, node):
            self.nodes.append(node)

        def __repr__(self):
            return 'Scene(' + repr(self.name) + ', ' + repr(self.nodes) + ')'

        def extract(self, scene):
            for node in self.nodes:
                node.extract(scene)
            return scene

    def load_uri(uri):
        if uri.startswith('data:'):
            magic_string = 'data:application/octet-stream;base64,'
            assert uri.startswith(magic_string)
            data = uri[len(magic_string):]
            data = base64.b64decode(data.encode('ascii'))
        else:
            with open(uri, 'rb') as f:
                data = f.read()
        return data

    buffers = []
    for buffer in root['buffers']:
        data = load_uri(buffer['uri'])
        buffers.append(data)

    def get_accessor_buffer(accessor_id):
        accessor = root['accessors'][accessor_id]

        buffer_view_id = accessor['bufferView']
        buffer_view = root['bufferViews'][buffer_view_id]
        byte_offset = buffer_view['byteOffset']
        byte_length = buffer_view['byteLength']
        buffer = buffers[buffer_view['buffer']]
        buffer = buffer[byte_offset:byte_offset + byte_length]

        dtype = component_types[accessor['componentType'] - 0x1400]
        dtype = vector_types[accessor['type']] + dtype
        count = accessor['count']

        array = np.frombuffer(buffer, dtype=dtype, count=count)
        return array

    def parse_scene(scene):
        res_scene = Scene(scene.get('name', 'Untitled'))
        for node_id in scene['nodes']:
            node = root['nodes'][node_id]
            res_node = Node(node.get('name', 'Untitled'))
            translation = node.get('translation')
            rotation = node.get('rotation')
            scale = node.get('scale')
            res_node.set_transform(translation, rotation, scale)
            if 'mesh' in node:
                mesh_id = node['mesh']
                mesh = root['meshes'][mesh_id]
                res_mesh = []
                for primitive in mesh['primitives']:
                    res_prim = Primitive()
                    if 'indices' in primitive:
                        indices_accessor_id = primitive['indices']
                        indices_array = get_accessor_buffer(indices_accessor_id)
                        res_prim.set_indices(indices_array)
                    for attr_name, accessor_id in primitive['attributes'].items():
                        array = get_accessor_buffer(accessor_id)
                        res_prim.add_attribute(attr_name, array)
                    res_mesh.append(res_prim)
                res_node.set_mesh(res_mesh)
            res_scene.add_node(res_node)
        return res_scene

    scene = root['scenes'][root.get('scene', 0)]
    res_scene = parse_scene(scene)
    return res_scene


__all__ = ['readgltf']
