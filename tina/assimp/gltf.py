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
        print(f'[Tina] Loading GLTF file: {path}...')
        with open(path, 'rb') as f:
            root = json.load(f)
    elif isinstance(path, dict):
        root = path
    else:
        root = json.load(path)

    class Primitive:
        def __init__(self):
            self.attrs = {}
            self.indices = None
            self.material = None

        def set_indices(self, array):
            self.indices = array

        def set_material(self, material):
            self.material = material

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
            material = None
            if self.material is not None:
                material = self.material.extract()
            scene.add_object(mesh, material)

    class Node:
        def __init__(self, name):
            self.name = name
            self.primitives = None
            self.trans = None

        def set_primitives(self, primitives):
            self.primitives = primitives

        def set_transform(self, *trans):
            self.trans = trans

        def __repr__(self):
            return 'Node(' + repr(self.name) + ', ' + repr(self.primitives) + ', ' + repr(self.trans) + ')'

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

            if self.primitives is not None:
                for primitive in self.primitives:
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

    class Material:
        def __init__(self, name):
            self.name = name
            self.pbr = None

        def set_pbr(self, pbr):
            self.pbr = pbr

        def extract(self):
            if self.pbr is None:
                return tina.Lambert()
            kwargs = {}
            for key, value in self.pbr.items():
                if key == 'baseColorFactor':
                    kwargs['basecolor'] = value[:3]
                elif key == 'baseColorTexture':
                    #assert value.get('texCoord', 0) == 0
                    img = images[value['index']]
                    kwargs['basecolor'] = tina.Texture(img)
                elif key == 'metallicFactor':
                    kwargs['metallic'] = value
                elif key == 'metallicTexture':
                    #assert value.get('texCoord', 0) == 0
                    img = images[value['index']]
                    kwargs['metallic'] = tina.Texture(img)
                elif key == 'roughnessFactor':
                    kwargs['roughness'] = value
                elif key == 'roughnessTexture':
                    #assert value.get('texCoord', 0) == 0
                    img = images[value['index']]
                    kwargs['roughness'] = tina.Texture(img)
                elif key == 'metallicRoughnessTexture':
                    img = images[value['index']]
                    print(img.dtype)
                    tina.ti.imshow(img)
                    kwargs['metallic'] = tina.Texture(img[..., 2])
                    kwargs['roughness'] = tina.Texture(img[..., 1])
            return tina.PBR(**kwargs)

    def load_uri(uri):
        if uri.startswith('data:'):
            index = uri.index('base64,')
            data = uri[index + len('base64,'):]
            data = base64.b64decode(data.encode('ascii'))
        else:
            with open(uri, 'rb') as f:
                data = f.read()
        return data

    print(f'[Tina] Loading GLTF buffers...')
    buffers = []
    for buffer in root['buffers']:
        data = load_uri(buffer['uri'])
        buffers.append(data)

    def get_buffer_view_bytes(buffer_view_id):
        buffer_view = root['bufferViews'][buffer_view_id]
        byte_offset = buffer_view['byteOffset']
        byte_length = buffer_view['byteLength']
        buffer = buffers[buffer_view['buffer']]
        buffer = buffer[byte_offset:byte_offset + byte_length]
        return buffer

    def get_accessor_buffer(accessor_id):
        accessor = root['accessors'][accessor_id]

        buffer_view_id = accessor['bufferView']
        buffer = get_buffer_view_bytes(buffer_view_id)

        dtype = component_types[accessor['componentType'] - 0x1400]
        dtype = vector_types[accessor['type']] + dtype
        count = accessor['count']

        array = np.frombuffer(buffer, dtype=dtype, count=count)
        return array

    print(f'[Tina] Loading GLTF images...')
    images = []
    if 'images' in root:
        for image in root['images']:
            #res_imag = Image(image.get('name', 'Untitled'))
            if 'bufferView' in image:
                buffer_view_id = image['bufferView']
                buffer = get_buffer_view_bytes(buffer_view_id)
            else:
                buffer = load_uri(image['uri'])
            from PIL import Image
            from io import BytesIO
            with BytesIO(buffer) as f:
                im = np.array(Image.open(f))
            im = np.swapaxes(im, 0, 1)
            images.append(im)

    print(f'[Tina] Loading GLTF materials...')
    materials = []
    if 'materials' in root:
        for material in root['materials']:
            res_matr = Material(material.get('name', 'Untitled'))
            if 'pbrMetallicRoughness' in material:
                res_matr.set_pbr(material['pbrMetallicRoughness'])
            materials.append(res_matr)

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
                res_prims = []
                for primitive in mesh['primitives']:
                    res_prim = Primitive()
                    if 'indices' in primitive:
                        indices_accessor_id = primitive['indices']
                        indices_array = get_accessor_buffer(indices_accessor_id)
                        res_prim.set_indices(indices_array)
                    for attr_name, accessor_id in primitive['attributes'].items():
                        array = get_accessor_buffer(accessor_id)
                        res_prim.add_attribute(attr_name, array)
                    if 'material' in primitive:
                        material_id = primitive['material']
                        res_prim.set_material(materials[material_id])
                    res_prims.append(res_prim)
                res_node.set_primitives(res_prims)
            res_scene.add_node(res_node)
        return res_scene

    print(f'[Tina] Loading GLTF scene...')
    scene = root['scenes'][root.get('scene', 0)]
    res_scene = parse_scene(scene)

    print(f'[Tina] Loading GLTF done')
    return res_scene


__all__ = ['readgltf']
