import bpy

from ..common import *
from ..advans import *
from .cache import IDCache


@ti.data_oriented
class OutputPixelConverter:
    def cook(self, color):
        if isinstance(color, ti.Expr):
            color = ti.Vector([color, color, color])
        elif isinstance(color, ti.Matrix):
            assert color.m == 1, color.m
            if color.n == 1:
                color = ti.Vector([color(0), color(0), color(0)])
            elif color.n == 2:
                color = ti.Vector([color(0), color(1), 0.])
            elif color.n in [3, 4]:
                color = ti.Vector([color(0), color(1), color(2)])
            else:
                assert False, color.n
        return color

    @ti.func
    def dump_body(self, img: ti.template(), use_bilerp: ti.template(),
            is_final: ti.template(), out: ti.template(), i, j, width, height):
        color = V(0., 0., 0.)
        if ti.static(use_bilerp):
            scale = ti.Vector(img.shape) / ti.Vector([width, height])
            pos = ti.Vector([i, j]) * scale
            color = bilerp(img, pos)
        else:
            color = img[i, j]
        color = aces_tonemap(color)
        if ti.static(is_final):
            base = (j * width + i) * 4
            out[base + 0] = color.x
            out[base + 1] = color.y
            out[base + 2] = color.z
            out[base + 3] = 1
        else:
            out[j * width + i] = self.rgb24(color)

    @ti.kernel
    def dump(self, img: ti.template(), use_bilerp: ti.template(),
            is_final: ti.template(), out: ti.ext_arr(), width: int, height: int):
        for ii, jj in ti.ndrange(img.shape[0], img.shape[1]):
            if ti.static(not use_bilerp):
                self.dump_body(img, False, is_final, out, ii, jj, width, height)
            else:
                j = jj
                while True:
                    if j >= height:
                        break
                    i = ii
                    while True:
                        if i >= width:
                            break
                        self.dump_body(img, True, is_final, out, i, j, width, height)
                        i += img.shape[0]
                    j += img.shape[1]

    @staticmethod
    @ti.func
    def rgb24(color):
        r, g, b = clamp(int(color * 255 + 0.5), 0, 255)
        return (b << 16) + (g << 8) + r


class BlenderEngine(tina.PTScene):
    def get_material_of_object(self, object):
        #if not object.tina_material_nodes:
        #    return None
        #material = get_material_from_node_group(object.tina_material_nodes)
        material = tina.Lambert()
        return material

    def __init__(self):
        super().__init__((bpy.context.scene.tina_resolution_x,
                          bpy.context.scene.tina_resolution_y),
            maxfaces=bpy.context.scene.tina_max_faces,
            taa=True)

        self.output = OutputPixelConverter()
        self.cache = IDCache(lambda o: (type(o).__name__, o.name))

        self.meshes = {}
        for object in bpy.context.scene.objects:
            if object.type == 'MESH':
                material = self.get_material_of_object(object)
                mesh = tina.MeshTransform(tina.SimpleMesh())
                self.meshes[object.name] = mesh, material
                self.add_object(mesh, material)

        self.skybox = tina.Atomsphere()

        self.color = ti.Vector.field(3, float, self.res)
        self.accum_count = 0
        self.need_update = True

    def render_scene(self, is_final):
        if self.need_update:
            self.update()
            self.need_update = False

        for object in bpy.context.scene.objects:
            if object.type == 'MESH':
                self.update_object(object)

        self.render()
        #import code; code.interact(local=locals())
        self.accum_count += 1

    def clear_samples(self):
        self.clear()
        self.accum_count = 0

    def is_need_redraw(self):
        return self.accum_count < bpy.context.scene.tina_viewport_samples

    def update_object(self, object):
        verts, norms, coors, world = self.cache.lookup(blender_get_object_mesh, object)
        if not len(verts):
            return

        mesh, material = self.meshes[object.name]
        mesh.set_transform(world)
        mesh.set_face_verts(verts)
        mesh.set_face_norms(norms)
        mesh.set_face_coors(coors)

    def update_region_data(self, region3d):
        pers = np.array(region3d.perspective_matrix)
        view = np.array(region3d.view_matrix)
        proj = pers @ np.linalg.inv(view)
        self.engine.set_camera(view, proj)

    def update_default_camera(self):
        camera = bpy.context.scene.camera
        render = bpy.context.scene.render
        depsgraph = bpy.context.evaluated_depsgraph_get()
        scale = render.resolution_percentage / 100.0
        proj = np.array(camera.calc_matrix_camera(depsgraph,
            x=render.resolution_x * scale, y=render.resolution_y * scale,
            scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y))
        view = np.linalg.inv(np.array(camera.matrix_world))
        self.engine.set_camera(view, proj)

    def dump_pixels(self, pixels, width, height, is_final):
        if isinstance(self, tina.PTScene):
            self.engine._get_image_f(self.color)
        else:
            self.color.copy_from(self.img)
        use_bilerp = not (width == self.res.x and height == self.res.y)
        self.output.dump(self.color, use_bilerp, is_final, pixels, width, height)

    def invalidate_callback(self, updates):
        for update in updates:
            object = update.id
            if update.is_updated_geometry:
                self.cache.invalidate(object)
            self.need_update = True


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


def blender_get_object_mesh(object):
    import bmesh
    bm = bmesh.new()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    object_eval = object.evaluated_get(depsgraph)
    bm.from_object(object_eval, depsgraph)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    verts = bmesh_verts_to_numpy(bm)[bmesh_faces_to_numpy(bm)]
    norms = bmesh_face_norms_to_numpy(bm)
    coors = bmesh_face_coors_to_numpy(bm)
    world = np.array(object.matrix_world)
    return verts, norms, coors, world
