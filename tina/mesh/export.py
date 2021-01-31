from ..advans import *


@ti.kernel
def _pre_compute_mesh_nfaces(mesh: ti.template()) -> int:
    mesh.pre_compute()
    return mesh.get_nfaces()


@ti.kernel
def _export_mesh_verts(mesh: ti.template(), out: ti.ext_arr()):
    for i in range(out.shape[0]):
        verts = mesh.get_face_verts(i)
        for j, vert in ti.static(enumerate(verts)):
            for k, v in ti.static(enumerate(vert)):
                out[i, j, k] = v

@ti.kernel
def _export_mesh_norms(mesh: ti.template(), out: ti.ext_arr()):
    for i in range(out.shape[0]):
        norms = mesh.get_face_norms(i)
        for j, norm in ti.static(enumerate(norms)):
            for k, n in ti.static(enumerate(norm)):
                out[i, j, k] = n

@ti.kernel
def _export_mesh_coors(mesh: ti.template(), out: ti.ext_arr()):
    for i in range(out.shape[0]):
        coors = mesh.get_face_coors(i)
        for j, coor in ti.static(enumerate(coors)):
            for k, c in ti.static(enumerate(coor)):
                out[i, j, k] = c


@ti.kernel
def _export_face_indices(mesh: ti.template(), out: ti.ext_arr()):
    for i in range(out.shape[0]):
        indices = mesh.get_face_indices(i)
        for j, v in ti.static(enumerate(indices)):
            out[i, j] = v


@ti.kernel
def _export_indiced_verts(mesh: ti.template(), out: ti.ext_arr()):
    for i in range(out.shape[0]):
        vert = mesh.get_indiced_vert(i)
        for j, v in ti.static(enumerate(vert)):
            out[i, j] = v


@ti.kernel
def _export_indiced_norms(mesh: ti.template(), out: ti.ext_arr()):
    for i in range(out.shape[0]):
        norm = mesh.get_indiced_norm(i)
        for j, v in ti.static(enumerate(norm)):
            out[i, j] = v


@ti.kernel
def _export_indiced_coors(mesh: ti.template(), out: ti.ext_arr()):
    for i in range(out.shape[0]):
        coor = mesh.get_indiced_coor(i)
        for j, v in ti.static(enumerate(coor)):
            out[i, j] = v


def export_simple_mesh(mesh):
    nfaces = _pre_compute_mesh_nfaces(mesh)
    npolygon = 3
    ret = {}
    if hasattr(mesh, 'get_npolygon'):
        npolygon = mesh.get_npolygon()
    assert hasattr(mesh, 'get_face_verts')
    out = np.empty((nfaces, npolygon, 3), dtype=np.float32)
    _export_mesh_verts(mesh, out)
    ret['fv'] = out
    if hasattr(mesh, 'get_face_norms'):
        out = np.empty((nfaces, npolygon, 3), dtype=np.float32)
        _export_mesh_norms(mesh, out)
        ret['fn'] = out
    if hasattr(mesh, 'get_face_coors'):
        out = np.empty((nfaces, npolygon, 2), dtype=np.float32)
        _export_mesh_coors(mesh, out)
        ret['ft'] = out
    return ret


def simple_mesh_to_connective(obj):
    nfaces = len(obj['fv'])
    npolygon = len(obj['fv'][0])
    ret = {}
    ret['f'] = np.arange(nfaces * npolygon).reshape(nfaces, npolygon)
    ret['v'] = obj['fv'].reshape(nfaces * npolygon, 3)
    if 'fvn' in obj:
        ret['vn'] = obj['fvn'].reshape(nfaces * npolygon, 3)
    if 'fvt' in obj:
        ret['vt'] = obj['fvt'].reshape(nfaces * npolygon, 3)
    return ret


def export_connective_mesh(mesh):
    if not hasattr(mesh, 'get_face_indices'):
        obj = export_simple_mesh(mesh)
        return simple_mesh_to_connective(obj)
    nfaces = _pre_compute_mesh(mesh)
    npolygon = 3
    ret = {}
    if hasattr(mesh, 'get_npolygon'):
        npolygon = mesh.get_npolygon()
    indices = np.empty((nfaces, npolygon), dtype=np.float32)
    _export_mesh_indices(indices)
    ret['f'] = indices
    nverts = np.max(indices)
    if hasattr(mesh, 'get_indiced_vert'):
        out = np.empty((nverts, npolygon, 3), dtype=np.float32)
        _export_mesh_indiced_verts(mesh, out)
        ret['v'] = out
    if hasattr(mesh, 'get_indiced_norm'):
        out = np.empty((nverts, npolygon, 3), dtype=np.float32)
        _export_mesh_indiced_norms(mesh, out)
        ret['vn'] = out
    if hasattr(mesh, 'get_indiced_coor'):
        out = np.empty((nverts, npolygon, 2), dtype=np.float32)
        _export_mesh_indiced_coors(mesh, out)
        ret['vt'] = out
    return ret
