from ..advans import *


@ti.kernel
def _pre_compute_pars_npars(pars: ti.template()) -> int:
    pars.pre_compute()
    return pars.get_npars()


@ti.kernel
def _export_pars_verts(pars: ti.template(), out: ti.ext_arr()):
    for i in range(out.shape[0]):
        vert = pars.get_particle_position(i)
        for j, v in ti.static(enumerate(vert)):
            out[i, j] = v

@ti.kernel
def _export_pars_sizes(pars: ti.template(), out: ti.ext_arr()):
    for i in range(out.shape[0]):
        size = pars.get_particle_radius(i)
        out[i] = size

@ti.kernel
def _export_pars_colors(pars: ti.template(), out: ti.ext_arr()):
    for i in range(out.shape[0]):
        color = pars.get_particle_color(i)
        for j, c in ti.static(enumerate(color)):
            out[i, j] = c


def export_simple_pars(pars):
    npars = _pre_compute_pars_npars(pars)
    ret = {}
    assert hasattr(pars, 'get_particle_position')
    out = np.empty((npars, 3), dtype=np.float32)
    _export_pars_verts(pars, out)
    ret['v'] = out
    if hasattr(pars, 'get_particle_radius'):
        out = np.empty(npars, dtype=np.float32)
        _export_pars_sizes(pars, out)
        ret['vr'] = out
    if hasattr(pars, 'get_particle_color'):
        out = np.empty((npars, 3), dtype=np.float32)
        _export_pars_colors(pars, out)
        ret['vc'] = out
    return ret
