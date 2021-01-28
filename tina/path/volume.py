from ..advans import *
from .geometry import *


@ti.data_oriented
class VolumeTracer:
    is_dedicated_tracer = True

    def __init__(self, **extra_options):
        pass

    def clear_objects(self):
        pass

    def add_object(self, voxl, mtlid):
        self.voxl = voxl

    @ti.func
    def sample_density(self, pos):
        return self.voxl.sample_volume(pos)
        #return 2.6 if (pos // 0.5).sum() % 2 == 0 else 0.0

    @ti.func
    def hit(self, ro, rd, maxfar=inf):
        # travel volume
        bmin, bmax = self.voxl.get_bounding_box()
        near, far = tina.ray_aabb_hit(bmin, bmax, ro, rd)

        hitind = -1
        depth = inf
        if near <= far:
            near = max(near, 0)
            far = min(far, maxfar)

            t = near
            int_rho = 0.0
            ran = -ti.log(ti.random())
            step = 0.01
            t += step * (.5 + .5 * ti.random())
            while t < far:
                dt = min(step, far - t)
                pos = ro + t * rd
                rho = self.sample_density(pos)
                int_rho += rho * dt
                if ran < int_rho:  # ti.random() > ti.exp(-Integrate rho*dt)
                    depth = t
                    hitind = 0
                    break
                t += dt

        return depth, hitind, V(0., 0.)

    @ti.func
    def calc_geometry(self, near, ind, uv, ro, rd):
        nrm = V(0., 0., 0.)
        return nrm, V(0., 0.)

    @ti.func
    def sample_light_pos(self, org):
        return V(0., 0., 0.), 0, 0.0

    @ti.kernel
    def update_emission(self, mtltab: ti.template()):
        pass

    def update(self):
        pass

    @ti.func
    def get_material_id(self, ind):
        return 1
