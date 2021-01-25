from .advans import *


@ti.func
def cubemap(dir):
    eps = 1e-7
    coor = V(0., 0.)
    if dir.z >= 0 and dir.z >= abs(dir.y) - eps and dir.z >= abs(dir.x) - eps:
        coor = V(3 / 8, 3 / 8) + V(dir.x, dir.y) / dir.z / 8
    if dir.z <= 0 and -dir.z >= abs(dir.y) - eps and -dir.z >= abs(dir.x) - eps:
        coor = V(7 / 8, 3 / 8) + V(-dir.x, dir.y) / -dir.z / 8
    if dir.x <= 0 and -dir.x >= abs(dir.y) - eps and -dir.x >= abs(dir.z) - eps:
        coor = V(1 / 8, 3 / 8) + V(dir.z, dir.y) / -dir.x / 8
    if dir.x >= 0 and dir.x >= abs(dir.y) - eps and dir.x >= abs(dir.z) - eps:
        coor = V(5 / 8, 3 / 8) + V(-dir.z, dir.y) / dir.x / 8
    if dir.y >= 0 and dir.y >= abs(dir.x) - eps and dir.y >= abs(dir.z) - eps:
        coor = V(3 / 8, 5 / 8) + V(dir.x, -dir.z) / dir.y / 8
    if dir.y <= 0 and -dir.y >= abs(dir.x) - eps and -dir.y >= abs(dir.z) - eps:
        coor = V(3 / 8, 1 / 8) + V(dir.x, dir.z) / -dir.y / 8
    return coor


@ti.func
def spheremap(dir):
    dir.z, dir.y = dir.y, -dir.z
    u, v = unspherical(dir)
    coor = V(v, u * 0.5 + 0.5)
    return coor


@ti.func
def unspheremap(coor):
    v, u = coor
    dir = spherical(u * 2 - 1, v)
    dir.y, dir.z = dir.z, -dir.y
    return dir


@ti.data_oriented
class Skybox:
    def __init__(self, path, scale=None, cubic=False):
        shape = path
        self.cubic = cubic
        if isinstance(shape, int):
            if cubic:
                shape = shape * 3 // 4, shape
            else:
                shape = 2 * shape, shape
        elif isinstance(shape, np.ndarray):
            shape = self._from_raw(shape)
        elif isinstance(shape, str):
            shape = self._from_image(shape)
        elif isinstance(shape, int):
            if self.cubic:
                shape = shape * 4, shape * 3
            else:
                shape = 2 * shape, shape
        else:
            assert isinstance(shape, (list, tuple)), shape
        self.img = ti.Vector.field(3, float, shape)
        if self.cubic:
            self.resolution = shape[1] * 2 // 3
        else:
            self.resolution = shape[1]
        self.shape = shape

        if scale is not None:
            @ti.materialize_callback
            @ti.kernel
            def scale_skybox():
                for I in ti.grouped(self.img):
                    self.img[I] *= scale

    def cook_from(self, src, nsamples=128):
        @ti.materialize_callback
        @ti.kernel
        def cook_skybox():
            ti.static_print('[Tina] Cooking skybox '
                    f'({"x".join(map(str, self.shape))} {nsamples} spp)...')
            for I in ti.grouped(self.img):
                res = V(0., 0., 0.)
                for i in range(nsamples):
                    J = I + V(ti.random(), ti.random())
                    dir = self.unmapcoor(J)
                    res += src.sample(dir)
                self.img[I] = res / nsamples

        return self

    def _from_raw(self, img):
        @ti.materialize_callback
        def init_skybox():
            @ti.kernel
            def init_skybox(img: ti.ext_arr()):
                for I in ti.grouped(self.img):
                    for k in ti.static(range(3)):
                        self.img[I][k] = img[I, k]
            init_skybox(img)

        return img.shape[:2]

    def _from_image(self, path):
        if not isinstance(path, np.ndarray):
            img = ti.imread(path)
            img = np.float32(img / 255)
        else:
            img = np.array(path)

        @ti.materialize_callback
        def init_skybox():
            @ti.kernel
            def init_skybox(img: ti.ext_arr()):
                for I in ti.grouped(self.img):
                    for k in ti.static(range(3)):
                        self.img[I][k] = ce_untonemap(img[I, k])
            init_skybox(img)

        return img.shape[:2]

    @ti.func
    def mapcoor(self, dir):
        if ti.static(self.cubic):
            coor = cubemap(dir)
            I = (self.shape[0] - 1) * coor
            return I
        else:
            coor = spheremap(dir)
            I = (V(*self.shape) - 1) * coor
            return I

    @ti.func
    def unmapcoor(self, I):
        ti.static_assert(not self.cubic)
        coor = I / (V(*self.shape) - 1)
        dir = unspheremap(coor)
        return dir

    @ti.func
    def sample(self, dir):
        if ti.static(self.cubic):
            I = (self.shape[0] - 1) * cubemap(dir)
            return bilerp(self.img, I)
        else:
            I = (V(*self.shape) - 1) * spheremap(dir)
            return bilerp(self.img, I)



def _get_sample_sky():
    g = 0.76;

    @ti.func
    def henyey_greenstein_phase_func(mu):
        return (1. - g*g) / ((4. * ti.pi) * pow(1. + g*g - 2.*g*mu, 1.5))


    @ti.func
    def isect_sphere(ray_origin, ray_direction, sphere):
        rc = sphere[0] - ray_origin;
        radius2 = sphere[1]**2
        tca = rc.dot(ray_direction);
        d2 = rc.norm_sqr() - tca**2

        hit = 0
        t0 = 0.0
        t1 = 0.0
        if not (d2 > radius2):
            hit = 1
            thc = ti.sqrt(radius2 - d2);
            t0 = tca - thc;
            t1 = tca + thc;

        return hit, t0, t1;

    betaR = V(5.5e-6, 13.0e-6, 22.4e-6);
    betaM = V3(21e-6);

    hR = 7994.0;
    hM = 1200.0;

    @ti.func
    def rayleigh_phase_func(mu):
        return 3. * (1. + mu*mu) / (16. * ti.pi);


    earth_radius = 6360e3;
    atmosphere_radius = 6420e3;
    atmosphere = [V(0, 0, 0), atmosphere_radius, 0];

    num_samples = 16;
    num_samples_light = 8;


    @ti.func
    def get_sun_light(ray_origin, ray_direction, optical_depthR, optical_depthM):
        t0 = 0.0
        t1 = 0.0
        hit, t0, t1 = isect_sphere(ray_origin, ray_direction, atmosphere);

        ret = 0
        if hit:
            ret = 1

            march_pos = 0.;
            march_step = t1 / float(num_samples_light);

            for i in range(num_samples_light):
                s = ray_origin + ray_direction * (march_pos + 0.5 * march_step);
                height = (s).norm() - earth_radius;
                if (height < 0.):
                    ret = 0
                    break

                optical_depthR += ti.exp(-height / hR) * march_step;
                optical_depthM += ti.exp(-height / hM) * march_step;

                march_pos += march_step;

        return ret;



    sun_power = 30.0;
    #sun_size = np.radians(4)

    @ti.func
    def get_incident_light(ray_origin, ray_direction, sun_dir):
        t0 = 0.0
        t1 = 0.0
        hit, t0, t1 = isect_sphere(ray_origin, ray_direction, atmosphere)

        ret = V3(0.)
        if hit:
            mu = ray_direction.dot(sun_dir);

            march_step = t1 / float(num_samples);

            phaseR = rayleigh_phase_func(mu);
            phaseM = henyey_greenstein_phase_func(mu);

            optical_depthR = 0.;
            optical_depthM = 0.;

            sumR = V3(0.);
            sumM = V3(0.);
            march_pos = 0.;

            for i in range(num_samples):
                s = ray_origin + ray_direction * (march_pos + 0.5 * march_step);
                height = (s).norm() - earth_radius;

                hr = ti.exp(-height / hR) * march_step;
                hm = ti.exp(-height / hM) * march_step;
                optical_depthR += hr;
                optical_depthM += hm;

                optical_depth_lightR = 0.;
                optical_depth_lightM = 0.;
                overground = get_sun_light(
                    s, sun_dir,
                    optical_depth_lightR,
                    optical_depth_lightM);

                if (overground):
                    tau = betaR * (optical_depthR + optical_depth_lightR) + betaM * 1.1 * (optical_depthM + optical_depth_lightM);
                    attenuation = ti.exp(-tau);

                    sumR += hr * attenuation;
                    sumM += hm * attenuation;

                march_pos += march_step;

            ret = sun_power * (sumR * phaseR * betaR + sumM * phaseM * betaM)

            #if mu >= ti.cos(sun_size):
            #    ret += V(4.1, 4.0, 3.8) * sun_power

        return ret

    @ti.func
    def sample_sky(dir):
        org = V(0., 0., earth_radius + 1.)
        sun_dir = V(0., -2., 1.).normalized()
        ret = get_incident_light(org, dir, sun_dir)
        return ret

    return sample_sky


# https://www.shadertoy.com/view/XtBXDz
@ti.data_oriented
class Atomsphere:
    def __init__(self):
        self.resolution = 512
        self.sample_sky = _get_sample_sky()

    @ti.func
    def sample(self, dir):
        dir.y, dir.z = -dir.z, dir.y
        sky = self.sample_sky(dir)
        ground = lerp((dir.xy / dir.z // 4).sum() % 2, 0.2, 0.7)
        return lerp(clamp(dir.z * 32, -1, 1) * 0.5 + 0.5, ground, sky)


@ti.data_oriented
class PlainSkybox:
    def __init__(self, color=(1., 1., 1.)):
        self.resolution = 64
        self.color = tovector(color)

    @ti.func
    def sample(self, dir):
        return self.color


@ti.data_oriented
class RotSkybox:
    def __init__(self, skybox):
        self.wraps = skybox

    @ti.func
    def trans(self, dir):
        dir.y, dir.z = dir.z, -dir.y
        return dir

    @ti.func
    def untrans(self, dir):
        dir.y, dir.z = -dir.z, dir.y
        return dir

    @ti.func
    def mapcoor(self, dir):
        self.wraps.mapcoor(self.trans(dir))

    @ti.func
    def unmapcoor(self, I):
        ti.static_assert(not self.cubic)
        coor = I / (V(*self.shape) - 1)
        dir = unspheremap(coor)
        return self.untrans(dir)

    @ti.func
    def sample(self, dir):
        return self.wraps.sample(self.trans(dir))
