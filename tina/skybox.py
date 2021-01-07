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
    def __init__(self, path, cubic=False):
        shape = path
        if isinstance(shape, int):
            if cubic:
                shape = shape * 3 // 4, shape
            else:
                shape = 2 * shape, shape
        elif isinstance(shape, str):
            shape = self._from_image(shape)
        else:
            assert isinstance(shape, (list, tuple)), shape
        img = ti.Vector.field(3, float, shape)
        self.img = img
        self.cubic = cubic
        if self.cubic:
            self.resolution = shape[1] * 2 // 3
        else:
            self.resolution = shape[1]
        self.shape = shape

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
    def sample(self, dir):
        if ti.static(self.cubic):
            I = (self.shape[0] - 1) * cubemap(dir)
            return bilerp(self.img, I)
        else:
            I = (V(*self.shape) - 1) * spheremap(dir)
            return bilerp(self.img, I)

    @ti.func
    def mapcoor(self, I):
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

