import taichi as ti


ti.static = lambda x, *xs: [x] + list(xs) if xs else x


def V(*xs):
    return ti.Vector(xs)


def totuple(x):
    if isinstance(x, ti.Matrix):
        x = x.entries
    if isinstance(x, list):
        x = tuple(x)
    if not isinstance(x, tuple):
        x = x,
    return x


def tovector(x):
    return ti.Vector(totuple(x))


@ti.func
def clamp(x, xmin, xmax):
    return min(xmax, max(xmin, x))


@ti.func
def bilerp(f: ti.template(), pos):
    p = float(pos)
    I = int(ti.floor(p))
    x = p - I
    y = 1 - x
    ti.static_assert(len(f.meta.shape) == 2)
    return (f[I + V(1, 1)] * x[0] * x[1] +
            f[I + V(1, 0)] * x[0] * y[1] +
            f[I + V(0, 0)] * y[0] * y[1] +
            f[I + V(0, 1)] * y[0] * x[1])


@ti.data_oriented
class IField:
    is_taichi_class = True

    @ti.func
    def _subscript(self, I):
        raise NotImplementedError

    def subscript(self, *indices):
        I = tovector(indices)
        return self._subscript(I)

    @ti.func
    def __iter__(self):
        raise NotImplementedError


@ti.data_oriented
class Meta:
    is_taichi_class = True

    def __init__(self, shape, dtype=None, vdims=None):
        self.dtype = dtype
        self.shape = totuple(shape)
        self.vdims = totuple(vdims)

    def __repr__(self):
        dtype = self.dtype
        if hasattr(dtype, 'to_string'):
            dtype = 'ti.' + dtype.to_string()
        elif hasattr(dtype, '__name__'):
            dtype = dtype.__name__
        return f'Meta({dtype}, {list(self.vdims)}, {list(self.shape)})'


class IShapeField(IField):
    meta = NotImplemented

    @ti.func
    def __iter__(self):
        for I in ti.grouped(ti.ndrange(*self.meta.shape)):
            yield I


@ti.data_oriented
class IRun:
    @ti.kernel
    def run(self):
        raise NotImplementedError


class FShape(IShapeField):
    def __init__(self, field, meta):
        assert isinstance(field, IField)
        assert isinstance(meta, Meta)

        self.field = field
        self.meta = meta

    @ti.func
    def _subscript(self, I):
        return self.field[I]


class FLike(IShapeField):
    def __init__(self, src, field):
        assert isinstance(src, IShapeField)
        assert isinstance(field, IField)

        self.field = field
        self.src = src
        self.meta = self.src.meta

    @ti.func
    def _subscript(self, I):
        return self.field[I]


class FCache(IShapeField, IRun):
    def __init__(self, src):
        assert isinstance(src, IShapeField)

        self.src = src
        self.meta = self.src.meta
        self.buf = Field(self.meta)

    @ti.kernel
    def run(self):
        for I in ti.static(self.src):
            self.buf[I] = self.src[I]

    @ti.func
    def _subscript(self, I):
        return self.buf[I]


class FDouble(IShapeField, IRun):
    def __init__(self, src):
        assert isinstance(src, IShapeField)

        self.src = src
        self.meta = self.src.meta
        self.cur = Field(self.meta)
        self.nxt = Field(self.meta)

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur

    def run(self):
        self._run(self.nxt, self.src)
        self.swap()

    @ti.kernel
    def _run(self, nxt: ti.template(), src: ti.template()):
        for I in ti.static(src):
            nxt[I] = src[I]

    @ti.func
    def _subscript(self, I):
        return self.cur[I]


class Field(IShapeField):
    def __init__(self, meta):
        assert isinstance(meta, Meta)

        self.meta = meta
        self.core = self.__mkfield(meta.dtype, meta.vdims, meta.shape)

    @staticmethod
    def __mkfield(dtype, vdims, shape):
        if len(vdims) == 0:
            return ti.field(dtype, shape)
        elif len(vdims) == 1:
            return ti.Vector.field(vdims[0], dtype, shape)
        elif len(vdims) == 2:
            return ti.Matrix.field(vdims[0], vdims[1], dtype, shape)
        else:
            assert False, vdims

    def subscript(self, *indices):
        return ti.subscript(self.core, *indices)

    @ti.func
    def __iter__(self):
        for I in ti.grouped(self.core):
            yield I

    def __repr__(self):
        return f'Field({self.meta})'

    def __str__(self):
        return str(self.core)

    def __getattr__(self, attr):
        return getattr(self.core, attr)


class FConst(IField):
    def __init__(self, value):
        self.value = value

    @ti.func
    def _subscript(self, I):
        return self.value


class FClamp(IField):
    def __init__(self, src, min=0, max=1):
        assert isinstance(src, IField)

        self.src = src
        self.min = min
        self.max = max

    @ti.func
    def _subscript(self, I):
        return clamp(self.src[I], self.min, self.max)


class FBound(IShapeField):
    def __init__(self, src):
        assert isinstance(src, IShapeField)

        self.src = src
        self.meta = self.src.meta

    @ti.func
    def _subscript(self, I):
        return self.src[clamp(I, 0, ti.Vector(self.meta.shape) - 1)]


class FRepeat(IShapeField):
    def __init__(self, src):
        assert isinstance(src, IShapeField)

        self.src = src
        self.meta = self.src.meta

    @ti.func
    def _subscript(self, I):
        return self.src[I % ti.Vector(self.meta.shape)]


class FMix(IField):
    def __init__(self, f1, f2, k1=1, k2=1):
        self.f1 = f1
        self.f2 = f2
        self.k1 = k1
        self.k2 = k2

    @ti.func
    def _subscript(self, I):
        return self.f1[I] * self.k1 + self.f2[I] * self.k2


class FMult(IField):
    def __init__(self, field, scale):
        self.field = field
        self.scale = scale

    @ti.func
    def _subscript(self, I):
        return self.field[I] * self.scale


class FSamIndex(IField):
    def __init__(self):
        pass

    @ti.func
    def _subscript(self, I):
        return I


class FChessboard(IField):
    def __init__(self, size):
        self.size = size

    @ti.func
    def _subscript(self, I):
        return (I // self.size).sum() % 2


class FGaussDist(IField):
    def __init__(self, center, radius, height=1):
        self.center = tovector(center)
        self.radius = radius
        self.height = height

    @ti.func
    def _subscript(self, I):
        r2 = (I - self.center).norm_sqr() / self.radius**2
        return self.height * ti.exp(-r2)


class FLaplacian(IShapeField):
    def __init__(self, src):
        assert isinstance(src, IShapeField)

        self.src = src
        self.meta = self.src.meta

    @ti.func
    def _subscript(self, I):
        dim = ti.static(len(self.meta.shape))
        res = -2 * dim * self.src[I]
        for i in ti.static(range(dim)):
            D = ti.Vector.unit(dim, i)
            res += self.src[I + D] + self.src[I - D]
        return res / (2 * dim)


class RFCopy(IRun):
    def __init__(self, dst, src):
        assert isinstance(dst, IShapeField)
        assert isinstance(src, IField)

        self.dst = dst
        self.src = src

    @ti.kernel
    def run(self):
        for I in ti.static(self.dst):
            self.dst[I] = self.src[I]


class RMerge(IRun):
    def __init__(self, *tasks):
        assert all(isinstance(t, IRun) for t in tasks)

        self.tasks = tasks

    def run(self):
        for t in self.tasks:
            t.run()


class RTimes(IRun):
    def __init__(self, task, times):
        assert isinstance(task, IRun)

        self.task = task
        self.times = times

    def run(self):
        for i in range(self.times):
            self.task.run()


@ti.data_oriented
class Canvas:
    def __init__(self, img, res=None):
        assert isinstance(img, IShapeField)

        self.img = img
        self.res = res or (512, 512)

    def _cook(self, color):
        if len(self.img.meta.vdims) == 0:
            color = ti.Vector([color, color, color])
        if self.img.meta.dtype not in [ti.u8, ti.i8]:
            color = ti.max(0, ti.min(255, ti.cast(color * 255 + 0.5, int)))
        return color

    @ti.func
    def image_at(self, i, j):
        ti.static_assert(len(self.img.meta.shape) == 2)
        scale = ti.Vector(self.img.meta.shape) / ti.Vector(self.res)
        pos = ti.Vector([i, j]) * scale
        r, g, b = self._cook(bilerp(self.img, pos))
        return int(r), int(g), int(b)

    @ti.kernel
    def render(self, out: ti.ext_arr(), res: ti.template()):
        for i in range(res[0] * res[1]):
            r, g, b = self.image_at(i % res[0], res[1] - 1 - i // res[0])
            if ti.static(ti.get_os_name() != 'osx'):
                out[i] = (r << 16) + (g << 8) + b
            else:
                alpha = -16777216
                out[i] = (b << 16) + (g << 8) + r + alpha

    def __iter__(self):
        gui = ti.GUI(res=self.res, fast_gui=True)
        while gui.running:
            gui.get_event(None)
            gui.running = not gui.is_pressed(gui.ESCAPE)
            yield gui
            self.render(gui.img, gui.res)
            gui.show()


def FLaplacianBlur(x):
    return FLike(x, FMix(x, FLaplacian(FBound(x)), 1, 1))


def FLaplacianStep(pos, vel, kappa):
    return FLike(pos, FMix(vel, FLaplacian(FBound(pos)), 1, kappa))


def FPosAdvect(pos, vel, dt):
    return FLike(pos, FMix(pos, vel, 1, dt))
