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


@ti.data_oriented
class IField:
    is_taichi_class = True

    @ti.func
    def _subscript(self, I):
        raise NotImplementedError

    @ti.func
    def sample(self, pos):
        raise NotImplementedError

    def subscript(self, *indices):
        I = ti.Vector(totuple(indices))
        return self._subscript(I)

    @ti.func
    def __iter__(self):
        raise NotImplementedError


class IShapeField(IField):
    shape = NotImplemented
    dtype = NotImplemented
    vdims = NotImplemented

    @ti.func
    def sample(self, pos):
        p = float(pos)
        I = int(ti.floor(p))
        x = p - I
        y = 1 - x
        ti.static_assert(len(self.shape) == 2)
        return (self[I + V(1, 1)] * x[0] * x[1] +
                self[I + V(1, 0)] * x[0] * y[1] +
                self[I + V(0, 0)] * y[0] * y[1] +
                self[I + V(0, 1)] * y[0] * x[1])

    @ti.func
    def __iter__(self):
        for I in ti.grouped(ti.ndrange(*self.shape)):
            yield I


@ti.data_oriented
class IRunnable:
    @ti.kernel
    def run(self):
        raise NotImplementedError


class FShape(IShapeField):
    def __init__(self, field, shape, dtype=None, vdims=None):
        assert isinstance(field, IField)

        self.field = field
        self.dtype = dtype
        self.shape = totuple(shape)
        self.vdims = totuple(vdims)

    @ti.func
    def _subscript(self, I):
        return self.field[I]


class FLike(IShapeField):
    def __init__(self, src, field):
        assert isinstance(src, IShapeField)
        assert isinstance(field, IField)

        self.field = field
        self.src = src
        self.dtype = self.src.dtype
        self.shape = self.src.shape
        self.vdims = self.src.vdims

    @ti.func
    def _subscript(self, I):
        return self.field[I]



class Field(IShapeField):
    def __init__(self, shape, dtype, vdims):
        self.dtype = dtype
        self.vdims = totuple(vdims)
        self.shape = totuple(shape)

        self.core = self.__mkfield(self.dtype, self.vdims, self.shape)

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
        dtype = self.dtype
        if hasattr(dtype, 'to_string'):
            dtype = 'ti.' + dtype.to_string()
        elif hasattr(dtype, '__name__'):
            dtype = dtype.__name__
        return f'Field({dtype}, {list(self.vdims)}, {list(self.shape)})'

    def __str__(self):
        return str(self.core)

    def __getattr__(self, attr):
        return getattr(self.core, attr)


class FConstant(IField):
    def __init__(self, value):
        self.value = value

    @ti.func
    def _subscript(self, I):
        return self.value


class FMix(IField):
    def __init__(self, f1, f2, k1, k2):
        self.f1 = f1
        self.f2 = f2
        self.k1 = k1
        self.k2 = k2

    @ti.func
    def _subscript(self, I):
        return self.f1[I] * self.k1 + self.f2[I] * self.k2


class FSampleIndex(IField):
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


class FLaplacian(IShapeField):
    def __init__(self, src):
        assert isinstance(src, IShapeField)

        self.src = src
        self.dtype = self.src.dtype
        self.shape = self.src.shape
        self.vdims = self.src.vdims

    @ti.func
    def _subscript(self, I):
        dim = ti.static(len(self.src.shape))
        res = -2 * dim * self.src[I]
        for i in ti.static(range(dim)):
            D = ti.Vector.unit(dim, i)
            res += self.src[I + D] + self.src[I - D]
        return res / (2 * dim)


class FieldCopy(IRunnable):
    def __init__(self, dst, src):
        assert isinstance(dst, Field)
        assert isinstance(src, IField)

        self.dst = dst
        self.src = src

    @ti.kernel
    def run(self):
        for I in ti.static(self.dst):
            self.dst[I] = self.src[I]


@ti.data_oriented
class Canvas:
    def __init__(self, img, res=None):
        assert isinstance(img, IShapeField)

        self.img = img
        self.res = res or (512, 512)

    def _cook(self, color):
        if len(self.img.vdims) == 0:
            color = ti.Vector([color, color, color])
        if self.img.dtype not in [ti.u8, ti.i8]:
            color = ti.max(0, ti.min(255, ti.cast(color * 255 + 0.5, int)))
        return color

    @ti.func
    def _image(self, i, j):
        ti.static_assert(len(self.img.shape) == 2)
        scale = ti.Vector(self.img.shape) / ti.Vector(self.res)
        pos = ti.Vector([i, j]) * scale
        r, g, b = self._cook(self.img.sample(pos))
        return int(r), int(g), int(b)

    @ti.kernel
    def render(self, out: ti.ext_arr(), res: ti.template()):
        for i in range(res[0] * res[1]):
            r, g, b = self._image(i % res[0], res[1] - 1 - i // res[0])
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


img = FShape(FChessboard(32), [512, 512], float, [])
img = FLike(img, FMix(img, FLaplacian(img), 1, 1))
img = FLike(img, FMix(img, FLaplacian(img), 1, 1))
for gui in Canvas(img):
    pass
