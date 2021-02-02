from ..advans import *


@ti.data_oriented
class Node:
    arguments = []
    defaults = []

    def __init__(self, **kwargs):
        self.params = {}
        for dfl, key in zip(self.defaults, self.arguments):
            if key not in kwargs:
                if dfl is None:
                    raise ValueError(f'`{key}` must specified for `{type(self)}`')
                value = dfl
            else:
                value = kwargs[key]
                del kwargs[key]

            if isinstance(value, (int, float, ti.Matrix)):
                value = Const(value)
            elif isinstance(value, (list, tuple)):
                value = Const(V(*value))
            elif isinstance(value, str):
                if any(value.endswith(x) for x in ['.png', '.jpg', '.bmp']):
                    value = Texture(value)
                else:
                    value = Input(value)
            self.params[key] = value

        for key in kwargs.keys():
            raise TypeError(
                    f"{type(self).__name__}() got an unexpected keyword argument '{key}', supported keywords are: {self.arguments}")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def param(self, key, *args, **kwargs):
        return self.params[key](*args, **kwargs)


class Const(Node):
    # noinspection PyMissingConstructor
    def __init__(self, value):
        self.value = value

    @ti.func
    def __call__(self):
        return self.value


class Param(Node):
    # noinspection PyMissingConstructor
    def __init__(self, dtype=float, dim=None, initial=0):
        if dim is not None:
            self.value = ti.Vector.field(dim, dtype, ())
        else:
            self.value = ti.field(dtype, ())

        self.initial = initial
        if initial != 0:
            @ti.materialize_callback
            def init_value():
                self.value[None] = self.initial

    @ti.func
    def __call__(self):
        return self.value[None]

    def make_slider(self, gui, title, min=0, max=1, step=0.01):
        self.slider = gui.slider(title, min, max, step)
        self.slider.value = self.initial

        @gui.post_show
        def post_show(gui):
            self.value[None] = self.slider.value


class Input(Node):
    g_pars = []

    @staticmethod
    def spec_g_pars(pars):
        Input.g_pars.insert(0, pars)

    @staticmethod
    def clear_g_pars():
        Input.g_pars.pop(0)

    # noinspection PyMissingConstructor
    def __init__(self, name):
        self.name = name

    @ti.func
    def __call__(self):
        return Input.g_pars[0][self.name]


class Texture(Node):
    arguments = ['texcoord']
    defaults = ['texcoord']

    def __init__(self, path, **kwargs):
        self.texture = texture_as_field(path)
        super().__init__(**kwargs)

    @ti.func
    def __call__(self):
        maxcoor = V(*self.texture.shape) - 1
        coor = self.param('texcoord') * maxcoor
        return bilerp(self.texture, coor)


class ChessboardTexture(Node):
    arguments = ['texcoord', 'size', 'color0', 'color1']
    defaults = ['texcoord', 0.1, 0.4, 0.9]

    @ti.func
    def __call__(self):
        size = self.param('size')
        color0 = self.param('color0')
        color1 = self.param('color1')
        texcoord = self.param('texcoord')
        return lerp((texcoord // size).sum() % 2, color0, color1)


class LerpTexture(Node):
    arguments = ['x0', 'x1', 'y0', 'y1', 'texcoord']
    defaults = [0.0, 1.0, 0.0, 0.0, 'texcoord']

    @ti.func
    def __call__(self):
        x0, x1, y0, y1, uv = tuple(map(self.param, self.arguments))
        return lerp(uv.x, x0, x1) + lerp(uv.y, x0, x1)


class LambdaNode(Node):
    def __init__(self, func, **kwargs):
        self.func = func
        self.arguments = list(kwargs.keys())
        self.defaults = [None for _ in self.arguments]
        super().__init__(**kwargs)

    def __call__(self):
        return self.func(self)


def lambda_node(func):
    import functools

    @functools.wraps(func)
    def wrapped(**kwargs):
        return LambdaNode(func, **kwargs)

    return wrapped
