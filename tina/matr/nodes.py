from ..advans import *


@ti.data_oriented
class Node:
    arguments = []
    defaults = []

    def __init__(self, **kwargs):
        self.params = {}
        for dfl, key in zip(self.defaults, self.arguments):
            value = kwargs.get(key, None)
            if value is None:
                if dfl is None:
                    raise ValueError(f'`{key}` must specified for `{type(self)}`')
                value = dfl

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
    g_pars = None

    @staticmethod
    def spec_g_pars(pars):
        Input.g_pars = pars

    @staticmethod
    def clear_g_pars():
        Input.g_pars = None

    # noinspection PyMissingConstructor
    def __init__(self, name):
        self.name = name

    @ti.func
    def __call__(self):
        return Input.g_pars[self.name]


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