from .advans import *


@ti.data_oriented
class TaichiRNG:
    def __init__(self):
        pass

    def random(self):
        return ti.random()

    def random_int(self):
        return ti.random(int)


@ti.data_oriented
class WangHashRNG:
    def __init__(self, seed):
        seed = self.noise_int(seed)
        self.seed = ti.expr_init(seed)

    def __del__(self):
        if hasattr(self, 'seed'):
            del self.seed

    @staticmethod
    @ti.func
    def _noise(x):
        value = ti.cast(x, ti.u32)
        value = (value ^ 61) ^ (value >> 16)
        value *= 9
        value ^= value << 4
        value *= 0x27d4eb2d
        value ^= value >> 15
        return value

    @classmethod
    @ti.func
    def noise(cls, x):
        u = cls.noise_int(x) >> 1
        return u * (2 / 4294967296)

    @classmethod
    @ti.func
    def noise_int(cls, x):
        if ti.static(not isinstance(x, ti.Matrix)):
            return cls._noise(x)
        index = ti.cast(x, ti.u32)
        value = cls._noise(index.entries[0])
        for i in ti.static(index.entries[1:]):
            value = cls._noise(i ^ value)
        return value

    @ti.func
    def random(self):
        ret = self.noise(self.seed)
        self.seed = self.seed + 1
        return ret

    @ti.func
    def random_int(self):
        ret = self.noise_int(self.seed)
        self.seed = self.seed + 1
        return ret
