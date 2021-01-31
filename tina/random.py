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


@ti.data_oriented
class UnixFastRNG(WangHashRNG):
    @staticmethod
    @ti.func
    def _noise(x):
        value = ti.cast(x, ti.u32)
        value = (value * 7**5) % (2**31 - 1)
        return value


@ti.data_oriented
class HammersleyRNG:
    def __init__(self, seed, time):
        seed = WangHashRNG.noise_int(seed)
        self.seed = ti.expr_init(seed)
        self.time = ti.expr_init(time)

    def __del__(self):
        if hasattr(self, 'seed'):
            del self.seed

    @staticmethod
    @ti.func
    def noise(x):
        i = int(x) & 0x1fffffff
        j = 0
        k = 1
        while i != 0:
            k <<= 1
            j <<= 1
            j |= i & 1
            i >>= 1
        return j / k

    @ti.func
    def random(self):
        ret = self.noise(self.seed + self.time)
        self.seed = WangHashRNG.noise_int(self.seed)
        return ret

    @ti.func
    def random_int(self):
        return int(self.random() * 2**20)


@ti.data_oriented
class SobolSequence:
    def __init__(self, m=32, n=512, skip=8):
        self.m = m
        self.n = n
        self.skip = self.skip0 = skip
        self.data = ti.field(float, (m, n))

        ti.materialize_callback(self.reseed)

    def clear(self):
        self.skip = self.skip0
        self.reseed()

    def reseed(self):
        import sobol
        arr = sobol.i4_sobol_generate(self.m, self.n, self.skip)
        self.data.from_numpy(arr)
        self.skip += 8


@ti.data_oriented
class SobolRNG:
    def __init__(self, seq, index):
        self.seq = seq
        index = WangHashRNG.noise_int(index)
        self.index = ti.expr_init(index)
        self.time = ti.expr_init(0)

    @ti.func
    def random(self):
        ret = self.seq.data[self.index % self.seq.m, self.time % self.seq.n]
        self.time += 1
        return ret

    @ti.func
    def random_int(self):
        return int(self.random() * 2**20)


'''
def binrev(i):
    j = 0
    k = 1
    while i != 0:
        k <<= 1
        j <<= 1
        j |= i & 1
        i >>= 1
    return j / k

def graycode(x):
    return x ^ (x >> 1)
'''
