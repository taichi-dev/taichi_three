from .common import *

@ti.data_oriented
class Memory:
    is_taichi_class = True

    def __init__(self, dtype, size=2**20):
        self.data = ti.field(dtype, size)
        self.dtype = dtype
        self.size = size
        self.man = MemoryAllocator(size)

    @ti.pyfunc
    def subscript(self, index):
        if ti.static(isinstance(index, slice)):
            ti.static_assert(ti.static(index.step is None))
            index_start = 0
            index_stop = self.size
            if ti.static(index.start is not None):
                index_start = index.start
            if ti.static(index.stop is not None):
                index_stop = index.stop
            return MemoryView(self, index_start, max(0, index_stop - index_start))
        return self.data[index]

    __getitem__ = subscript

    def __setitem__(self, index, value):
        self.data[index] = value

    @ti.kernel
    def _to_numpy(arr: ti.ext_arr(), base: int, size: int):
        for i in range(size):
            arr[i] = self.data[base + i]

    def to_numpy(self, base, size):
        arr = np.empty(size, ti.to_numpy_type(self.dtype))
        self._to_numpy(arr, base, size)
        return arr

    @ti.kernel
    def _from_numpy(arr: ti.ext_arr(), base: int, size: int):
        for i in range(size):
            self.data[base + i] = arr[i]

    def from_numpy(self, arr, base, size):
        assert arr.shape[0] == size, f'{arr.shape[0]} != {size}'
        self.mem._from_numpy(arr, base, size)

    def malloc_memory_view(self, size):
        base = self.man.malloc(size)
        return self[base:base + size]

    def free_memory_view(self, ptr):
        assert isinstance(ptr, MemoryView)
        return self.man.free(ptr.base)

    def __repr__(self):
        return f'Memory(dtype={self.dtype!r}, size={self.size!r}, id={hex(id(self))})'

    def variable(self):
        return self

class MemoryAllocator:
    def __init__(self, size):
        self.size = size
        self.free_chunk = [(0, self.size)]
        self.used_chunk = []

    def malloc(self, size):
        for i, (chk_base, chk_size) in enumerate(self.free_chunk):
            if chk_size >= size: 
                del self.free_chunk[i]
                if chk_size != size:
                    rest_chunk = (chk_base + size, chk_size - size)
                    self.free_chunk.insert(i, rest_chunk)
                base = chk_base
                break
        else:
            raise RuntimeError('Out of memory!')
        self.used_chunk.append((base, size))
        return base

    def free(self, base):
        for i, (chk_base, chk_size) in enumerate(self.used_chunk):
            if chk_base == base:
                del self.used_chunk[i]
                size = chk_size
                break
        else:
            raise RuntimeError(f'Invalid pointer: {base!r}')

        new_chunk = (base, size)
        self.free_chunk.insert(i, new_chunk)

@ti.data_oriented
class MemoryView:
    is_taichi_class = True

    def __init__(self, root, base, size):
        self.root = root
        self.base = base
        self.size = size

    @ti.pyfunc
    def subscript(self, index):
        if ti.static(isinstance(index, slice)):
            ti.static_assert(ti.static(index.step is None))
            index_start = 0
            index_stop = self.size
            if ti.static(index.start is not None):
                index_start = index.start
            if ti.static(index.stop is not None):
                index_stop = index.stop
            base = self.base + index_start
            size = max(0, min(self.size, index_stop) - index_start)
            return MemoryView(self.root, base, size)
        return self.root[self.base + index]

    __getitem__ = subscript

    def __setitem__(self, index, value):
        self.root[self.base + index] = value

    def to_numpy(self):
        self.root.to_numpy(self.base, self.size)

    def from_numpy(self, arr):
        self.root.from_numpy(arr, self.base, self.size)

    def __repr__(self):
        return f'MemoryView({self.root!r}, base={self.base!r}, size={self.size!r})'

    def variable(self):
        return self


class Buffer(MemoryView):
    def __init__(self, root, size):
        self.root = root
        self.size = size
        base = root.man.malloc(size)
        super().__init__(root, base, size)

    def __del__(self):
        if hasattr(self, 'base') and hasattr(self, 'root'):
            self.root.man.free(self.base)


@ti.data_oriented
class Launcher:
    def __init__(self, maxargs=64):
        self.maxargs = maxargs
        self.bsize = ti.field(int, maxargs)
        self.bbase = ti.field(int, maxargs)

    def call(self, func, **kwargs):
        roots = []
        bbase = [0] * self.maxargs
        bsize = [0] * self.maxargs
        assert len(kwargs) < self.maxargs
        for i, (key, buf) in enumerate(kwargs.items()):
            bbase[i] = buf.base
            bsize[i] = buf.size
            roots.append((buf.root, key))
        self.bbase.from_numpy(np.array(bbase))
        self.bsize.from_numpy(np.array(bsize))
        self._call(func, tuple(roots))

    def __call__(self, func):
        def wrapped(this):
            kwargs = {}
            for key in this.lanes:
                val = getattr(this, key)
                kwargs[key] = val
            self.call(func, **kwargs)

        return wrapped

    @ti.kernel
    def _call(self, func: ti.template(), roots: ti.template()):
        func(namespace((key, MemoryView(root, ti.subscript(self.bbase, i), ti.subscript(self.bsize, i))) for i, (root, key) in enumerate(roots)))


lane = Launcher()
root = Memory(float)


@ti.data_oriented
class MyProc:
    def __init__(self):
        self.buf = Buffer(root, 5)
        self.lanes = ['buf']

    @lane
    @ti.func
    def func(self):
        for i in range(self.buf.size):
            print(self.buf[i])


proc = MyProc()
proc.func()
exit(1)
