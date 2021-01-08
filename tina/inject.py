import taichi as ti
import functools
import atexit
import time


_cbs = []


@atexit.register
def _exit_callback():
    for func in _cbs:
        func()


def _inject(module, name, enable=True):
    if not enable:
        return lambda x: x

    def decorator(hook):
        func = getattr(module, name)
        if hasattr(func, '_injected'):
            func = func._injected

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            _taichi_skip_traceback = 1
            clb = hook(*args, **kwargs)
            ret = func(*args, **kwargs)
            if clb is not None:
                clb(ret)
            return ret

        wrapped._injected = func
        setattr(module, name, wrapped)
        return hook

    return decorator


@_inject(ti.Kernel, 'materialize')
def _(self, key=None, args=None, arg_features=None):
    self._key = key

    if key is None:
        key = (self.func, 0)
    if not self.runtime.materialized:
        self.runtime.materialize()
    if key in self.compiled_functions:
        return
    grad_suffix = ""
    if self.is_grad:
        grad_suffix = "_grad"
    name = "{}_c{}_{}{}".format(self.func.__qualname__,
                                        self.kernel_counter, key[1],
                                        grad_suffix)
    self._kname[key] = name

    @_cbs.append
    def callback():
        if self._profile.get(name):
            x = sorted(self._profile[name])
            if len(x) % 2 == 0:
                dt = (x[len(x) // 2] + x[len(x) // 2 - 1]) / 2
            else:
                dt = x[len(x) // 2]
            print(f'[{max(x):8.05f} {dt:8.05f} {len(x):4d}] {name}')


@_inject(ti.Kernel, '__call__')
def _(self, *args, **kwargs):
    def callback(ret):
        t1 = time.time()
        dt = t1 - t0
        self._profile.setdefault(self._kname[self._key], []).append(dt)

    if not hasattr(self, '_profile'):
        self._profile = {}
    if not hasattr(self, '_kname'):
        self._kname = {}

    #ti.trace('calling [' + self.func.__qualname__ + ']')
    t0 = time.time()
    return callback


print('[Tina] Taichi JIT injected!')

__all__ = []


if __name__ == '__main__':
    @ti.data_oriented
    class ODOP:
        @ti.kernel
        def func(self):
            print(233)

    c = ODOP()
    c.func()
    exit(1)
