import taichi as ti
import functools
import time


def inject(module, name):
    def decorator(hook):
        func = getattr(module, name)

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            _taichi_skip_traceback = 1
            hook(*args, **kwargs)
            return func(*args, **kwargs)

        setattr(module, name, wrapped)
        return hook

    return decorator


@inject(ti.Kernel, 'materialize')
def materialize(self, key=None, args=None, arg_features=None):
    if key is None:
        key = (self.func, 0)
    if not self.runtime.materialized:
        self.runtime.materialize()
    if key in self.compiled_functions:
        return
    grad_suffix = ""
    if self.is_grad:
        grad_suffix = "_grad"
    kernel_name = "{}_c{}_{}{}".format(self.func.__name__,
                                        self.kernel_counter, key[1],
                                        grad_suffix)
    print(time.time(), 'Kernel JIT:', kernel_name)


@inject(ti.Func, '__call__')
def materialize(self, *args):
    if not ti.inside_kernel():
        return
    #print(time.time(), 'Func call:', self.func)