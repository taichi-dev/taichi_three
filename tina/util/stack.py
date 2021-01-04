from ..common import *


@ti.data_oriented
class Stack:
    def __init__(self, N_mt=(512, 512), N_len=32, field=None):
        if ti.cfg.arch == ti.cpu and ti.cfg.cpu_max_num_threads == 1 or ti.cfg.arch == ti.cc:
            N_mt = 1, 1
        print('[Tina] Using', 'x'.join(map(str, N_mt)), 'threads')
        self.N_mt = N_mt
        self.N_len = N_len
        self.val = ti.field(int) if field is None else field
        self.blk1 = ti.root.dense(ti.ij, N_mt)
        self.blk2 = self.blk1.dense(ti.k, N_len)
        self.blk2.place(self.val)
        self.len = ti.field(int, N_mt)

    @staticmethod
    def instance():
        return Stack.g_stack

    @ti.func
    def ndrange(self, shape):
        for i, j in ti.ndrange(*self.N_mt):
            stack = self.Proxy(self, V(i, j))
            setattr(Stack, 'g_stack', stack)
            x = i
            while x < shape[0]:
                y = j
                while y < shape[1]:
                    I = V(x, y)
                    yield I
                    y += self.N_mt[1]
                    break
                x += self.N_mt[0]
                break
            delattr(Stack, 'g_stack')

    @ti.data_oriented
    class Proxy:
        def __init__(self, stack, mtid):
            self.stack = stack
            self.mtid = mtid

        def __getattr__(self, attr):
            return getattr(self.stack, attr)

        @ti.func
        def size(self):
            return self.len[self.mtid]

        @ti.func
        def clear(self):
            self.len[self.mtid] = 0

        @ti.func
        def push(self, val):
            l = self.len[self.mtid]
            self.val[self.mtid, l] = val
            self.len[self.mtid] = l + 1

        @ti.func
        def pop(self):
            l = self.len[self.mtid]
            val = self.val[self.mtid, l - 1]
            self.len[self.mtid] = l - 1
            return val
