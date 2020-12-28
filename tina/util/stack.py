from ..common import *


@ti.data_oriented
class Stack:  # consumes 64 MiB by default:
    def __init__(self, N_mt=512**2, N_len=64, field=None):
        self.val = ti.field(int) if field is None else field
        self.blk1 = ti.root.dense(ti.i, N_mt)
        self.blk2 = self.blk1.dense(ti.j, N_len)
        self.blk2.place(self.val)

        self.len = ti.field(int, N_mt)

    def get(self, mtid):
        return self.Proxy(self, mtid)

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
