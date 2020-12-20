from ..common import *


@ti.data_oriented
class VoxlEditBase:
    def __init__(self, voxl):
        self.voxl = voxl

    def __getattr__(self, attr):
        return getattr(self.voxl, attr)
