from ..common import *
from .base import VoxlEditBase


class VolumeScale(VoxlEditBase):
    def __init__(self, voxl, scale=1):
        super().__init__(voxl)

        self.scale = ti.field(float, ())

        @ti.materialize_callback
        def init_scale():
            self.scale[None] = scale

    def set_scale(self, scale):
        self.scale[None] = scale

    @ti.func
    def sample_volume(self, pos):
        return self.voxl.sample_volume(pos) * self.scale[None]
