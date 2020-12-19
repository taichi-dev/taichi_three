from ..common import *


@ti.data_oriented
class MeshEditBase:
    def __init__(self, mesh):
        self.mesh = mesh

    def __getattr__(self, attr):
        return getattr(self.mesh, attr)
