from ..common import *


@ti.data_oriented
class ParsEditBase:
    def __init__(self, pars):
        self.pars = pars

    def __getattr__(self, attr):
        return getattr(self.pars, attr)
