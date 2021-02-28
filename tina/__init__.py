__version__ = (0, 1, 1)
__author__ = 'archibate <1931127624@qq.com>'
__license__ = 'MIT'

print('[Tina] version', '.'.join(map(str, __version__)))


def require_version(*ver):
    def tos(ver):
        return '.'.join(map(str, ver))

    msg = f'This program requires Tina version {tos(ver)} to work.\n'
    msg += f'However your installed Tina version is {tos(__version__)}.\n'
    msg += f'Try run `pip install taichi-tina=={tos(ver)}` to upgrade/downgrade.'
    if __version__ > ver:
        print(msg)
    elif __version__ < ver:
        raise RuntimeError(msg)


from .lazimp import *
from .hacker import *
from .common import *
from .advans import *

if __import__('tina').lazyguard:
    from .shield import *
    from .util import *
    from .matr import *
    from .core import *
    from .path import *
    from .assimp import *
    from .mesh import *
    from .pars import *
    from .voxl import *
    from .scene import *
    from .skybox import *
    from .random import *
    from .probe import *
    from .postp import *
