bl_info = {
        'name': 'Tina',
        'description': 'A real-time soft renderer based on Taichi programming language',
        'author': '彭于斌 <1931127624@qq.com>',
        'version': (0, 0, 1),
        'blender': (2, 81, 0),
        'location': 'Render -> Tina',
        'support': 'COMMUNITY',
        'wiki_url': 'https://github.com/taichi-dev/taichi_three/wiki',
        'tracker_url': 'https://github.com/taichi-dev/taichi_three/issues',
        'category': 'Render',
}

__version__ = bl_info['version']
__author__ = bl_info['author']
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


def register():
    print('[Tina] registering as blender addon')
    from . import blend
    blend.register()


def unregister():
    print('[Tina] unregistering as blender addon')
    from . import blend
    blend.unregister()


from .hacker import *
from .common import *
from .advans import *
from .core import *
from .util import *
from .assimp import *
from .mesh import *
from .pars import *
from .voxl import *
from .scene import *
