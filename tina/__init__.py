bl_info = {
        'name': 'Tina',
        'description': 'A real-time soft renderer based on Taichi programming language',
        'author': '彭于斌 <1931127624@qq.com>',
        'version': (0, 0, 0),
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
from .mesh import *
from .pars import *
from .voxl import *
from .scene import *
