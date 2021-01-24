bl_info = {
        'name': 'Tina',
        'description': 'A soft renderer based on Taichi programming language',
        'author': 'archibate <1931127624@qq.com>',
        'version': (0, 0, 1),
        'blender': (2, 81, 0),
        'location': 'Render -> Tina',
        'support': 'COMMUNITY',
        'wiki_url': 'https://github.com/taichi-dev/taichi_three/wiki',
        'tracker_url': 'https://github.com/taichi-dev/taichi_three/issues',
        'category': 'Render',
}

import tina

if tina.lazyguard:
    from .render_engine import *
