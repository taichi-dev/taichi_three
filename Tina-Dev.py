bl_info = {
        'name': 'Tina (dev mode)',
        'description': 'A soft-renderer based on Taichi programming language',
        'author': 'archibate <1931127624@qq.com>',
        'version': (0, 0, 0),
        'blender': (2, 81, 0),
        'location': 'Render -> Tina',
        'support': 'TESTING',
        'wiki_url': 'https://github.com/archibate/tina/wiki',
        'tracker_url': 'https://github.com/archibate/tina/issues',
        'warning': 'Development mode',
        'category': 'Render',
}


import sys
sys.path.insert(0, '/home/bate/Develop/three_taichi')


registered = False


def register():
    print('Tina-Dev register...')
    import tina_blend
    tina_blend.register()

    global registered
    registered = True
    print('...register done')


def unregister():
    print('Tina-Dev unregister...')
    import tina_blend
    tina_blend.unregister()

    global registered
    registered = False
    print('...unregister done')


def reload_addon():
    import tina
    import tina_blend
    if registered:
        tina_blend.unregister()
    tina.__lazyreload__()
    tina_blend.__lazyreload__()
    tina_blend.register()


@eval('lambda x: x()')
def _():
    class Reload:
        def __repr__(self):
            import os
            import bpy
            os.system('clear')
            reload_addon()
            bpy.context.scene.frame_current = bpy.context.scene.frame_current
            return 'reloaded'

    __import__('bpy').a = Reload()
