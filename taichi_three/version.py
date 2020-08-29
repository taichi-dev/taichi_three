version = (0, 0, 3)
taiglsl_version = (0, 0, 5)
taichi_version = (0, 6, 24)

print(f'[Tai3D] version {".".join(map(str, version))}')

try:(lambda o:(lambda p:o.exists(p)or open(p,'w').close()and 0)(o.join(o.dirname(o.abspath(__import__('taichi_three').__file__)),'.dismiss_screen')))(__import__('os').path)or print('''\
===========================================================
Thank you for choosing Taichi THREE!
The package is work in progress and your feedback can be
more than important to us :-)
So if you encounter any problem, or you've some cool ideas,
please let us know by opening an issue on GitHub:
https://github.com/taichi-dev/taichi_three/issues
===========================================================
''',end='')
except:pass
