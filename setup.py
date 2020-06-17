project_name = 'taichi_three'
from taichi_three.version import version, taichi_version, taiglsl_version
description = 'A Taichi extension library helps you rendering 3D scenes'
long_description = '''
Taichi THREE
============

Taichi THREE is an extension library of the `Taichi Programming Language <https://github.com/taichi-dev/taichi>`_, that helps rendering your 3D scenes into nice-looking 2D images to display in GUI.
'''
classifiers = [
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Multimedia :: Graphics',
    'Topic :: Games/Entertainment :: Simulation',
    'Operating System :: OS Independent',
]
python_requires = '>=3.6'
install_requires = [
    'taichi>=' + taichi_version,
    'taichi-glsl>=' + taiglsl_version,
]

import setuptools

setuptools.setup(
    name=project_name,
    version=version,
    author='彭于斌',
    author_email='1931127624@qq.com',
    description=description,
    long_description=long_description,
    classifiers=classifiers,
    python_requires=python_requires,
    install_requies=install_requires,
    packages=setuptools.find_packages(),
)
