# Taichi THREE

[![Downloads](https://pepy.tech/badge/taichi-three)](https://pepy.tech/project/taichi-three)
[![Latest Release](https://img.shields.io/github/v/release/taichi-dev/taichi_three)](https://github.com/taichi-dev/taichi_three/releases)

Taichi THREE is a node-based soft renderer based on the [Taichi Programming Language](https://github.com/taichi-dev/taichi) to render 3D scenes into nice-looking 2D images in real-time (work in progress).

[GitHub](https://github.com/taichi-dev/taichi_three) | [Examples](https://github.com/taichi-dev/taichi_three/tree/master/examples) | [Getting started](https://t3.142857.red/#/hello_cube.md)

![Example](https://github.com/taichi-dev/public_files/raw/master/binding_textures.png)
![Example](https://github.com/taichi-dev/public_files/raw/master/transform_models.png)
![Example](https://github.com/taichi-dev/public_files/raw/master/taichi/mass_spring_3d.gif)

High-level features are:

- Both CPU / GPU available
- Multi-camera support
- Physically-based rendering
- Node-based materials
- A built-in OBJ loader
- Shadow mapping


# Installation

First of all, let's install Taichi THREE on your computer.
Make sure you're using Python **3.6/3.7/3.8**, and **64-bit**.
And then install Taichi THREE via `pip`:

```bash
python3 -m pip install taichi_three
```

To verify the installation, type this command into the Python shell:

```py
import taichi_three as t3
```

## Troubleshooting

See https://taichi.readthedocs.io/en/latest/install.html#troubleshooting for issues related to Taichi.

If you encounter problems using Taichi THREE, please let me know by [opening an issue at GitHub](https://github.com/taichi-dev/taichi_three/issues/new), many thanks!

## Install from latest master branch

If you'd like to keep catching up with latest Taichi THREE updates, please clone it from [our GitHub repository](https://github.com/taichi-dev/taichi_three). Then build and install it:

```bash
git clone https://github.com/taichi-dev/taichi_three.git
# try this mirror repository on Gitee if the above is too slow:
# git clone https://gitee.com/archibate/taichi_three.git

cd taichi_three
python3 -m pip install -r requirements.txt  # install `taichi` and `taichi-glsl`
python3 -m pip install wheel                # required for the next step
python3 setup.py bdist_wheel                # create a `.whl` file
pip install -U dist/taichi_three-0.0.6-py3-none-any.whl
```
