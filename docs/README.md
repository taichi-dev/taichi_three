# Taichi THREE

Taichi THREE is a rendering pipeline based on the [Taichi Programming Language](https://github.com/taichi-dev/taichi) to render 3D scenes into nice-looking 2D images in real-time (work in progress).

[GitHub](https://github.com/taichi-dev/taichi_three) | [Examples](https://github.com/taichi-dev/taichi_three/tree/master/examples) | [Getting started](installation.md)

## Features

High-level features are:

- Both CPU / GPU available
- Multi-camera support
- Physically-based rendering
- Shadow mapping
- A built-in OBJ loader

Features may be available in the future:

* Deferred shading (screen-space shading).
* Image-based lighting and environment maps (skybox).
* Path tracing scheme, e.g. cornell box.
* Differentiable rendering (#18).

## Advantanges

Despite it's a soft renderer which means there'll be overheads compared with a true GPU rendering pipeline. It provides better flexibility.

TODO: more demostrations to be added.