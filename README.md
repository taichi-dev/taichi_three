# Tina

A real-time soft renderer based on the [Taichi](https://github.com/taichi-dev/taichi) programming language.

Checkout [`docs/`](https://github.com/taichi-dev/taichi_three/tree/master/docs) to getting started.

Checkout [`examples/`](https://github.com/taichi-dev/taichi_three/tree/master/examples) for mixed examples.

NOTE: the renderer has been previously named Taichi THREE, see bottom.

## Installation

For end users:
```bash
python3 -m pip install taichi-tina
```

For developers:
```bash
git clone https://github.com/taichi-dev/taichi_three.git --depth=1
cd taichi_three
python3 -m pip install wheel
python3 setup.py bdist_wheel
python3 -m pip install dist/*.whl
```

## Features

Here's a list of important features currently Tina have:

* particles - `docs/particles.py`
* triangle meshes - `docs/triangle.py`
* connective meshes - `tests/connect.py`
* smooth shading - `docs/smooth.py`
* mesh grid - `examples/meshgrid_wave.py`
* mesh wireframe & anti-aliasing - `docs/wireframe.py`
* construct surface from particles - `examples/mciso_mpm3d.py`
* foreground & background color - `examples/pars_mpm3d.py`
* lighting & materials - `docs/lighting.py`
* loading GLTF scene - `docs/gltf.py`
* transforming models - `docs/transform.py`
* image-based lighting, IBL - `examples/material_ball.py`
* real-time rendering volume - `docs/volume.py`
* loading OBJ models - `docs/monkey.py`
* path tracing mode - `docs/pathtrace.py`
* real-time ray tracing (WIP) - `tests/rtx.py`
* detect element under cursor (WIP) - `tests/probe.py`
* Blender addon (WIP) - [`Taichi-Blend`](https://github.com/taichi-dev/taichi_blend)

If you didn't find your feature of interest, feel free to [open an issue](https://github.com/taichi-dev/taichi_three/issues/new/choose) for requesting it :)

## Legacy

Hello, dear Taichi THREE users:

The core is completely re-written after Taichi THREE v0.0.9 is released making the API more intuitive and much easier to maintain in hope for make it available to everyone. It now supports rendering not only triangle meshes but also particles and volumes (more to be added).
Don't worry, many of the good things from legacy Taichi THREE are left, like cook-torrance, mesh editing nodes, OBJ importer, marching cube... but some of the very core parts, including the triangle rasterizer, is completely thrown away and replaced by a more efficient algorithm that is friendly to GPUs when faces are a bit large. The new rasterizer also make compilation a bit faster and no more growing compilation time when there are a lot of models (reliefs #26). Also note that the camera system is completely re-written (sorry @Zony-Zhao and @victoriacity!) and now we no longer have dead-lock (万向节死锁) at +Y and -Y.
The re-written renderer is renamed to `Tina` to celebrate the huge changes in its API and important refactors and steps as an answer to issue #21, source of the name might be [`Seventina`](https://www.bilibili.com/video/BV1zt411U7ZE), one of my favorite song of harumakigohan :laughing: Another reason is that we don't need a very long `import seventina as t3` when import, we could directly use the original package name by `import tina` and `tina.Scene()`, what do you think?
Also thanks to the cleaned code structure, the renderer now also have a Blender integration work in progress, see [`Taichi-Blend`](https://github.com/taichi-dev/taichi_blend), and [video demo here](https://www.bilibili.com/video/BV17i4y157xx).
I won't rename the repo `taichi-dev/taichi_three` very soon to prevent dead links to this project.
Also note that the dependency of [`taichi_glsl`](https://github.com/taichi-dev/taichi_glsl) is removed after the transition to `tina`, we invent own utility functions like `bilerp` and hacks like `ti.static` to prevent sync with the `taichi_glsl` repo (prevent issues like https://github.com/taichi-dev/taichi_three/issues/32#issuecomment-747265261).
Thank for watching and making use of my project! Your attention is my biggest motivation. Please let me know if you have issues or fun with it. I'll keep pushing Tina and Taichi forward, I promise :heart:

The legacy version of Taichi THREE could still be found at the [`legacy`](https://github.com/taichi-dev/taichi_three/tree/master/legacy) branch.
And here is an video introducing the usage of Tina: https://www.bilibili.com/video/BV1ft4y1r7oW
Finally, the new renderer Tina could be installed using command: `pip install taichi-tina`.
What do you think about the new name and these huge refactors? Inputs are welcome!
