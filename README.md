# Tina

A real-time soft renderer based on the [Taichi](https://github.com/taichi-dev/taichi) programming language

See [`docs/`](https://github.com/taichi-dev/taichi_three/tree/master/docs) to getting started.
See [`examples/`](https://github.com/taichi-dev/taichi_three/tree/master/examples) for usage examples.

NOTE: the renderer has been previously named Taichi THREE, see below.

## Installation

```bash
git clone https://github.com/taichi-dev/taichi_three.git --depth=1
cd taichi_three
python3 -m pip install wheel
python3 setup.py bdist_wheel
python3 -m pip install dist/*.whl
```

## Legacy

The core is completely re-written after Taichi THREE v0.0.9 is released making the API more intuitive and much easier to maintain in hope for make it available to everyone.
Don't worry, many of the good things from legacy Taichi THREE are left, like cook-torrance, mesh editing nodes, OBJ importer, marching cube... but some of the very core parts, including the triangle rasterizer, is completely thrown away and replaced by a more efficient algorithm that is friendly to GPUs when faces are a bit large. The new rasterizer also make compilation a bit faster and no more growing compilation time when there are a lot of models (reliefs #26). Also note that the camera system is completely re-written (sorry @Zony-Zhao and @victoriacity!) and now we no longer have dead-lock (万向节死锁) at +Y and -Y.
The re-written renderer is renamed to `Tina` to celebrate the huge changes in its API and important refactors and steps as an answer to issue #21, source of the name might be [`Seventina`](https://www.bilibili.com/video/BV1zt411U7ZE), one of my favorite song of harumakigohan :laughing: Another reason is that we don't need a very long `import seventina as t3` when import, we could directly use the original package name by `import tina` and `tina.Scene()`, what do you think?
Also thanks to the cleaned code structure, the renderer now also have a Blender integration work in progress, see [`Taichi-Blend`](https://github.com/taichi-dev/taichi_blend), and [video demo here](https://www.bilibili.com/video/BV17i4y157xx).
I won't rename the repo `taichi-dev/taichi_three` very soon to prevent dead links to this project.
Also note that the dependency of [`taichi_glsl`](https://github.com/taichi-dev/taichi_glsl) is removed after the transition to `tina`, we invent own utility functions like `bilerp` and hacks like `ti.static` to prevent sync with the `taichi_glsl` repo (prevent issues like https://github.com/taichi-dev/taichi_three/issues/32#issuecomment-747265261).
The legacy version of Taichi THREE could still be found at [tag `v0.0.9`](https://github.com/taichi-dev/taichi_three/tree/master/v0.0.9).
