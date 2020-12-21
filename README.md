# Tina

A real-time soft renderer based on the [Taichi](https://github.com/taichi-dev/taichi) programming language.

See [`docs/`](https://github.com/taichi-dev/taichi_three/tree/master/docs) to getting started.
See [`examples/`](https://github.com/taichi-dev/taichi_three/tree/master/examples) for usage examples.

NOTE: the renderer has been previously named Taichi THREE, see below.

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
