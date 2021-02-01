# Tina

A real-time soft renderer based on the [Taichi](https://github.com/taichi-dev/taichi) programming language.

Checkout [`docs/`](https://github.com/taichi-dev/taichi_three/tree/master/docs) for API demos to getting started.

Checkout [`examples/`](https://github.com/taichi-dev/taichi_three/tree/master/examples) for application examples.

NOTE: the renderer has been previously named Taichi THREE, see bottom.

# Installation

End users may install Tina from PyPI:
```bash
python3 -m pip install taichi-tina
```

# Features

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
* screen blooming effect - `tests/blooming.py`
* temporal anti-aliasing, TAA - `docs/options.py`
* fast approximate anti-aliasing, FXAA - `tests/fxaa.py`
* image-based lighting, IBL - `examples/ibl_matball.py`
* screen space ambient occlusion, SSAO - `tests/ssao.py`
* screen space reflection, SSR - `tests/ssr.py`
* real-time rendering volume - `docs/volume.py`
* loading OBJ models - `docs/monkey.py`
* path tracing mode - `docs/pathtrace.py`
* bidir path tracing (WIP) - `tests/bdpt.py`
* detect element under cursor (WIP) - `tests/probe.py`
* Blender addon (WIP) - [`Taichi-Blend`](https://github.com/taichi-dev/taichi_blend)

If you didn't find your feature of interest, feel free to [open an issue](https://github.com/taichi-dev/taichi_three/issues/new/choose) for requesting it :)

# Developer installation

If you'd like to make use of Tina in your own project or contribute to Tina:

Thank for your support! You may like to clone it locally instead of the
end-user installation so that you could easily modify its source code to
best fit your own needs :)

Here's the suggested script for Linux users:

```bash
cd
pip uninstall -y taichi-tina  # uninstall end-user version
pip uninstall -y taichi-tina  # twice for sure :)
git clone https://github.com/taichi-dev/taichi_three.git --depth=10
# try the link below (a mirror site in China) if the above is too slow:
# git clone https://gitee.com/archibate/taichi_three.git --depth=10
cd taichi_three
echo export PYTHONPATH=`pwd` >> ~/.bashrc  # add path for Python to search
source ~/.bashrc  # reload the configuration to take effect
```

Or, feel free to make use of `virtualenv` if you're familiar with it :)

## Verifying developer installation

After that, you may try this command to verify installation:

```bash
python -c "import tina; print(tina)"
```

It should shows up the path to the repository, e.g.:
```
<module 'tina' from '/home/bate/Develop/taichi_three/tina/__init__.py'>
```
Congratulations! Now you may `import tina` in your project to have fun.

Message containing `site-packages` may mean something wrong with PYTHONPATH:
```
<module 'tina' from '/lib/python3.8/site-packages/tina/__init__.py'>
```

## How to contribute

If you've done with some new features, or bug fixes with developer mode:

I would appericate very much if you'd like to contribute them by
opening an [pull request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/about-pull-requests)
so that people (including me) could share your works :) To do so:

```bash
vim tina/somefile.py # suppose you've now modified some file ready to share..
git checkout -b fix-bug  # switch to a new branch with name 'fix-bug'
git add .  # add and commit the file change
git commit -m "Hello, I fixed a bug in tina/somefile.py"
git push -u origin fix-bug
```

Then visit https://github.com/taichi-dev/taichi/pull/new/fix-bug and click
`Create Pull Request`, so that it will open a new pull request for you.
Then I'll reply to your pull request soon, thank for the contribution!

## Folder structures

```bash
$ ls
tina/        # the main source code of Tina
tina_blend/  # the Blender addon module (WIP)
assets/      # binary assets (models, images) for the demos
docs/        # simple demos aimed to demonstrate the usage of basic APIs
examples/    # advanced demos to show what applications Tina is capable to do
tests/       # some work in progress (WIP) features that remain testing
benchmarks/  # some benchmark scripts to test the performance of Tina
setup.py     # the script to build the PyPI package for Tina

$ ls tina/
assimp/      # asset (model) loaders, e.g. GLTF, OBJ, INP..
core/        # codebase related to the real-time rasterizer
path/        # codebase related to the offline path tracer
mesh/        # mesh storing and editing nodes
voxl/        # volume / voxel storing nodes
pars/        # particle set storing nodes
scene/       # define the tina.Scene class
matr/        # material and shading nodes
util/        # some useful utilities for Tina and your project
blend/       # Blender addon module (WIP)
cli/         # command line interface (WIP)
postp/       # post-processing, mainly about tone mapping
skybox.py    # implementing cube map and spherical map for skybox
random.py    # random number generators, e.g., Taichi built-in, Wang's hash
common.py    # some common functions that might be used by Tina or your project
advans.py    # some advanced functions that might be used by Tina or your project
hacker.py    # some dirty hacks into Taichi to make Tina easier to maintain
lazimp.py    # lazy importing infrastructure to reduce import time
inject.py    # profiling Taichi JIT compilation time (WIP)
shield.py    # make Taichi fields to support pickle (WIP)
probe.py     # inspect geometry from screen space (WIP)
```

# Legacy

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
