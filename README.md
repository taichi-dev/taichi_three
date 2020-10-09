Taichi THREE
============

[![Downloads](https://pepy.tech/badge/taichi-three)](https://pepy.tech/project/taichi-three)
[![Latest Release](https://img.shields.io/github/v/release/taichi-dev/taichi_three)](https://github.com/taichi-dev/taichi_three/releases)

Taichi THREE is an extension library of the [Taichi Programming Language](https://github.com/taichi-dev/taichi) to render 3D scenes into nice-looking 2D images in real-time (work in progress).


![Example](https://github.com/taichi-dev/public_files/raw/master/binding_textures.png)
![Example](https://github.com/taichi-dev/public_files/raw/master/transform_models.png)
![Example](https://github.com/taichi-dev/public_files/raw/master/taichi/mass_spring_3d.gif)


Changelog
=========

New in 0.0.6
------------

Notable changes:
* Support physics-based rendering (PBR), roughness and metallic can be textures.
* Support `t3.ScatterModel` for particle model renderer.
* Support `t3.writeobj` for exporting OBJ files.
* Add CLI tools, use `python -m taichi_three` to invoke.
* Support shadow mapping.

Internal updates:
* Setup the fundemental infrastructure for docsify.
* Standardize multi-buffer infrastructure for camera.
* Standardize texture infrastructure, e.g.:

```py
model.add_texture('color', ti.imread('assets/cloth.jpg'))
model.add_texture('roughness', ti.imread('assets/rough.jpg'))
model.add_texture('metallic', np.array([[0.5]]))  # uniform metallic everywhere
```

New in 0.0.5
------------

* Support smooth shading by interpolating colors at vertices.
* Support `t3.ModelPP` for per-pixel light samping instead of color interpolation.
* Support specifying normal map as textures, by using `t3.ModelPP.from_obj(obj, texture, normtex)`.
* Support overriding `model.pixel_shader` and `model.vertex_shader` for customized shader.

**API breaking changes**:
* `t3.Model` now must take `faces`, `pos`, `tex`, and `nrm` as input, use an array with size 1 to dummy them.
* Use `t3.Model.from_obj` instead to initialize Model from `.obj` files.


New in 0.0.3
------------

**Full camera control:** use `scene.camera.from_mouse(gui)` after a `gui.get_event()` call (without arguments) to control the camera with the mouse.
* Drag with the left mouse button pressed to orbit the camera.
* Drag with the right mouse button or use the wheel to zoom in and out.
* Drag with the middle mouse buttom to pan the camera.

Other updates:
* Support binding textures to models.
* Backface culling for perspective cameras.
* Used the area method to compute barycentric coordinates.
* Clip the triangle bounding box to prevent drawing outside the camera.

New in latest master
--------------------

Notable changes:
* Preview our WIP documentation at https://t3.142857.red.
* Add `t3.SimpleModel` that simply do color interpolation between vertices.
* Refactor `t3.CookTorrance`, now use `model.shading_type = t3.BlinnPhong` if you want non-PBR shading.
* Add OBJ edit helpers, `t3.objflipaxis`, `t3.objmknorm`, `t3.objflipface`, `t3.objshow`.
* Framebuffered texcoor - get model surface coordinate by mouse, see `examples/screen_to_texcoor.py`.
* Separate camera control logic from `t3.Camera` to `t3.CameraCtl`.
* Support `t3.AmbientLight` and ambient occulsion.

Minor fixes:
* Fix an artifect in perspective mode due to texture coordinate interpolation.
* Use fixed point number in depth buffer for better atomic performance.
* Use ModelView matrix for pre-calculated camera + model transform.
* Support non-equal width and height in camera.
* Make shadow smoother using bilerp.


TODO list
---------

Minor fixes:
* Fix black-pixels artifect in `ms_cloth.py`.
* Fix the coordinate system to be right-handed.
* Fix color artifects on edges due to interpolation.
* Refactor field initialization before materialization.
* Fix shadow artifects on 90-deg faces.
* Make shadow camera less ad-hoc.

Major steps:
* Standardize affine system - L2W, W2C, C2D.
* Add some helpers fo creating primitive geometries.
* Support shadow mapping for `t3.PointLight`.
* Support `t3.CutoffLight` for light cones.
* Support anti-aliasing.

Ambitions:
* Use a node system for materials.
* Path tracing scheme, e.g. cornell box.
* Support image-based lighting and environment maps (skybox).
* Support deferred shading and SSAO.
* Support screen-space reflection.
* Differentiable rendering (#18).



Help
====

Installation
------------

1. Install Taichi THREE via `pip` for end-users:

```bash
# Python 3.6/3.7/3.8 (64-bit)
pip install taichi_three
```

2. Clone and install latest Taichi THREE from `dev` branch:

```bash
# Python 3.6/3.7/3.8 (64-bit)
pip install taichi taichi_glsl
python setup.py build install
```


How to play
-----------

First, import Taichi and Taichi THREE:
```py
import taichi as ti
import taichi_three as t3

ti.init(ti.cpu)
```

---

Then, create a scene using:
```py
scene = t3.Scene()
```

NOTE: scene creation should be put before any kernel invocation or host access,
i.e. before materialization, so that `Scene.__init__` could define its internal
tensors without an error.

TL;DR: Put this line as forward as possible! Ideally right below `ti.init()`.

---

After that, load the model(s), and feed them into `scene`:

```py
model = t3.Model(t3.readobj('assets/monkey.obj', scale=0.6))
scene.add_model(model)
```

If you want to add texture, read the texture image and feed it into `model`:

```py
model.load_texture(ti.imread('assets/cloth.jpg'))
```

NOTE: model creations should also be put as forward as possible too.

---

Then, create the camera(s), and put it into `scene`:

```py
camera = t3.Camera()
scene.add_camera(camera)
``` 

NOTE: camera creations should also be put as forward as possible.

---

Also don't forget to set the light:
```py
light = t3.Light()
scene.add_light(light)
```

---

Finally, create a GUI. And here goes the main loop:

```py
gui = ti.GUI('Monkey')
while gui.running:
    scene.render()            # render the model(s) into image
    gui.set_image(camera.img)  # display the result image
    gui.show()
```

---

Checkout the [`examples/`](https://github.com/taichi-dev/taichi_three/tree/master/examples) directory for more runnable examples.

Loading models
--------------

Taichi-three uses a **left-handed** coordinate system where the +Z axis points **from** the camera **towards** the object. Therefore when exporting meshes from a modeling software (e.g., Blender), the axis directions should be set as "+Z forward, +Y up" so that the model will be oriented corrected in the taichi-three camera.
