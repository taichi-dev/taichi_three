Taichi THREE
============

[![Downloads](https://pepy.tech/badge/taichi-three)](https://pepy.tech/project/taichi-three)
[![Latest Release](https://img.shields.io/github/v/release/taichi-dev/taichi_three)](https://github.com/taichi-dev/taichi_three/releases)

Taichi THREE is an extension library of the [Taichi Programming Language](https://github.com/taichi-dev/taichi) to render 3D scenes into nice-looking 2D images in real-time (work in progress).


![Example](https://github.com/taichi-dev/taichi_three/raw/master/assets/monkey.png)


Installation
------------

Install Taichi THREE with `pip`:

```bash
# Python 3.6/3.7/3.8 (64-bit)
pip install taichi_three
```

This should also install its dependencies `taichi` and `taichi_glsl` as well.


How to play
-----------

First, import Taichi and Taichi THREE:
```py
import taichi as ti
import taichi_three as t3

ti.init(ti.gpu)
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
texture = ti.imread('assets/cloth.jpg')
model = t3.Model(t3.readobj('assets/monkey.obj', scale=0.6), tex=texture)
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
