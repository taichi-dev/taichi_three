# Hello triangle

The goal of this section is to give a brief introduction to Taichi THREE. We will start by setting up a scene, with a simple triangle, as many OpenGL tutorial starts from too.
A working example is provided at the bottom of the page in case you get stuck and need help.

## Before we start

First of all, let's install Taichi THREE. Make sure you're using Python 3.6/3.7/3.8, and 64-bit. And then install Taichi THREE via `pip`:

```bash
python3 -m pip install taichi_three
```

To verify the installation, type this command into the Python shell:

```py
import taichi_three as t3
```

## Creating the scene

To actually be able to display anything with Taichi THREE, we need three things: scene, camera, and model, so that we can render the scene with camera.

Scene - the collection of all renderer resources.
Camera - take images from scene and display it on the screen.
Model - the object to be displayed in the scene, a triangle in this case.

```py
scene = t3.Scene()

camera = t3.Camera()
scene.add_camera(camera)

model = t3.ModelEZ(faces_n=1, pos_n=3)  # compared to t3.Model, t3.ModelEZ is easier for beginners :)
scene.add_model(model)
```

## Loading mesh data

```py
model.pos[0] = [+0.0, +0.5, 0.0]  # top
model.pos[1] = [-0.5, -0.5, 0.0]  # right
model.pos[2] = [+0.5, -0.5, 0.0]  # left
model.faces[0] = [0, 1, 2]        # each triangle face contains 3 indices into its vertices
```

## Visualizing the scene

If you copied the code from above and run, you wouldn't be able to see anything. This is because we're not actually rendering anything yet. For that, we need what's called an animate loop.

The good news is that we've already intergrated a simple GUI system that is able to render animation in real-time, to utilize it, just:

```py
        gui = t3.GUI('Hello Triangle')
        while gui.running:
            scene.render()
            gui.set_image(camera.img)  # blit the image captured by `camera`
            gui.show()
```

Till now running the code successfully show gives you an white triangle in the middle of screen:

![0_1](0_1.gif)

## Specifying vertex colors

Mesh vertices could also has properties. The most commonly used property is, of course, color.

To specify a color for each vertex, simply assign some RGB values to the `model.clr` field:

```py
model.pos[0] = [+0.0, +0.5, 0.0]
model.pos[1] = [-0.5, -0.5, 0.0]
model.pos[2] = [+0.5, -0.5, 0.0]
model.clr[0] = [1.0, 0.0, 0.0]    # red
model.clr[1] = [0.0, 1.0, 0.0]    # green
model.clr[2] = [0.0, 0.0, 1.0]    # blue
model.faces[0] = [0, 1, 2]
```

Then the color of each pixel in the triangle will then be a **interpolation** of its 3 vertices, via its barycentric coordinate.

Run it and you should see a colorful triangle as shown below:

![0_2](0_2.gif)

## Controling the camera with mouse

We can also move the camera by mouse. To do so, we'll need to capture some mouse events in our GUI loop, and feed it into the `camera`:

```py
        gui = t3.GUI('Hello Triangle')
        while gui.running:
            gui.get_event(None)  # receive mouse and key events from GUI
            camera.from_mouse(gui)  # let the camera to process the mouse events
            scene.render()
            gui.set_image(camera.img)
            gui.show()
```

Now use LMB to orbit around the scene, MMB to move the center of view, RMB to scale the scene.


## Appendix

The final complete code of this example is provided here in case you need it:

```py
import taichi_three as t3

scene = t3.Scene()
model = t3.ModelEZ(faces_n=1, pos_n=3)
scene.add_model(model)

camera = t3.Camera()
scene.add_camera(camera)

model.pos[0] = [+0.0, +0.5, 0.0]
model.pos[1] = [+0.5, -0.5, 0.0]
model.pos[2] = [-0.5, -0.5, 0.0]
model.clr[0] = [1.0, 0.0, 0.0]
model.clr[1] = [0.0, 1.0, 0.0]
model.clr[2] = [0.0, 0.0, 1.0]
model.faces[0] = [0, 1, 2]

gui = t3.GUI('Hello Triangle')
while gui.running:
    scene.render()
    gui.get_event(None)
    camera.from_mouse(gui)
    gui.set_image(camera.img)
    gui.show()
```