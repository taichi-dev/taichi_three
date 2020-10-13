# Hello cube

The goal of this section is to give a brief introduction to Taichi THREE. We will start by setting up a scene, with a simple triangle, as many OpenGL tutorial starts from too.
A working example is provided at the bottom of the page in case you get stuck and need help.

## Creating the scene

To actually be able to display anything with Taichi THREE, we need three things: scene, camera, and model, so that we can render the scene with camera.

Scene - the collection of all renderer resources.
Camera - take images from scene and display it on the screen.
Model - the object to be displayed in the scene, a triangle in this case.

Let's add the scene and camera now:

```py
scene = t3.Scene()
camera = t3.Camera()
scene.add_camera(camera)
```

## Add a cube as model

To make the model display the geometry we desired (cube in this case), we need to load some geometry data into it. To do so:


```py
obj = t3.Geometry.cube()        # create a cube geometry object
model = t3.Model.from_obj(obj)  # create a model from a geometry object
scene.add_model(model)
```

?> Finally, don't forget to **add** the model and camera to scene.

## Adding lights

Running the code above will gives you a very completely dark image. You should also get an warning message `Warning: no lights` in this situation.

So, to make our `t3.Model` visible, we should add a **light** source:

```py
light = t3.Light(dir=[-0.2, -0.6, -1.0])  # parallel light with a specific direction
scene.add_light(light)
```

![2_1](2_1.gif)

Nice job!

## Appendix

And here's the final code of this section:

[hello_cube.py](_media/hello_cube.py ':include :type=code')