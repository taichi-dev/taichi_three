# Hello triangle

The goal of this section is to give a brief introduction to Taichi THREE. We will start by setting up a scene, with a simple triangle, as many OpenGL tutorial starts from too.
A working example is provided at the bottom of the page in case you get stuck and need help.

## Creating the scene

To actually be able to display anything with Taichi THREE, we need three things: scene, camera, and model, so that we can render the scene with camera.

Scene - the collection of all renderer resources.
Camera - take images from scene and display it on the screen.
Model - the object to be displayed in the scene, a triangle in this case.

```py
scene = t3.Scene()
camera = t3.Camera()
scene.add_camera(camera)

model = t3.SimpleModel()
scene.add_model(model)
```

Finally, don't forget to **add** the model and camera to scene.

## Loading mesh data

In the world of modern graphics, a **model** is composed of many pieces of triangle faces, and their vertices.
Each triangle face has 3 vertices at its corner.

To draw a triangle, we need to specify the **position** of its vertices, in the format of (X, Y, Z) coordinates.

In this case, we want to draw a triangle `ABC` with its vertices at coordinates `A(0, 0.5, 0)`, `B(0.5, -0.5, 0)`, `C(-0.5, -0.5, 0)`.

We can load vertices by using our OpenGL-alike API:

```py
model.gl.Begin('GL_TRIANGLES')       # begin drawing triangle(s)
model.gl.Vertex(+0.0, +0.5, 0.0)     # A
model.gl.Vertex(+0.5, -0.5, 0.0)     # B
model.gl.Vertex(-0.5, -0.5, 0.0)     # C
model.gl.End()
```

## Visualizing the scene

If you copied the code from above and run, you wouldn't be able to see anything. This is because we're not actually rendering anything yet. For that, we need what's called an animate loop.

The good news is that we've already intergrated a simple GUI system that is able to render animation in real-time, to utilize it, just:

```py
gui = t3.GUI('Hello Triangle')
while gui.running:
    scene.render()
    gui.set_image(camera.img)  # blit the image captured by `camera` to screen
    gui.show()
```

Till now running the code successfully show gives you an white triangle in the middle of screen:

![0_1](0_1.gif)

## Controling the camera with mouse

We can also move the camera by mouse. To do so, we'll need to capture some mouse events in our GUI loop, and feed it into the `camera`:

```py
gui = t3.GUI('Hello Triangle')
while gui.running:
    gui.get_event(None)     # receive mouse and key events from GUI
    camera.from_mouse(gui)  # let the camera to process the mouse events
    scene.render()
    gui.set_image(camera.img)
    gui.show()
```

Now use **LMB to orbit** around the scene, **MMB to move** the center of view, **RMB to scale** the scene.
Feel like moving in 3D, right? Exactly what we want!


## Face culling

Turning the camera around using LMB, you may already noticed that the triangle is **invisible** when we moved to its back.

That's because Taichi THREE enforces the **face culling policy**: a face is only visible when the vertices are **clockwise**.
Why do we set this strange limitation? Well, you'll know in the later sections.

![face_culling](http://learnopengl.com/img/advanced/faceculling_windingorder.png)

For example, we used `[0, 1, 2]` as vertex indices, so the ordered vertices are:
1. ``x=+0.0, y=+0.5``
2. ``x=-0.5, y=-0.5``
3. ``x=+0.5, y=-0.5``

Looking from -Z direction, they are clockwise (therfore visible).
Looking from +Z direction, they are counter-clockwise (therfore invisible).

So if you flip the vertex order to ACB, you'll find the triangle only visible from the back (by moving camera around):

```py
model.gl.Begin('GL_TRIANGLES')       # begin drawing triangles
model.gl.Vertex(+0.0, +0.5, 0.0)     # A
model.gl.Vertex(-0.5, -0.5, 0.0)     # C
model.gl.Vertex(+0.5, -0.5, 0.0)     # B
model.gl.End()                       # update the draw buffer
```

## Specifying vertex colors

Mesh vertices could also has properties. The most commonly used property is, of course, color.

To specify a color for the triangle, simply call `model.gl.glColor` with RGB values before all vertices:

```py
model.gl.Begin('GL_TRIANGLES')       # begin drawing triangles
model.gl.Color(1.0, 0.0, 0.0)        # RGB (255, 0, 0), aka red
model.gl.Vertex(+0.0, +0.5, 0.0)     # A
model.gl.Vertex(+0.5, -0.5, 0.0)     # B
model.gl.Vertex(-0.5, -0.5, 0.0)     # C
model.gl.End()                       # update the draw buffer
```

Then you should get a red triangle now.

## Color interpolation

More interestingly, you can specify a different color for each vertices:

```py
model.gl.Begin('GL_TRIANGLES')       # begin drawing triangles
model.gl.Color(1.0, 0.0, 0.0)        # red
model.gl.Vertex(+0.0, +0.5, 0.0)     # A
model.gl.Color(0.0, 1.0, 0.0)        # green
model.gl.Vertex(+0.5, -0.5, 0.0)     # B
model.gl.Color(0.0, 0.0, 1.0)        # blue
model.gl.Vertex(-0.5, -0.5, 0.0)     # C
model.gl.End()                       # update the draw buffer
```

Then the color of each pixel in the triangle will then be a **interpolation** of its 3 vertices, via its barycentric coordinate:

![0_2](0_2.gif)

Congrats!

## Appendix

And here's the final code of this section:

[hello_triangle.py](_media/hello_triangle.py ':include :type=code')