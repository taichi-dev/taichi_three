# Loading Models

The goal of this section is to give a true introduction to Taichi THREE models.


## From `t3.SimpleModel` to `t3.Model`

As you may ask in the [previous section](hello_triangle.md), the `t3.SimpleModel` - might be great for beginners and 2D games - seems only capable of interpolating colors between vertices. We can hardly obtain a realistic shading from it.

So here comes the `t3.Model`, it's more powerful than `t3.SimpleModel`. It can store the `pos`, `texcoor`, `normal` information of vertices separately.
Therefore it can utilize the data fields like `vt` and `vn` from the industrial standard - [wavefront OBJ format](https://docs.blender.org/manual/zh-hans/latest/addons/import_export/scene_obj.html), to obtain a much realistic image.

## Loading a cube geometry into `t3.Model`

You can easily load OBJ files into `t3.Model` by using the handy helper function `t3.Model.from_obj`:
```py
obj = t3.readobj('assets/cube.obj')
model = t3.Model.from_obj(obj)
scene.add_model(model)
```

?> Of course, you can still use `model.gl` to hand-write a cube, it's just more convinent to load from an OBJ file.

## Adding lights

Running the code above will gives you a very completely dark image. You should also get an eye-catching warning message `Warning: no lights` in this situation.

So, to make `t3.Model` visible, we should add a **light** source:

```py
light = t3.Light(dir=[-0.2, -0.6, 1.0])  # parallel light with a specific direction
scene.add_light(light)
```

![2_1](2_1.gif)

## Advanced lighting

You may also specify a color for it, yellow, for example:

```py
light = t3.Light(dir=[0.3, -1.0, 0.8], color=[1.0, 1.0, 0.0])
```

Or a point source light:

```py
light = t3.PointLight(position=[0.6, -0.8, -1.2], color=[1.0, 1.0, 0.0])
```

Or a ambient light, which doesn't care the face orientation:

```py
light = t3.AmbientLight(color=[1.0, 1.0, 0.0])
```

?> Also note that you may have multiple lights in a single scene. Have a try :)


## Appendix

And here's the final code of this section:

[shading_models.py](_media/shading_models.py ':include :type=code')