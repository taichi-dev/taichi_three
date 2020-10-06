# Advanced Models

The goal of this section is to give a true introduction to Taichi THREE models.


## From `t3.SimpleModel` to `t3.Model`

As you may ask in the [previous section](loading_models.md), the `t3.SimpleModel` - might be great for beginners and 2D games - seems only capable of interpolating colors between vertices. We can hardly obtain a realistic shading from it.

So here comes the `t3.Model`, it's more powerful than `t3.SimpleModel`. It can store the `pos`, `texcoor`, `normal` information of vertices separately.
Therefore it can utilize the data fields like `vt` and `vn` from OBJ files, to obtain a much realistic image.

## Loading models into `t3.Model`

To load OBJ files into `t3.Model` properly:

```py
obj = t3.readobj('assets/cube.obj')
model = t3.Model(faces_n=len(obj['f']), pos_n=len(obj['vp']), tex_n=len(obj['vt']), nrm_n=len(obj['vn']))
scene.add_model(model)

model.pos.from_numpy(obj['vp'])
model.tex.from_numpy(obj['vt'])
model.nrm.from_numpy(obj['vn'])
model.faces.from_numpy(obj['f'])
```

Or equivalently, we provide a handy helper function `t3.Model.from_obj` to simplify these stuffs:
```py
obj = t3.readobj('assets/cube.obj')
model = t3.Model.from_obj(obj)
scene.add_model(model)
```

You may also initialize or modify `model.pos`, `model.tex`, `model.nrm` manually when required, the display will be updated in real-time.
This allows you to visualize mesh deformations, see [ms_cloth.py](https://github.com/taichi-dev/taichi_three/blob/master/examples/ms_cloth.py) for example.


## Adding lights

Running the code above will gives you a very dark image, you can hardly find the border of cube thanks to post processing.
You should also get an eye-catching warning message `Warning: no lights` in this situation.

So, to make `t3.Model` visible and good-looking, we should add a light source:

```py
light = t3.Light(dir=[0.3, -1.0, 0.8])   # parallel light with a specific direction
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

## Appendix

Also note that you may have multiple lights in a single scene.

[Click here for downloading the final complete code of this section](/advanced_models.py)