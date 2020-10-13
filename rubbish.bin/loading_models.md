# Loading models

The goal of this section is to teach you how to load and display an [wavefront OBJ format](https://docs.blender.org/manual/zh-hans/latest/addons/import_export/scene_obj.html) model using Taichi THREE.


## Loading an OBJ file into `t3.Model`

You can easily load OBJ models into `t3.Model`:
```py
obj = t3.readobj('cube.obj')
model = t3.Model.from_obj(obj)
scene.add_model(model)
```