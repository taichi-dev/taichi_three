Changelog
=========

New in 0.0.8
------------

API breaking changes:
* `t3.Model.from_obj` is now deprecated, use `t3.Model(t3.Mesh.from_obj(obj))` instead.
* Fix the coordinate system to be right-handed: +X right, +Y up, +Z forward; +Z points FROM scene TO camera.

Notable changes:
* Set up a node-alike system for materials, see `examples/physics_based_rendering.py` for example.
* Add `t3.Geometry.cube()`, `t3.Geometry.cylinder()` for creating mesh without reading disk.
* Support Temporal Anti-Aliasing (TAA), use `t3.Camera(taa=True)` to enable it.
* Support `t3.readobj('cube.obj', orient='xyZ')` for orientation fixes.
* Add `t3.MeshMakeNormal`, `t3.MeshGrid`, and `t3.QuadToTri` mesh nodes.

Minor updates:
* Add `t3.objmerge`, `t3.objbreakdown` and `t3.objautoscale` for OBJ editing helpers.
* Fix broken `t3.ScatterModel`.

New in 0.0.7
------------

Notable changes:
* Preview our WIP documentation at https://t3.142857.red.
* Add `t3.SimpleModel` that simply do color interpolation between vertices.
* Refactor `t3.CookTorrance`, now use `model.shading_type = t3.BlinnPhong` if you want non-PBR shading.
* Add OBJ edit helpers, `t3.objflipaxis`, `t3.objmknorm`, `t3.objflipface`, `t3.objshow`.
* Framebuffered texcoor - get model surface coordinate by mouse, see `examples/screen_to_texcoor.py`.
* Separate camera control logic from `t3.Camera` to `t3.CameraCtl`.
* Support `t3.AmbientLight` and ambient occulsion.
* Set up the basis of path tracing scheme.
* Shadow mapping is broken.

Minor fixes:
* Fix an artifect in perspective mode due to texture coordinate interpolation.
* Deprecate `t3.AutoInit`, use `ti.materialize_callback` for better initializaion.
* Use fixed point number in depth buffer for better atomic performance.
* Use ModelView matrix for pre-calculated camera + model transform.
* Support non-equal width and height in camera.
* Make shadow smoother using bilerp.

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

API breaking changes:

Notable changes:
* Support SSAO by `t3.SSAO` node.
* Support deferred shading by using `t3.DeferredMaterial` and `t3.DeferredShading` nodes.
* Set up multi-material ID system, use `t3.objunpackmtls` to separate OBJ by their material.
* Add a variety of nodes, including `t3.GaussianBlur`, `t3.ImgBinaryOp`, `t3.SuperSampling`...
* Add `t3.Skybox` as model.

Minor fixes:
* Make camera buffer update less ad-hoc.
* Fix `t3.readobj` default orientation to be `-xyz`.

TODO list
---------

Minor fixes:
* Fix black-pixels artifect in `ms_cloth.py`.
* Fix color artifects on edges due to interpolation.
* Fix shadow artifects on 90-deg faces.
* Adapt ambient occulsion to node system.
* Fix broken shadow mapping since v0.0.7.
* Support scaling in ORTHO mode.

Major steps:
* Standardize affine system - L2W, W2C, C2D.
* Support shadow mapping for `t3.PointLight`.
* `t3.Light` should be a subclass of `t3.Model`?
* Support `t3.CutoffLight` for light cones.

Ambitions:
* Standardize path tracing scheme, e.g. cornell box.
* Further push forward node-alike system in all other fields.
* Support image-based lighting and environment maps (skybox).
* Support screen-space reflection.
* Differentiable rendering (#18).
