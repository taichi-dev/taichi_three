# Binding textures

In the [previous section](loading_models.md), you've successfully loaded a `cube.obj` and display it.
But its surface is just plain white till now, can't we make it more colorful, look like a real thing?

## Adding texture

Now I want to make the cube look like an wooden box. What we need is **texture**.

Let's download this image and save it to `container2.png`:

![container2.png](https://learnopengl.com/img/textures/container2.png)

?> Or, feel free to use your own images :)

Then, add this line to load the image onto the model:

```py
model.add_texture('color', t3.imread('container2.png'))
```

The `'color'` tells Taichi THREE to **sample** colors from that image.

Running it you'll obtain a wooden container:

![3_1](3_1.gif)

## Specular rate

Pretty cool, right? But the metal border of the container doesn't look like metal.

To make metal look like metal, we want it to **shine**.
We may use the Blinn-Phong shader, and set its `specular` paramter to `1.0`.
The `specular` parameter is simply the specular rate of material.
The higher the `specular` is, the more the material shines.

```py
model.shading = t3.BlinnPhong       # use Blinn-Phong shader
model.add_uniform('specular', 1.0)  # set a uniform specular rate
```

![3_2](3_2.gif)

## Specular as texture

But wait, we don't want the wood shine too!

So, `add_uniform` can only specify an parameter uniformly over the whole model.
It can't deal with a model with multiple materials on its face.

To specify a different specular rate per-pixel, we need a **specular map**, it's also a kind of texture.
Each pixel in the specular map represents how specular rate is at that point.

Let's download this image and save to `container2_specular.png`:

![container2_specular.png](https://learnopengl.com/img/textures/container2_specular.png)

?> See? Only the metal border pixels are set so only these pixels will shine.

And replace the `add_uniform` with this line in our script:

```py
model.add_texture('specular', t3.imread('container2_specular.png'))
```

![3_3](3_3.gif)

Now metal are shining and woods look like woods.
Congrats on yielding a more realistic result!


## Appendix

And here's the final code of this section:

[binding_textures.py](_media/binding_textures.py ':include :type=code')