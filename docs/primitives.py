import tina

scene = tina.Scene(smoothing=True, texturing=True, taa=True)

#mesh = tina.PrimitiveMesh.sphere()
mesh = tina.PrimitiveMesh.cylinder()
wire = tina.MeshToWire(mesh)
scene.add_object(mesh)
scene.add_object(wire)

gui = tina.ti.GUI('primitives')

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
