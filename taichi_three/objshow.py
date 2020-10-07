def objshow(obj):
    import taichi_three as t3

    scene = t3.Scene()
    model = (t3.ModelLow if lowp else t3.Model).from_obj(obj)
    if args.texture is not None:
        model.add_texture('color', ti.imread(args.texture))
    if args.normtex is not None:
        model.add_texture('normal', ti.imread(args.normtex))
    if args.metallic is not None:
        model.add_texture('metallic', ti.imread(args.metallic))
    if args.roughness is not None:
        model.add_texture('roughness', ti.imread(args.roughness))
    scene.add_model(model)
    camera = t3.Camera(res=(args.resx, args.resy))
    if args.ortho:
        camera.type = camera.ORTHO
    scene.add_camera(camera)
    light = t3.Light([0.4, -1.5, 0.8])
    scene.add_light(light)

    gui = ti.GUI('Model', camera.res)
    while gui.running:
        gui.get_event(None)
        gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
        camera.from_mouse(gui)
        scene.render()
        gui.set_image(camera.img)
        gui.show()