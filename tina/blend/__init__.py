import bpy
import bgl

from . import worker, options


class TinaRenderEngine(bpy.types.RenderEngine):
    # These custom members are used by blender to set up the
    # RenderEngine; define its internal name, visible name and capabilities.
    bl_idname = "TINA"
    bl_label = "Tina"
    bl_use_preview = True

    # Init is called whenever a new render engine instance is created. Multiple
    # instances may exist at the same time, for example for a viewport and final
    # render.
    def __init__(self):
        self.scene_data = None
        self.draw_data = None
        self.updated = False

    # When the render engine instance is destroy, this is called. Clean up any
    # render engine data here, for example stopping running render threads.
    def __del__(self):
        pass

    # This is the method called by Blender for both final renders (F12) and
    # small preview for materials, world and lights.
    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)

        # Fill the render result with a flat color. The framebuffer is
        # defined as a list of pixels, each pixel itself being a list of
        # R,G,B,A values.

        pixels = worker.render_main(self.size_x, self.size_y)
        pixels = pixels.reshape(self.size_x * self.size_y, 4).tolist()

        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, self.size_x, self.size_y)
        result.layers[0].passes["Combined"].rect = pixels
        self.end_result(result)

    # For viewport renders, this method gets called once at the start and
    # whenever the scene or 3D viewport changes. This method is where data
    # should be read from Blender in the same thread. Typically a render
    # thread will be started to do the work while keeping Blender responsive.
    def view_update(self, context, depsgraph):
        # print('view_update')
        region = context.region
        view3d = context.space_data
        region3d = context.region_data
        scene = depsgraph.scene
        perspective = region3d.perspective_matrix.to_4x4()

        viewport_changed = [self.draw_data is None or not hasattr(self, '_cac_old_perspective') or self.draw_data.perspective != self._cac_old_perspective, setattr(self, '_cac_old_perspective', perspective)][0]

        # Get viewport dimensions
        worker.invalidate_main(depsgraph.updates, viewport_changed)

        self.updated = True

    # For viewport renders, this method is called whenever Blender redraws
    # the 3D viewport. The renderer is expected to quickly draw the render
    # with OpenGL, and not perform other expensive work.
    # Blender will draw overlays for selection and editing on top of the
    # rendered image automatically.
    def view_draw(self, context, depsgraph):
        # print('view_draw')
        region = context.region
        region3d = context.region_data
        scene = depsgraph.scene
        perspective = region3d.perspective_matrix.to_4x4()

        # Get viewport dimensions
        dimensions = region.width, region.height

        # Bind shader that converts from scene linear to display space,
        bgl.glEnable(bgl.GL_BLEND)
        bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE_MINUS_SRC_ALPHA)
        self.bind_display_space_shader(scene)

        if not self.draw_data or self.updated \
            or self.draw_data.dimensions != dimensions \
            or self.draw_data.perspective != perspective:
            # print('CustomDrawData')
            self.draw_data = CustomDrawData(dimensions, perspective, region3d)
            self.updated = False

        self.draw_data.draw()

        self.unbind_display_space_shader()
        bgl.glDisable(bgl.GL_BLEND)


def compile_shader(type, source):
    shader = bgl.glCreateShader(type)
    bgl.glShaderSource(shader, source)
    bgl.glCompileShader(shader)
    status = bgl.Buffer(bgl.GL_INT, 1)
    bgl.glGetShaderiv(shader, bgl.GL_COMPILE_STATUS, status)
    if status[0] != bgl.GL_TRUE:
        infoLog = bgl.Buffer(bgl.GL_BYTE, 1024)
        length = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGetShaderInfoLog(shader, 1024, length, infoLog)
        raise RuntimeError(''.join(chr(infoLog[i]) for i in range(length[0])))
    return shader


def create_program(*shaders):
    program = bgl.glCreateProgram()
    for shader in shaders:
        bgl.glAttachShader(program, shader)
    bgl.glLinkProgram(program)
    status = bgl.Buffer(bgl.GL_INT, 1)
    bgl.glGetProgramiv(program, bgl.GL_LINK_STATUS, status)
    if status[0] != bgl.GL_TRUE:
        infoLog = bgl.Buffer(bgl.GL_BYTE, 1024)
        length = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGetProgramInfoLog(program, 1024, length, infoLog)
        raise RuntimeError(''.join(chr(infoLog[i]) for i in range(length[0])))
    return program


class CustomDrawData:
    program = create_program(
        compile_shader(bgl.GL_VERTEX_SHADER, '''
#version 330 core

in vec2 pos;
in vec2 texCoord;
out vec2 fragCoord;

void main()
{
    fragCoord = texCoord;
    gl_Position = vec4(texCoord * 2 - 1, 0.0, 1.0);
}
'''),
        compile_shader(bgl.GL_FRAGMENT_SHADER, '''
#version 330 core

in vec2 fragCoord;
out vec4 fragColor;

uniform sampler2D tex0;

void main()
{
    vec3 color = texture2D(tex0, fragCoord).rgb;
    color = clamp(color, 0, 1);
    fragColor = vec4(color, 1.0);
}
'''))

    def __init__(self, dimensions, perspective, region3d):
        self.dimensions = dimensions
        self.perspective = perspective

        width = bpy.context.scene.tina_resolution_x
        height = bpy.context.scene.tina_resolution_y
        pixels = worker.render_main(width, height, region3d)

        # Generate dummy image buffer
        pixels = bgl.Buffer(bgl.GL_INT, width * height, pixels)

        # Generate texture
        self.texture = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenTextures(1, self.texture)
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture[0])
        bgl.glTexImage2D(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGBA16F, width, height, 0, bgl.GL_RGBA, bgl.GL_UNSIGNED_INT_8_8_8_8_REV, pixels)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

        bgl.glUseProgram(self.program)

        # Generate vertex array
        self.vertex_array = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenVertexArrays(1, self.vertex_array)
        bgl.glBindVertexArray(self.vertex_array[0])

        texturecoord_location = bgl.glGetAttribLocation(self.program, "texCoord")
        position_location = bgl.glGetAttribLocation(self.program, "pos")
        tex0_location = bgl.glGetAttribLocation(self.program, "tex0")

        bgl.glUniform1i(tex0_location, 0)

        bgl.glEnableVertexAttribArray(texturecoord_location)
        bgl.glEnableVertexAttribArray(position_location)

        # Generate geometry buffers for drawing textured quad
        width, height = dimensions
        position = [0.0, 0.0, width, 0.0, width, height, 0.0, height]
        position = bgl.Buffer(bgl.GL_FLOAT, len(position), position)
        texcoord = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        texcoord = bgl.Buffer(bgl.GL_FLOAT, len(texcoord), texcoord)

        self.vertex_buffer = bgl.Buffer(bgl.GL_INT, 2)

        bgl.glGenBuffers(2, self.vertex_buffer)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, self.vertex_buffer[0])
        bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, position, bgl.GL_STATIC_DRAW)
        bgl.glVertexAttribPointer(position_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)

        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, self.vertex_buffer[1])
        bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, texcoord, bgl.GL_STATIC_DRAW)
        bgl.glVertexAttribPointer(texturecoord_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)

        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, 0)
        bgl.glBindVertexArray(0)

    def __del__(self):
        bgl.glDeleteBuffers(2, self.vertex_buffer)
        bgl.glDeleteVertexArrays(1, self.vertex_array)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)
        bgl.glDeleteTextures(1, self.texture)

    def draw(self):
        bgl.glUseProgram(self.program)
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture[0])
        bgl.glBindVertexArray(self.vertex_array[0])
        bgl.glDrawArrays(bgl.GL_TRIANGLE_FAN, 0, 4)
        bgl.glBindVertexArray(0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)


# RenderEngines also need to tell UI Panels that they are compatible with.
# We recommend to enable all panels marked as BLENDER_RENDER, and then
# exclude any panels that are replaced by custom panels registered by the
# render engine, or that are not supported.
def get_panels():
    exclude_panels = {
        'VIEWLAYER_PT_filter',
        'VIEWLAYER_PT_layer_passes',
    }

    panels = []
    for panel in bpy.types.Panel.__subclasses__():
        if hasattr(panel, 'COMPAT_ENGINES') and 'BLENDER_RENDER' in panel.COMPAT_ENGINES:
            if panel.__name__ not in exclude_panels:
                panels.append(panel)

    return panels


def register():
    bpy.utils.register_class(TinaRenderEngine)

    for panel in get_panels():
        panel.COMPAT_ENGINES.add('TINA')

    options.register()
    worker.register()


def unregister():
    bpy.utils.unregister_class(TinaRenderEngine)

    for panel in get_panels():
        if 'TINA' in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove('TINA')

    worker.unregister()
    options.unregister()
