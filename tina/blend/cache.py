class IDCache:
    def __init__(self, ident):
        self.lut = {}
        self.ident = ident

    def lookup(self, func, object):
        idx = self.ident(object)
        if idx in self.lut:
            return self.lut[idx]
        value = func(object)
        self.lut[idx] = value
        return value

    def clear(self):
        self.lut.clear()

    def invalidate(self, object):
        idx = self.ident(object)
        if idx in self.lut:
            del self.lut[idx]


def view_plane(camd, winx, winy, xasp, yasp):    
    #/* fields rendering */
    ycor = yasp / xasp
    use_fields = False
    if (use_fields):
      ycor *= 2

    def BKE_camera_sensor_size(p_sensor_fit, sensor_x, sensor_y):
        #/* sensor size used to fit to. for auto, sensor_x is both x and y. */
        if (p_sensor_fit == 'VERTICAL'):
            return sensor_y;

        return sensor_x;

    if (camd.type == 'ORTHO'):
      #/* orthographic camera */
      #/* scale == 1.0 means exact 1 to 1 mapping */
      pixsize = camd.ortho_scale
    else:
      #/* perspective camera */
      sensor_size = BKE_camera_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
      pixsize = (sensor_size * camd.clip_start) / camd.lens

    #/* determine sensor fit */
    def BKE_camera_sensor_fit(p_sensor_fit, sizex, sizey):
        if (p_sensor_fit == 'AUTO'):
            if (sizex >= sizey):
                return 'HORIZONTAL'
            else:
                return 'VERTICAL'

        return p_sensor_fit

    sensor_fit = BKE_camera_sensor_fit(camd.sensor_fit, xasp * winx, yasp * winy)

    if (sensor_fit == 'HORIZONTAL'):
      viewfac = winx
    else:
      viewfac = ycor * winy

    pixsize /= viewfac

    #/* extra zoom factor */
    pixsize *= 1 #params->zoom

    #/* compute view plane:
    # * fully centered, zbuffer fills in jittered between -.5 and +.5 */
    xmin = -0.5 * winx
    ymin = -0.5 * ycor * winy
    xmax =  0.5 * winx
    ymax =  0.5 * ycor * winy

    #/* lens shift and offset */
    dx = camd.shift_x * viewfac # + winx * params->offsetx
    dy = camd.shift_y * viewfac # + winy * params->offsety

    xmin += dx
    ymin += dy
    xmax += dx
    ymax += dy

    #/* fields offset */
    #if (params->field_second):
    #    if (params->field_odd):
    #        ymin -= 0.5 * ycor
    #        ymax -= 0.5 * ycor
    #    else:
    #        ymin += 0.5 * ycor
    #        ymax += 0.5 * ycor

    #/* the window matrix is used for clipping, and not changed during OSA steps */
    #/* using an offset of +0.5 here would give clip errors on edges */
    xmin *= pixsize
    xmax *= pixsize
    ymin *= pixsize
    ymax *= pixsize

    return xmin, xmax, ymin, ymax


def projection_matrix(camd):
    r = bpy.context.scene.render
    left, right, bottom, top = view_plane(camd, r.resolution_x, r.resolution_y, 1, 1)

    farClip, nearClip = camd.clip_end, camd.clip_start

    Xdelta = right - left
    Ydelta = top - bottom
    Zdelta = farClip - nearClip

    mat = [[0]*4 for i in range(4)]

    mat[0][0] = nearClip * 2 / Xdelta
    mat[1][1] = nearClip * 2 / Ydelta
    mat[2][0] = (right + left) / Xdelta #/* note: negate Z  */
    mat[2][1] = (top + bottom) / Ydelta
    mat[2][2] = -(farClip + nearClip) / Zdelta
    mat[2][3] = -1
    mat[3][2] = (-2 * nearClip * farClip) / Zdelta

    return sum([c for c in mat], [])
