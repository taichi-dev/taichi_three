from ..common import *
from .simple import *


@ti.data_oriented
class PrimitiveMesh(SimpleMesh):
    def __init__(self, faces):
        if not isinstance(faces, np.ndarray):
            faces = np.array(faces, dtype=np.float32)
        verts = faces[:, :, 0]
        norms = faces[:, :, 1]
        coors = faces[:, :, 2]
        super().__init__(maxfaces=len(verts), npolygon=len(verts[0]))

        @ti.materialize_callback
        def init_mesh():
            self.set_face_verts(verts)
            self.set_face_norms(norms)
            self.set_face_coors(coors)

    @classmethod
    def sphere(cls, lons=32, lats=24, rad=1):
        def at(lat, lon):
            lat /= lats
            lon /= lons
            coor = lon, lat, 0
            lat = (lat - 0.5) * np.pi
            lon = (lon * 2 - 1) * np.pi
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            norm = x, y, z
            vert = x * rad, y * rad, z * rad
            return vert, norm, coor

        faces = []
        for lat in range(lats):
            for lon in range(lons):
                v1 = at(lat, lon)
                v2 = at(lat, lon + 1)
                v3 = at(lat + 1, lon + 1)
                v4 = at(lat + 1, lon)
                if lat != 0:
                    faces.append([v1, v2, v3])
                if lat != lats - 1:
                    faces.append([v3, v4, v1])
        return cls(faces)

    @classmethod
    def cylinder(cls, lons=32, lats=4, rad=1, hei=2):
        def at(lat, lon, rad=rad):
            lat /= lats
            lon /= lons
            coor = lon, lat, 0
            lat = lat - 0.5
            lon = (lon * 2 - 1) * np.pi
            x = np.cos(lon)
            y = np.sin(lon)
            z = lat
            vert = x * rad, y * rad, z * hei
            norm = x, y, 0
            return vert, norm, coor

        faces = []
        for lat in range(lats):
            for lon in range(lons):
                v1 = at(lat, lon)
                v2 = at(lat, lon + 1)
                v3 = at(lat + 1, lon + 1)
                v4 = at(lat + 1, lon)
                faces.append([v1, v2, v3])
                faces.append([v3, v4, v1])
        for lon in range(lons):
            norm = 0, 0, -1
            coor = 0, 0, -1
            v1 = (0, 0, -hei / 2), norm, coor
            v2 = at(0, lon)[0], norm, coor
            v3 = at(0, lon + 1)[0], norm, coor
            faces.append([v3, v2, v1])
        for lon in range(lons):
            norm = 0, 0, 1
            coor = 0, 0, 1
            v1 = (0, 0, hei / 2), norm, coor
            v2 = at(lats, lon)[0], norm, coor
            v3 = at(lats, lon + 1)[0], norm, coor
            faces.append([v1, v2, v3])
        return cls(faces)

    @classmethod
    def asset(cls, name):
        obj = tina.readobj('assets/' + name + '.obj', quadok=True)
        verts = obj['v'][obj['f'][:, :, 0]]
        coors = obj['vt'][obj['f'][:, :, 1]]
        norms = obj['vn'][obj['f'][:, :, 2]]
        faces = []
        for vs, cs, ns in zip(verts, coors, norms):
            cs = list(cs)
            for i, c in enumerate(cs):
                cs[i] = list(c) + [0]
            faces.append(list(zip(vs, cs, ns)))
        return cls(faces)
