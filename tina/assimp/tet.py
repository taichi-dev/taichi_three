class tetgen_reader:
    def __init__(self, filename, index_start=1):
        self.vertex = []
        self.face = []
        self.tet = []
        # read nodes from *.nodes file
        with open(filename+".node", "r") as file:
            vn = int(file.readline().split()[0])
            for i in range(vn):
                # [x, y, z]
                self.vertex += [float(x) for x in file.readline().split()[1:4]]

        # read faces from *.triangles file (only for rendering)
        with open(filename+".face", "r") as file:
            fn = int(file.readline().split()[0])
            for i in range(fn):
                # triangle
                self.face += [int(ind) -
                              index_start for ind in file.readline().split()[1:4]]

        # read elements from *.ele file
        with open(filename+".ele", "r") as file:
            en = int(file.readline().split()[0])
            for i in range(en):
                # tetrahedron
                self.tet += [int(ind) -
                             index_start for ind in file.readline().split()[1:5]]

        self.vertex_n = int(len(self.vertex)/3)
        self.face_n = int(len(self.face)/3)
        self.tet_n = int(len(self.tet)/4)


class inp_reader:
    def __init__(self, filename, scale=1.0):
        self.vertex = []
        self.vertex_old_id = []
        self.face = []
        self.face_old_id = []
        self.tet = []
        self.tet_old_id = []
        self.group = []
        self.group_name = []
        vertex_old_id_lut = {}
        face_old_id_lut = {}
        tet_old_id_lut = {}
        # read every line
        # determin which mode it is
        # 0. NA 1. node 2. faces 3. elements 4. groups
        # refresh_reading_group = True will create a new element set
        reading_mode = 0
        refresh_reading_set = False
        with open(filename) as file:
            while(True):
                # dont include the end of line, \n
                line = file.readline()[0:-1]
                # reach the end ?
                if not line:
                    break
                # print(line[0:10])
                # determine reading mode
                if line[0] == '*':
                    if line[1:5].upper() == 'NODE':
                        reading_mode = 1
                        # print('reading nodes')
                        continue

                    elif line[1:8].upper() == 'ELEMENT':
                        # print('reading elements')
                        type_id_pos_start = line.find('type=')+5
                        type_id_pos_end = line.find(',', type_id_pos_start)
                        type_id = line[type_id_pos_start:type_id_pos_end]
                        # print(type_id)
                        if type_id == 'CPS3':
                            reading_mode = 2
                            # print('reading triangle elements')
                        elif type_id == 'C3D4':
                            reading_mode = 3
                            # print('reading tetra elements')
                        else:
                            reading_mode = 0
                            # print('Unknown type')

                        continue

                    elif line[1:12].upper() == 'ELSET,ELSET':
                        reading_mode = 4
                        refresh_reading_set = True
                        # print('reading elements set')
                        set_name = line[13:]
                        self.group_name.append(set_name)
                        # print(set_name)
                        continue

                    else:
                        # * means this line is just a comment
                        # \n means a blank space
                        continue

                if reading_mode == 1:
                    old_idstr, *pos_list = line.split(',')
                    old_id = int(old_idstr)
                    pos = [float(x)/scale for x in pos_list]
                    # print(old_id)
                    # print(pos)
                    vertex_old_id_lut[old_id] = len(self.vertex_old_id)
                    self.vertex_old_id.append(old_id)
                    self.vertex.append(pos)

                elif reading_mode == 2:
                    old_id, *face_list = map(int, line.split(','))
                    face_old_id_lut[old_id] = len(self.face_old_id)
                    self.face_old_id.append(old_id)
                    faces_node_id = [vertex_old_id_lut[id] for id in face_list]
                    self.face.append(faces_node_id)
                    # print(old_id)
                    # print(faces_node_id)

                elif reading_mode == 3:
                    old_id, *tetra_list = map(int, line.split(','))
                    tetras_node_id = [vertex_old_id_lut[id] for id in tetra_list]
                    tet_old_id_lut[old_id] = len(self.tet_old_id)
                    self.tet_old_id.append(old_id)
                    self.tet.append(tetras_node_id)
                    # print(old_id)
                    # print(tetras_node_id)

                elif reading_mode == 4:
                    if refresh_reading_set:
                        # create a new element set then append
                        element_id_list = line.split(',')
                        temp_set = []
                        for id in element_id_list:
                            if id != ' ':
                                # assume only face element is included
                                temp_set.append(
                                    face_old_id_lut[int(id)])

                        self.group.append(temp_set)
                        refresh_reading_set = False
                    else:
                        # call the existing set end
                        element_id_list = line.split(',')
                        temp_set = []
                        for id in element_id_list:
                            if id != ' ':
                                self.group[-1].append(
                                    face_old_id_lut[int(id)])

        # print(len(self.vertex_old_id))
        # print(len(self.vertex))
        # print(len(self.face_old_id))
        # print(len(self.face))
        # print(len(self.tet_old_id))
        # print(len(self.tet))
        # print(self.tet_old_id)
        # print(self.group_name)
        # print(self.group)

        self.vertex_n = int(len(self.vertex))
        self.face_n = int(len(self.face))
        self.tet_n = int(len(self.tet))
        # print(self.vertex_n)
        # print(self.face_n)
        # print(self.tet_n)

    @property
    def group_vertex_id(self):
        if not hasattr(self, '_group_vertex_id'):
            self._group_vertex_id = []
            # generating unique vertex id in a group
            for i in range(len(self.group)):
                temp_f_list = self.group[i]
                temp_unique_v_list = []
                for f_id in temp_f_list:
                    fs_v_id = self.face[f_id]
                    for v_id in fs_v_id:
                        temp_unique_v_list.append(v_id)

                temp_unique_v_list = list(set(temp_unique_v_list))
                temp_unique_v_list.sort()
                self._group_vertex_id.append(temp_unique_v_list)

        return self._group_vertex_id

    # all the outside faces' vertex id
    @property
    def is_face_vertex(self):
        if not hasattr(self, '_is_face_vertex'):
            # setting if a vertex is a face vertex
            self._is_face_vertex = [0 for _ in range(len(self.vertex))]
            for f_id in range(len(self.face)):
                for v_id in self.face[f_id]:
                    self._is_face_vertex[v_id] = 1

        return self._is_face_vertex

    # those vertex connected faces
    @property
    def face_vertex_connectivity(self):
        if not hasattr(self, '_face_vertex_connectivity'):
            temp_f_v_connectivity = [[] for _ in range(len(self.vertex))]
            # push back connected face id to each vertex
            for f_id in range(len(self.face)):
                for v_id in self.face[f_id]:
                    temp_f_v_connectivity[v_id].append(f_id)

            for i in range(len(self.vertex)):
                temp_f_v_connectivity[i] = list(set(temp_f_v_connectivity[i]))

            # print(temp_f_v_connectivity)
            self._face_vertex_connectivity = temp_f_v_connectivity

        return self._face_vertex_connectivity

    @property
    def dict(self):
        import numpy as np
        return {
            'v': np.array(self.vertex),
            'f': np.array(self.face),
            't': np.array(self.tet),
        }


if __name__ == '__main__':
    import timeit
    print(timeit.timeit(lambda: inp_reader('geometry/hand.inp'), number=10))
    r = inp_reader('geometry/hand.inp')
    print(r.dict)
