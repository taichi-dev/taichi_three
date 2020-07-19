import numpy as np


def readobj(path, scale=None):
    vertices = []
    textures = []
    faces = []
    faces_texture = []
    with open(path, 'r') as myfile:
        data = myfile.readlines()

        # cache vertices
        for line in data:
            try:
                type, coors = line.split(maxsplit=1)
                coors = [float(_) for _ in coors.split()]
            except ValueError:
                continue

            if type == 'v':
                vertices.append(coors)
            elif type == 'vt':
                textures.append(coors)

        # cache faces
        # DONT merge this 'for loop'
        # must initialize vertices for faces to work
        for line in data:
            try:
                type, idxs = line.split(maxsplit=1)
                idxs = idxs.split()
            except ValueError:
                continue

            # line looks like 'f 5/1/1 1/2/1 4/3/1'
            # or 'f 314/380/494 382/400/494 388/550/494 506/551/494' for quads
            if type == 'f':
                faceVertices = []
                faceVerticesTexture = []

                for i in range(len(idxs)):
                    # splitted[i] should look like '5/1/1'
                    # for vertex/vertex UV coords/vertex Normal  (indexes number in the list)
                    # the index in 'f 5/1/1 1/2/1 4/3/1' STARTS AT 1 !!!
                    index = int((idxs[i].split('/'))[0]) - 1
                    faceVertices.append(index)
                    if idxs[i].split('/').__len__() > 1:
                        texture_index = int((idxs[i].split('/'))[1]) - 1
                        faceVerticesTexture.append(texture_index)

                if len(faceVertices) == 4:
                    faces.append([faceVertices[0], faceVertices[1], faceVertices[2]])
                    faces.append([faceVertices[2], faceVertices[3], faceVertices[0]])
                else:
                    faces.append(faceVertices)
                
                if faceVerticesTexture.__len__() != 0:
                    if len(faceVertices) == 4:
                        faces_texture.append([faceVerticesTexture[0], faceVerticesTexture[1], faceVerticesTexture[2]])
                        faces_texture.append([faceVerticesTexture[2], faceVerticesTexture[3], faceVerticesTexture[0]])
                    else:
                        faces_texture.append(faceVerticesTexture)

    ret = {}

    faces = np.array(faces)
    vertices = np.array(vertices)
    ret['v'] = vertices.astype(np.float32) * scale
    ret['f'] = faces.astype(np.int32)
    
    if faces_texture.__len__() != 0:
        faces_texture = np.array(faces_texture)
        ret['ft'] = faces_texture.astype(np.int32)

    textures = np.array(textures)
    ret['vt'] = textures.astype(np.float32)

    return ret
