import numpy as np


def readobj(path):
    vertices = []
    faces = []
    with open(path, 'r') as myfile:
        data = myfile.readlines()
        # cache vertices

        for line in data:
            if line.startswith('v '):
                splitted = line.split()
                vertex = [float(splitted[1]),
                          float(splitted[2]),
                          float(splitted[3])
                          ]
                vertices.append(vertex)

        # cache faces
        # DONT merge this 'for loop'
        # must initialize vertices for faces to work
        print('length', len(vertices))
        for line in data:
            # line looks like 'f 5/1/1 1/2/1 4/3/1'
            if line.startswith('f '):
                splitted = line.split()
                faceVertices = []

                for i in range(1, 4):
                    # splitted[i] should look like '5/1/1'
                    # for vertex/vertex UV coords/vertex Normal  (indexes number in the list)
                    # the index in 'f 5/1/1 1/2/1 4/3/1' STARTS AT 1 !!!
                    index = int((splitted[i].split('/'))[0]) - 1
                    print(index)
                    faceVertices.append(vertices[index])
                faces.append(faceVertices)

        return np.array(faces)
