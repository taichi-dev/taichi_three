import numpy as np



def _tri_append(faces, indices):
    if len(indices) == 3:
        faces.append(indices)
    elif len(indices) == 4:
        faces.append([indices[0], indices[1], indices[2]])
        faces.append([indices[2], indices[3], indices[0]])
    elif len(indices) > 4:
        for n in range(1, len(indices) - 1):
            faces.append([indices[0], indices[n], indices[n + 1]])
    else:
        assert False, len(indices)


def readobj(path, scale=None):
    if path.endswith('.obj'):
        ret = read_OBJ(path)
    elif path.endswith('.npz'):
        ret = read_NPZ(path)
    else:
        assert False, f'Unrecognized file format: {path}'

    if scale is not None:
        ret['vp'] = ret['vp'] * scale
    return ret

def writeobj(path, obj):
    if path.endswith('.obj'):
        write_OBJ(path, obj)
    elif path.endswith('.npz'):
        write_NPZ(path, obj)
    else:
        assert False, f'Unrecognized file format: {path}'


def read_OBJ(path):
    vp = []
    vt = []
    vn = []
    faces = []

    with open(path, 'r') as myfile:
        lines = myfile.readlines()

    # cache vertices
    for line in lines:
        try:
            type, fields = line.split(maxsplit=1)
            fields = [float(_) for _ in fields.split()]
        except ValueError:
            continue

        if type == 'v':
            vp.append(fields)
        elif type == 'vt':
            vt.append(fields)
        elif type == 'vn':
            vn.append(fields)

    # cache faces
    for line in lines:
        try:
            type, fields = line.split(maxsplit=1)
            fields = fields.split()
        except ValueError:
            continue

        # line looks like 'f 5/1/1 1/2/1 4/3/1'
        # or 'f 314/380/494 382/400/494 388/550/494 506/551/494' for quads
        if type != 'f':
            continue

        # a field should look like '5/1/1'
        # for vertex/vertex UV coords/vertex Normal  (indexes number in the list)
        # the index in 'f 5/1/1 1/2/1 4/3/1' STARTS AT 1 !!!
        indices = [[int(_) - 1 if _ else 0 for _ in field.split('/')] for field in fields]

        _tri_append(faces, indices)

    ret = {}
    ret['vp'] = np.array([[0, 0, 0]], dtype=np.float32) if len(vp) == 0 else np.array(vp, dtype=np.float32)
    ret['vt'] = np.array([[0, 0]], dtype=np.float32) if len(vt) == 0 else np.array(vt, dtype=np.float32)
    ret['vn'] = np.array([[0, 0, 0]], dtype=np.float32) if len(vn) == 0 else np.array(vn, dtype=np.float32)
    ret['f'] = np.zeros((1, 3, 3), dtype=np.float32) if len(faces) == 0 else np.array(faces, dtype=np.int32)
    return ret


def write_OBJ(path, obj, name='Object'):
    with open(path, 'w') as f:
        f.write('# Taichi THREE saved OBJ file\n')
        f.write('# https://github.com/taichi-dev/taichi-three\n')
        f.write(f'o {name}\n')
        for pos in obj['vp']:
            f.write(f'v {" ".join(map(str, pos))}\n')
        for pos in obj['vt']:
            f.write(f'vt {" ".join(map(str, pos))}\n')
        for pos in obj['vn']:
            f.write(f'vn {" ".join(map(str, pos))}\n')
        f.write('s off\n')
        for face in obj['f']:
            f.write(f'f {" ".join("/".join(map(str, f + 1)) for f in face)}\n')


def write_NPZ(path, obj):
    data = {}
    data['vp'] = obj['vp']
    data['vt'] = obj['vt']
    data['vn'] = (obj['vn'] * (2**15 - 1)).astype(np.int16)
    data['f'] = obj['f'].astype(np.uint16)
    np.savez(path, **data)

def read_NPZ(path):
    data = np.load(path)

    ret = {}
    ret['vp'] = data['vp']
    ret['vt'] = data['vt']
    ret['vn'] = data['vn'].astype(np.float32) / (2**15 - 1)
    ret['f'] = data['f'].astype(np.int32)
    return ret
