#from tina.common import *


#verts, faces = tina.readobj('assets/bunny.obj', simple=True)
#verts = verts[faces]
import numpy as np
pos = np.load('assets/fluid.npy')[:3]


def expandBits(v):
    v = (v * 0x00010001) & 0xFF0000FF;
    v = (v * 0x00000101) & 0x0F00F00F;
    v = (v * 0x00000011) & 0xC30C30C3;
    v = (v * 0x00000005) & 0x49249249;
    return v;

def morton3D(x, y, z):
    x = min(max(x * 1024, 0), 1023);
    y = min(max(y * 1024, 0), 1023);
    z = min(max(z * 1024, 0), 1023);
    xx = expandBits(int(x));
    yy = expandBits(int(y));
    zz = expandBits(int(z));
    return xx * 4 + yy * 2 + zz;


def generateHierarchy(ids, codes, first, last):
    if first == last:
        return ids[first]

    split = findSplit(codes, first, last)

    childA = generateHierarchy(ids, codes, first, split)
    childB = generateHierarchy(ids, codes, split + 1, last)

    return (childA, childB)


def countLeadingZeros(x):
    n = 0
    while x != 0:
        x >>= 1
        n += 1
    return 32 - n


def findSplit(codes, first, last):
    code_first = codes[first]
    code_last = codes[last]

    if code_first == code_last:
        return (first + last) >> 1

    common_prefix = countLeadingZeros(code_first ^ code_last)

    split = first
    step = last - first

    while True:
        step = (step + 1) >> 1
        new_split = split + step
        if new_split < last:
            code_split = codes[new_split]
            split_prefix = countLeadingZeros(code_first ^ code_split)
            if split_prefix > common_prefix:
                split = new_split

        if step <= 1:
            break

    return split


ids = range(len(pos))
codes = [morton3D(*pos[i]) for i in ids]
_ = sorted(zip(ids, codes), key=lambda x: x[1])
ids = [_ for _, __ in _]
codes = [_ for __, _ in _]
tree = generateHierarchy(ids, codes, 0, len(ids) - 1)
print(pos)
print(tree)
exit(1)
