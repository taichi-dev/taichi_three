import numpy as np
import sys

def pfmwrite(path, im):
    im = im.swapaxes(0, 1)
    scale = max(1e-10, -im.min(), im.max())
    h, w = im.shape[:2]
    with open(path, 'wb') as f:
        f.write(b'PF\n' if len(im.shape) >= 3 else b'Pf\n')
        f.write(f'{w} {h}\n'.encode())
        f.write(f'{scale if sys.byteorder == "big" else -scale}\n'.encode())
        f.write((im / scale).astype(np.float32).tobytes())
