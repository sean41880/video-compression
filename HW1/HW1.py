import sys
from PIL import Image
import numpy as np
import os

def to_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)

def main(in_path, out_dir):
    os.makedirs(out_dir, exist_ok = True)
    img = Image.open(in_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32)

    R = arr[:,:,0] 
    G = arr[:,:,1] 
    B = arr[:,:,2]

    # RGB grayscale
    Image.fromarray(to_uint8(R), mode="L").save(os.path.join(out_dir, "R.png"))
    Image.fromarray(to_uint8(G), mode="L").save(os.path.join(out_dir, "G.png"))
    Image.fromarray(to_uint8(B), mode="L").save(os.path.join(out_dir, "B.png"))

    # YUV (BT.601)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.169 * R - 0.331 * G + 0.5 * B + 128.0
    V =  0.5 * R - 0.419 * G - 0.081 * B + 128.0
    Image.fromarray(to_uint8(Y), mode="L").save(os.path.join(out_dir, "Y.png"))
    Image.fromarray(to_uint8(U), mode="L").save(os.path.join(out_dir, "U.png"))
    Image.fromarray(to_uint8(V), mode="L").save(os.path.join(out_dir, "V.png"))

    # YCbCr (BT.601 full-range)
    Cb  = 128.0 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr  = 128.0 + 0.5 * R - 0.418688 * G - 0.081312 * B
    Image.fromarray(to_uint8(Cb), mode="L").save(os.path.join(out_dir, "Cb.png"))
    Image.fromarray(to_uint8(Cr), mode="L").save(os.path.join(out_dir, "Cr.png"))

if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "lena.png"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    main(in_path, out_dir)