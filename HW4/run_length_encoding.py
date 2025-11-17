import numpy as np
from PIL import Image
import os

def load_image(image_path):
    """Load a grayscale image as a NumPy array."""
    return np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)

def dct_2d(image):
    """Compute the 2D-DCT of an image using NumPy operations."""
    M, N = image.shape
    u = np.arange(M).reshape(-1, 1)
    x = np.arange(M).reshape(1, -1)
    alpha_u = np.sqrt(1 / M) * (u == 0) + np.sqrt(2 / M) * (u != 0)
    basis = np.cos((2 * x + 1) * u * np.pi / (2 * M))
    return alpha_u * (basis @ image @ basis.T) * alpha_u.T

def idct_2d(dct):
    """Compute the 2D-IDCT of DCT coefficients using NumPy operations."""
    M, N = dct.shape
    u = np.arange(M).reshape(-1, 1)
    x = np.arange(M).reshape(1, -1)
    alpha_u = np.sqrt(1 / M) * (u == 0) + np.sqrt(2 / M) * (u != 0)
    basis = np.cos((2 * x + 1) * u * np.pi / (2 * M))
    return alpha_u * (basis.T @ dct @ basis) * alpha_u.T

def quantize(block, quant_table):
    """Quantize a block using the given quantization table."""
    return np.round(block / quant_table).astype(np.int32)

def dequantize(block, quant_table):
    """Dequantize a block using the given quantization table."""
    return block * quant_table

def zigzag_scan(block):
    """Perform zigzag scan on an 8x8 block."""
    zigzag_order = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]
    return [block[i, j] for i, j in zigzag_order]

def run_length_encode(block):
    """Perform run-length encoding on a zigzag-scanned block."""
    zigzag = zigzag_scan(block)
    encoded = []
    count = 0

    for value in zigzag:
        if value == 0:
            count += 1
        else:
            if count > 0:
                encoded.append((0, count))
                count = 0
            encoded.append((value, 0))

    if count > 0:
        encoded.append((0, count))

    return encoded

def run_length_decode(encoded):
    """Perform run-length decoding to reconstruct a zigzag-scanned block."""
    zigzag = []

    for value, count in encoded:
        if value == 0:
            zigzag.extend([0] * count)
        else:
            zigzag.append(value)

    block = np.zeros((8, 8), dtype=np.int32)
    zigzag_order = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]

    for idx, (i, j) in enumerate(zigzag_order):
        block[i, j] = zigzag[idx]

    return block

def main():
    quant_table_1 = np.array([
        [10, 7, 6, 10, 14, 24, 31, 37],
        [7, 8, 11, 16, 25, 36, 36, 33],
        [8, 10, 14, 24, 34, 41, 37, 34],
        [8, 10, 13, 17, 31, 52, 48, 37],
        [11, 13, 22, 34, 41, 65, 62, 46],
        [14, 21, 33, 38, 49, 62, 68, 55],
        [29, 38, 47, 52, 62, 73, 72, 61],
        [43, 55, 57, 59, 67, 60, 62, 59]
    ])

    quant_table_2 = np.full((8, 8), 59)

    image = load_image("lena.png")
    height, width = image.shape

    # Divide image into 8x8 blocks
    blocks = [
        image[i:i + 8, j:j + 8]
        for i in range(0, height, 8)
        for j in range(0, width, 8)
    ]

    encoded_sizes = []

    for quant_table in [quant_table_1, quant_table_2]:
        encoded_blocks = []

        for block in blocks:
            dct_block = dct_2d(block)
            quantized_block = quantize(dct_block, quant_table)
            encoded_block = run_length_encode(quantized_block)
            encoded_blocks.append(encoded_block)

        # Calculate encoded size
        encoded_size = sum(len(block) for block in encoded_blocks)
        encoded_sizes.append(encoded_size)

        # Decode and reconstruct image
        decoded_blocks = [
            dequantize(run_length_decode(block), quant_table)
            for block in encoded_blocks
        ]

        reconstructed_image = np.zeros_like(image)

        for idx, block in enumerate(decoded_blocks):
            i = (idx // (width // 8)) * 8
            j = (idx % (width // 8)) * 8
            reconstructed_image[i:i + 8, j:j + 8] = idct_2d(block)

        Image.fromarray(np.clip(reconstructed_image, 0, 255).astype(np.uint8)).save(f"reconstructed_quant_table_{quant_table[0, 0]}.png")

    print(f"Encoded sizes: {encoded_sizes}")

if __name__ == "__main__":
    main()
