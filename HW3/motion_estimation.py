import numpy as np
from PIL import Image
import time

def load_image(image_path):
    """Load a grayscale image as a NumPy array."""
    return np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)

def block_matching(full_frame, ref_frame, block_size, search_range):
    """Perform full search block matching for motion estimation."""
    height, width = full_frame.shape
    motion_vectors = np.zeros((height // block_size, width // block_size, 2), dtype=np.int32)
    residual = np.zeros_like(full_frame)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            best_mse = float('inf')
            best_dx, best_dy = 0, 0

            for dx in range(-search_range, search_range + 1):
                for dy in range(-search_range, search_range + 1):
                    x_start = i + dx
                    y_start = j + dy

                    if x_start < 0 or y_start < 0 or x_start + block_size > height or y_start + block_size > width:
                        continue

                    block = ref_frame[x_start:x_start + block_size, y_start:y_start + block_size]
                    mse = np.mean((full_frame[i:i + block_size, j:j + block_size] - block) ** 2)

                    if mse < best_mse:
                        best_mse = mse
                        best_dx, best_dy = dx, dy

            motion_vectors[i // block_size, j // block_size] = [best_dx, best_dy]
            reconstructed_block = ref_frame[i + best_dx:i + best_dx + block_size, j + best_dy:j + best_dy + block_size]
            residual[i:i + block_size, j:j + block_size] = full_frame[i:i + block_size, j:j + block_size] - reconstructed_block

    return motion_vectors, residual

def motion_compensation(ref_frame, motion_vectors, block_size):
    """Perform motion compensation using motion vectors."""
    height, width = ref_frame.shape
    reconstructed_frame = np.zeros_like(ref_frame)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            dx, dy = motion_vectors[i // block_size, j // block_size]
            reconstructed_frame[i:i + block_size, j:j + block_size] = ref_frame[i + dx:i + dx + block_size, j + dy:j + dy + block_size]

    return reconstructed_frame

def psnr(original, reconstructed):
    """Calculate the PSNR between two images."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def main():
    block_size = 8
    search_ranges = [8, 16, 32]

    # Load images
    full_frame = load_image("one_gray.png")
    ref_frame = load_image("two_gray.png")

    for search_range in search_ranges:
        print(f"Search Range: {search_range}")

        # Full search block matching
        start_time = time.time()
        motion_vectors, residual = block_matching(full_frame, ref_frame, block_size, search_range)
        full_search_time = time.time() - start_time

        # Motion compensation
        reconstructed_frame = motion_compensation(ref_frame, motion_vectors, block_size)

        # PSNR evaluation
        psnr_value = psnr(full_frame, reconstructed_frame)

        # Save results
        Image.fromarray(reconstructed_frame.astype(np.uint8)).save(f"reconstructed_{search_range}.png")
        Image.fromarray(residual.astype(np.uint8)).save(f"residual_{search_range}.png")

        print(f"Full Search Time: {full_search_time:.4f} seconds")
        print(f"PSNR: {psnr_value:.2f} dB")

if __name__ == "__main__":
    main()
