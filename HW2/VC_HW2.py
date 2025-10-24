import numpy as np
from PIL import Image
import time
import os

def dct_2d(image):
    """Compute the 2D-DCT of an image using NumPy operations."""
    M, N = image.shape
    
    # Create DCT basis matrices
    u = np.arange(M).reshape(-1, 1)
    x = np.arange(M).reshape(1, -1)
    alpha_u = np.sqrt(1 / M) * (u == 0) + np.sqrt(2 / M) * (u != 0)
    
    basis = np.cos((2 * x + 1) * u * np.pi / (2 * M))
    dct = alpha_u * (basis @ image @ basis.T) * alpha_u.T

    return dct

def idct_2d(dct):
    """Compute the 2D-IDCT of DCT coefficients using NumPy operations."""
    M, N = dct.shape

    # Create IDCT basis matrices
    u = np.arange(M).reshape(-1, 1)
    x = np.arange(M).reshape(1, -1)
    alpha_u = np.sqrt(1 / M) * (u == 0) + np.sqrt(2 / M) * (u != 0)

    basis = np.cos((2 * x + 1) * u * np.pi / (2 * M))
    image = alpha_u * (basis.T @ dct @ basis) * alpha_u.T

    return image

def psnr(original, reconstructed):
    """Calculate the PSNR between two images."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def visualize_dct(dct):
    """Visualize DCT coefficients in the log domain."""
    log_dct = np.log(np.abs(dct) + 1)  # Add 1 to avoid log(0)
    return (log_dct / np.max(log_dct) * 255).astype(np.uint8)

def dct_1d(vector):
    """Compute the 1D-DCT of a vector."""
    N = len(vector)
    u = np.arange(N).reshape(-1, 1)
    x = np.arange(N).reshape(1, -1)
    alpha_u = np.sqrt(1 / N) * (u == 0) + np.sqrt(2 / N) * (u != 0)

    basis = np.cos((2 * x + 1) * u * np.pi / (2 * N))
    return alpha_u * (basis @ vector)

def two_1d_dct(image):
    """Compute the 2D-DCT using two 1D-DCTs with optimized matrix operations."""
    # Apply 1D-DCT to rows
    row_dct = dct_1d(image.T).T

    # Apply 1D-DCT to columns
    col_dct = dct_1d(row_dct)

    return col_dct

def main():
    # Load the image and convert to grayscale
    input_path = "lena.png"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # image = Image.open(input_path).convert("L").resize((128, 128))
    image = Image.open(input_path).convert("L")
    image_array = np.asarray(image, dtype=np.float32)

    # 2D-DCT
    start_time = time.time()
    dct_coefficients = dct_2d(image_array)
    dct_time = time.time() - start_time

    # Two 1D-DCT
    start_time = time.time()
    two_1d_dct_coefficients = two_1d_dct(image_array)
    two_1d_dct_time = time.time() - start_time

    # Compare runtimes
    print(f"2D-DCT Time: {dct_time:.4f} seconds")
    print(f"Two 1D-DCT Time: {two_1d_dct_time:.4f} seconds")

    # Visualize Two 1D-DCT coefficients
    dct_visual = visualize_dct(two_1d_dct_coefficients)
    Image.fromarray(dct_visual).save(os.path.join(output_dir, "two_1d_dct_visual.png"))

    # Save Two 1D-DCT coefficients for validation
    np.save(os.path.join(output_dir, "two_1d_dct_coefficients.npy"), two_1d_dct_coefficients)

    # 2D-IDCT
    start_time = time.time()
    reconstructed_image = idct_2d(dct_coefficients)
    idct_time = time.time() - start_time

    # Save reconstructed image
    reconstructed_image_clipped = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    Image.fromarray(reconstructed_image_clipped).save(os.path.join(output_dir, "reconstructed.png"))

    # PSNR Evaluation
    psnr_value = psnr(image_array, reconstructed_image)

    # Print results
    print(f"2D-DCT Time: {dct_time:.4f} seconds")
    print(f"2D-IDCT Time: {idct_time:.4f} seconds")
    print(f"PSNR: {psnr_value:.2f} dB")

if __name__ == "__main__":
    main()
