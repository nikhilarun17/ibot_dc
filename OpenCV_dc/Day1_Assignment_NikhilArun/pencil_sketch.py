import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def pencil_sketch(image_path, kernel_size=(21, 21)):
    """
    This function takes an image path as input and returns a pencil sketch version of the image.
    1. Reads the image from the given path.
    2. Converts the image to grayscale.
    3. Inverts the grayscale image.
    4. Applies Gaussian blur to the inverted image.
    5. Blends the grayscale image with the inverted blurred image to create a pencil sketch effect.
    6. Returns the pencil sketch image.     
    Parameters:
    image_path (str): Path to the input image file.
    kernel_size (tuple): Size of the Gaussian kernel. Default is (21, 21
    Returns:
    np.ndarray: Pencil sketch version of the input image.
    """

    # Validate image path
    if image_path is None:
        raise ValueError("Image path must be provided")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not os.path.isfile(image_path):
        raise ValueError(f"Invalid path - not a file: {image_path}")
    
    # Load the image and convert to grayscale
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image. Invalid image format or corrupted file: {image_path}")
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image and apply Gaussian blur
    img_invert = 255 - img_gray
    img_blur = cv2.GaussianBlur(img_invert, kernel_size, 0)
    
    # Create the pencil sketch effect by blending the grayscale image with the inverted blurred image
    img_blur_invert = 255 - img_blur
    img_sketch = np.clip((img_gray/img_blur_invert)*256, 0, 255)
    return img_sketch

def display_results(original,sketch,save_path=None):
    """
    This function displays the original and pencil sketch images side by side.
    Parameters:
    original (np.ndarray): Original input image.
    sketch (np.ndarray): Pencil sketch version of the input image.
    save_path (str): Optional path to save the displayed figure.
    """
    
    # Display the original and pencil sketch images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(sketch, cmap='gray')
    axes[1].set_title("Pencil Sketch")
    axes[1].axis('off')
    
    plt.tight_layout()
    if save_path:
        cv2.imwrite(save_path, sketch)
        print(f"Pencil sketch image saved to: {save_path}")
    else:   
        print("No save path provided, skipping saving the image.")
    plt.show()

def main():
    # Display the pencil sketch effect
    # Take input image path and save path from user
    try:
        image_path = input("Enter the path to the image file: ")
        original_image = cv2.imread(image_path)
        pencil_sketch_image = pencil_sketch(image_path)
        save_path = input("Enter the path to save the output image (or press Enter to skip saving): ")
        display_results(original_image, pencil_sketch_image, save_path if save_path else None)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")  

if __name__ == "__main__":
    main()