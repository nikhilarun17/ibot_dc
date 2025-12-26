import cv2
import matplotlib.pyplot as plt
import numpy as np

def preprocess_image(image_path):
    """Load and preprocess the image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    return (image, blurred)

def detect_circles(gray_image, dp=1, min_dist=25, param1=50, param2=50, min_radius=10, max_radius=400):
    """Detect circles in the preprocessed image using Hough Transform."""
    circles = cv2.HoughCircles(gray_image, 
                               cv2.HOUGH_GRADIENT, 
                               dp, 
                               min_dist, 
                               param1=param1, 
                               param2=param2, 
                               minRadius=min_radius, 
                               maxRadius=max_radius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
    return circles

def visualize_circles(image,circles,save_path=None):
    """Visualize detected circles on the image."""
    if circles is not None:
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def calculate_statistics(circles):
    """Calculate statistics of detected circles."""
    if circles is None:
        return None
    
    radii = circles[0, :, 2]
    stats = {
        'count': len(radii),
        'mean_radius': np.mean(radii),
        'std_radius': np.std(radii),
        'min_radius': np.min(radii),
        'max_radius': np.max(radii)
    }
    return stats

def main(image_path, save_path=None):
    image, preprocessed_image = preprocess_image(image_path)
    circles = detect_circles(preprocessed_image)
    visualize_circles(image, circles, save_path)
    stats = calculate_statistics(circles)
    
    if stats:
        print("Detected Circles Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        print("No circles detected.")
if __name__ == "__main__":
    image_path = input("Enter the path to your image: ")  
    save_path = input("Enter the path to save the output image (or press Enter to skip saving): ")
    if save_path.strip() == "":
        save_path = None     
    main(image_path, save_path)