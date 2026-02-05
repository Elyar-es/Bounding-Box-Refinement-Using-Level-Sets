import numpy as np
from skimage.feature import hog
import cv2
import joblib


def load_model(model_path="random_forest.pkl"):
    """
    Load the trained model from a file.
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

#### SECOND ITER #####
def extract_features(image):
    
    fixed_size = (128, 128)  # Resize all images to 128x128
    resized_image = cv2.resize(image, fixed_size)

    # Convert to grayscale (HOG works on grayscale)
    resized_image = (resized_image * 255).astype(np.uint8)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

    # Extract pixel intensity features
    pixel_features = gray.flatten()  # Flatten to 1D array

    # Extract HOG features
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),  # Set pixels per cell
        cells_per_block=(2, 2),   # Set cells per block
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )

    # Combine all features
    combined_features = np.concatenate([pixel_features, hog_features])
    print(len(combined_features))
    return combined_features