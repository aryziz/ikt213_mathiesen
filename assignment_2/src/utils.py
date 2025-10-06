import cv2 as cv
import numpy as np
import os

def read_image(path: str) -> np.ndarray:
    """Reads an image from the specified path.

    Args:
        path (str): The path to the image file.

    Returns:
        np.ndarray: The read image.
    """
    image = cv.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """Saves an image to the specified path.

    Args:
        image (np.ndarray): The image to be saved.
        path (str): The path where the image will be saved.
    """
    print("Saving image to: ", path, "...", sep="")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv.imwrite(path, image)

def padding(image: np.ndarray, border_width: int = 100) -> np.ndarray:
    """Makes a border with the reflection of the image with border_width=100

    Args:
        image (np.ndarray): The input image.
        border_width (int): Width of the border to be added. Default is 100.
    """
    return cv.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv.BORDER_REFLECT)
    
def crop(image: np.ndarray, x0, x1, y0, y1) -> np.ndarray:
    """Crops the image to the specified coordinates.

    Args:
        image (np.ndarray): The input image.
        x0 (int): The starting x-coordinate.
        x1 (int): The ending x-coordinate.
        y0 (int): The starting y-coordinate.
        y1 (int): The ending y-coordinate.

    Returns:
        np.ndarray: The cropped image.
    """
    if len(image.shape) < 2:
        raise ValueError("Input image must have at least 2 dimensions")
    
    h, w = image.shape[:2]
    
    if x0 < 0 or x1 > w or y0 < 0 or y1 > h:
        raise ValueError("Crop coordinates are out of image bounds")
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Invalid crop coordinates: x0 must be less than x1 and y0 must be less than y1")
    
    return image[y0:y1, x0:x1]

def resize(image, width, height):
    """Resizes the image to the specified width and height.

    Args:
        image (np.ndarray): The input image.
        width (int): The desired width.
        height (int): The desired height.

    Returns:
        np.ndarray: The resized image.
    """
    h, w = image.shape[:2]
    if width > w or height > h:
        raise ValueError("New dimensions must be less than or equal to the original dimensions")
    return cv.resize(image, (width, height))

def copy(image: np.ndarray, emptyPictureArray: np.ndarray) -> np.ndarray:
    """Creates a manual copy of the input image.

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: A copy of the input image.
    """
    h, w = image.shape[:2]
    emptyPictureArray = np.zeros((h, w, 3), dtype=image.dtype)
    for i in range(h):
        for j in range(w):
            emptyPictureArray[i, j] = image[i, j]
    
    return emptyPictureArray

def grayscale(image: np.ndarray) -> np.ndarray:
    """Converts the input image to grayscale.

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The grayscale image.
    """
    if len(image.shape) == 2:
        return image  # Already grayscale
    elif len(image.shape) == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        raise ValueError("Input image must have 2 or 3 dimensions")
    
def hsv(image: np.ndarray) -> np.ndarray:
    """Converts the input image to HSV color space.

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The image in HSV color space.
    """
    if len(image.shape) == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2HSV)
    else:
        raise ValueError("Input image must have 3 dimensions")
    
    
def hue_shifted(image: np.ndarray, emptyPictureArray: np.ndarray, hue: int = 50) -> np.ndarray:
    """Shift image color values in RGB by the specified hue amount.

    Args:
        image (np.ndarray): The input image.
        hue (int): The amount to shift the hue.

    Returns:
        np.ndarray: The hue-shifted image.
    """
    
    if emptyPictureArray is None or emptyPictureArray.size == 0 or emptyPictureArray.shape != image.shape:
        work = image.copy()
    else:
        np.copyto(emptyPictureArray, image)
        work = emptyPictureArray
    
    shifted = (work.astype(np.int16) + int(hue)) % 256
    shifted = shifted.astype(np.uint8)
    return shifted

def smoothing(image: np.ndarray, ksize=(15, 15)) -> np.ndarray:
    """Applies Gaussian smoothing to the input image.

    Args:
        image (np.ndarray): The input image.
        ksize (tuple): The kernel size for Gaussian smoothing.

    Returns:
        np.ndarray: The smoothed image.
    """
    return cv.GaussianBlur(image, ksize, 0, borderType=cv.BORDER_DEFAULT)

def rotation(image: np.ndarray, rotation_angle: int) -> np.ndarray:
    """Rotates the input image by the specified angle. If rotation angle is 90, rotate the image 90 degrees clockwise. If 180, rotate 180 degrees, etc.

    Args:
        image (np.ndarray): The input image.
        rotation_angle (int): The angle to rotate the image.

    Returns:
        np.ndarray: The rotated image.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, -rotation_angle, 1.0)
    rotated_image = cv.warpAffine(image, rotation_matrix, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
    return rotated_image