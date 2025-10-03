import cv2 as cv
import numpy as np
import os.path
from typing import Literal

def save_image(image: np.ndarray, path: str) -> None:
    """Saves an image to the specified path.

    Args:
        image (np.ndarray): The image to be saved.
        path (str): The path where the image will be saved.
    """
    print("Saving image to: ", path, "...", sep="")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv.imwrite(path, image)

def denoise_image(image: np.ndarray) -> np.ndarray:
    """Applies Gaussian blur to denoise the input image.

    Args:
        image (np.ndarray): Input image in grayscale.

    Returns:
        np.ndarray: Denoised image.
    """
    denoised_image = cv.GaussianBlur(image, (3, 3), 0)
    return denoised_image

def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    """Applies Sobel edge detection to the input image.

    Args:
        image (np.ndarray): Input image in grayscale.

    Returns:
        np.ndarray: Image with Sobel edges detected.
    """
    image = denoise_image(image)

    sobel_xy = cv.Sobel(src=image, ddepth=cv.CV_64F, dx=1, dy=1, ksize=1)
    save_image(sobel_xy, os.path.join("solutions", "task_1_sobel.png"))
    return sobel_xy

def canny_edge_detection(image: np.ndarray, threshold_1=50, threshold_2=50) -> np.ndarray:
    """Applies Canny edge detection to the input image.

    Args:
        image (np.ndarray): Input image in grayscale.

    Returns:
        np.ndarray: Image with Canny edges detected.
    """
    image = denoise_image(image)

    canny_edges = cv.Canny(image, threshold_1, threshold_2)
    save_image(canny_edges, os.path.join("solutions", "task_2_canny.png"))
    return canny_edges


def template_match(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Performs template matching using normalized cross-correlation. Mark only red rectangles around detected templates.

    Args:
        image (np.ndarray): Input image in grayscale.
        template (np.ndarray): Template image in grayscale.

    Returns:
        np.ndarray: Image with rectangles drawn around detected templates.
    """
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    result = cv.matchTemplate(image_gray, template_gray, cv.TM_CCOEFF_NORMED)
    threshold = 0.9
    locations = np.where(result >= threshold)

    w, h = template_gray.shape[::-1]
    for pt in zip(*locations[::-1]):
        cv.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    save_image(image, os.path.join("solutions", "task_3_template_matching.png"))
    return image

def resize(image: np.ndarray, up_or_down: Literal["Up", "Down"], scale_factor: int = 2) -> np.ndarray:
    """Resizes the input image by a given scale factor. Must resize using pyramids.

    Args:
        image (np.ndarray): Input image.
        scale_factor (int): Factor by which to scale the image.
        up_or_down (Literal["Up", "Down"]): Direction of scaling. "Up" for enlargement, "Down" for reduction.

    Returns:
        np.ndarray: Resized image.
    """
    rows, cols, _channels = map(int, image.shape)
    
    if up_or_down == "Up":
        image = cv.pyrUp(image, dstsize=(cols*scale_factor, rows*scale_factor))
        print(f"** Zoom in: Image * {scale_factor}")
    elif up_or_down == "Down":
        image = cv.pyrDown(image, dstsize=(cols//scale_factor, rows//scale_factor))
        print(f"** Zoom out: Image / {scale_factor}")
    else:
        raise ValueError("up_or_down must be either 'Up' or 'Down'")
    
    save_image(image, os.path.join("solutions", f"task_4_resize_{up_or_down.lower()}_{str(scale_factor)}.png"))

    return image