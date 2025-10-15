import cv2 as cv
import numpy as np

from utils import harris_corner_detector, align_images

if __name__ == "__main__":
    img = cv.imread("data/reference_img.png")
    
    img = harris_corner_detector(img, threshold=0.01)
    
    cv.imwrite("solutions/ref_harris.png", img)
    
    align_img = cv.imread("data/align_this.jpg")
    ref_img = cv.imread("data/reference_img.png")
    
    img, h = align_images(align_img, ref_img, 5000, 0.15)
    
    cv.imwrite("solutions/aligned.png", img)