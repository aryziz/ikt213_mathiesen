import cv2 as cv
import numpy as np
from utils import *

def main():
    try:
        img_arr = read_image("images/input/lena-1.png")
    except FileNotFoundError as e:
        print(e)
        return
    
    save_image(padding(img_arr, border_width=100), "images/output/lena_padded.png")
    save_image(crop(img_arr, 200, 500, 200, 500), "images/output/lena_cropped.png")
    save_image(resize(img_arr, 100, 200), "images/output/lena_resized.png")
    save_image(copy(img_arr, np.array([]),), "images/output/lena_copied.png")
    save_image(grayscale(img_arr), "images/output/lena_grayscaled.png")
    save_image(hsv(img_arr), "images/output/lena_hsv.png")
    save_image(hue_shifted(img_arr, np.empty_like(img_arr),50), "images/output/lena_hue_shifted.png")
    save_image(smoothing(img_arr), "images/output/lena_smoothed.png")
    save_image(rotation(img_arr, 90), "images/output/lena_rotated90.png")
    save_image(rotation(img_arr, 180), "images/output/lena_rotated180.png")

if __name__ == "__main__":
    main()