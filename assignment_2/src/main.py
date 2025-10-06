import cv2 as cv
import numpy as np
from utils import *

def main():
    try:
        img_arr = read_image("data/lena-1.png")
    except FileNotFoundError as e:
        print(e)
        return
    
    save_image(padding(img_arr, border_width=100), "solutions/lena_padded.png")
    save_image(crop(img_arr, 80, 130, 80, 130), "solutions/lena_cropped.png")
    save_image(resize(img_arr, 200, 200), "solutions/lena_resized.png")
    save_image(copy(img_arr, np.array([]),), "solutions/lena_copied.png")
    save_image(grayscale(img_arr), "solutions/lena_grayscaled.png")
    save_image(hsv(img_arr), "solutions/lena_hsv.png")
    save_image(hue_shifted(img_arr, np.empty_like(img_arr),50), "solutions/lena_hue_shifted.png")
    save_image(smoothing(img_arr), "solutions/lena_smoothed.png")
    save_image(rotation(img_arr, 90), "solutions/lena_rotated90.png")
    save_image(rotation(img_arr, 180), "solutions/lena_rotated180.png")

if __name__ == "__main__":
    main()