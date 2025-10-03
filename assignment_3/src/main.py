import cv2 as cv
from utils import *

def task_1():
    file_path = os.path.join("data", "lambo.png")
    lambo_img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    sobel_edge_detection(lambo_img)
    
def task_2():
    file_path = os.path.join("data", "lambo.png")
    lambo_img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    canny_edge_detection(lambo_img)


def task_3():
    shapes_path = os.path.join("data", "shapes.png")
    template_path = os.path.join("data", "shapes_template.jpg")
    
    shapes_img = cv.imread(shapes_path)
    template_img = cv.imread(template_path)
    template_match(shapes_img, template_img)
    
def task_4():
    lambo_path = os.path.join("data", "lambo.png")
    lambo_img = cv.imread(lambo_path)
    
    resize(lambo_img, "Up", 2)
    resize(lambo_img, "Down", 2)

def main():
    task_1()
    task_2()
    task_3()
    task_4()


if __name__ == "__main__":
    main()