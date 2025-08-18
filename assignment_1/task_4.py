import cv2 as cv

class ImageProcessor:
    def __init__(self, image_path: str):
        try:
            self.image = cv.imread(image_path)
        except Exception as e:
            raise ValueError(f"An error occurred while loading the image: {e}")
        if self.image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")

    def get_image(self) -> cv.Mat:
        return self.image

def print_image_information(image: cv.Mat) -> None:
    print("Image Information:")
    print("="*30)
    print(f"Height: {image.shape[0]}")
    print(f"Width: {image.shape[1]}")
    print(f"Channels: {image.shape[2] if len(image.shape) == 3 else 1}")
    print(f"Size: {image.size} bytes")
    print(f"Data type: {image.dtype}")
    print("-"*30)
    

def main() -> None:
    img: ImageProcessor = ImageProcessor("img/lena-1.png")
    
    print_image_information(img.get_image())

if __name__ == "__main__":
    main()