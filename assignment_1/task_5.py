import cv2 as cv
import os

def write_to_file(file_path: str, content: str) -> None:
    """Helper function to write content to a file.
    
    Args:
        file_path (str): Path to the file where content will be written.
        content (str): Content to write to the file.
    """
    with open(file_path, "w") as file:
        file.write(content)
        
    print("Camera information saved to", file_path)

def get_camera_info() -> cv.VideoCapture:
    """Function to retrieve camera properties and write them to a file.
    """
    print("Retrieving camera information...")
    cap = cv.VideoCapture(0)
    print("Camera opened:", cap.isOpened())
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    
    return cap

def save_camera_properties(cap: cv.VideoCapture, file_path: str) -> None:
    """Function to save camera properties to a file.
    
    Args:
        cap (cv.VideoCapture): The VideoCapture object for the camera.
        file_path (str): Path to the file where camera properties will be saved.
    """
    properties = {
        "FPS": cap.get(cv.CAP_PROP_FPS),
        "Height": cap.get(cv.CAP_PROP_FRAME_HEIGHT),
        "Width": cap.get(cv.CAP_PROP_FRAME_WIDTH),
    }
    
    content = "\n".join(f"{key}: {value}" for key, value in properties.items())
    write_to_file(file_path, content)


def main():
    cap_info = get_camera_info()
    if cap_info:
        save_camera_properties(cap_info, os.path.join("solutions/", "camera_outputs.txt"))
        cap_info.release()
    

if __name__ == "__main__":
    main()