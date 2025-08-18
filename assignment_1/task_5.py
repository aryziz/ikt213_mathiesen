import cv2 as cv
import os

def write_camera_info() -> None:
    """Function to retrieve camera properties and write them to a file.
    """
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    
    try:
        fps = cap.get(cv.CAP_PROP_FPS)
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        
        output_file = os.path.join(os.getcwd(),"solutions", "camera_outputs.txt")
        with open(output_file, "w") as file:
            file.write(f"FPS: {fps}\n")
            file.write(f"Height: {height}\n")
            file.write(f"Width: {width}\n")
        
    except Exception as e:
        print(f"Error retrieving camera properties: {e}")
        return None
    
    finally:
        cap.release()

def main():
    print("Retrieving camera information...")
    write_camera_info()
    print("Camera information saved to solutions/camera_info.txt")

if __name__ == "__main__":
    main()