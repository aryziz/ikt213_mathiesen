from utils import *
from typing import Optional

def compare_images(dataset: str, results_dir: str, imgA: Optional[str]=None, imgB: Optional[str]=None):
    """First perform data check on dataset, then optionally compare a custom pair of images.

    Args:
        dataset (str): Dataset directory path.
        results_dir (str): Directory to save results.
        imgA (str, optional): Path to first custom image. Defaults to None.
        imgB (str, optional): Path to second custom image. Defaults to None.
    """
    ORB_THRESH  = 12
    SIFT_THRESH = 12
    process_datacheck(dataset, results_dir, orb_thresh=ORB_THRESH, sift_thresh=SIFT_THRESH)
    
    if imgA and imgB:
        r = eval_pair(imgA, imgB, results_dir, "Custom_pair",
                      orb_thresh=ORB_THRESH, sift_thresh=SIFT_THRESH)
        print("\nCustom pair:")
        print("  ORB score (good matches):", r["orb"]["score"], " time:", f"{r['orb']['time_s']*1000:.1f} ms")
        print("  SIFT score (RANSAC inliers):", r["sift"]["score"], " time:", f"{r['sift']['time_s']*1000:.1f} ms")

if __name__ == "__main__":
    DATASET_PATH_UIA = r"data/input/uia"
    RESULTS_DIR_UIA  = r"solutions/uia"

    UIA_IMG1 = r"data/input/uia/same/uia-front-close.jpg"
    UIA_IMG2 = r"data/input/uia/same/uia-front-far.png"

    compare_images(DATASET_PATH_UIA, RESULTS_DIR_UIA)
    
    DATASET_PATH_FP = r"data/input/fingerprint"
    RESULTS_DIR_FP  = r"solutions/fingerprint"
    FP_IMG1 = r"data/input/fingerprint/same_1/101_6.tif"
    FP_IMG2 = r"data/input/fingerprint/same_1/101_7.tif"
    
    compare_images(DATASET_PATH_FP, RESULTS_DIR_FP, FP_IMG1, FP_IMG2)
