from utils import *

def compare_images(dataset, results_dir, imgA=None, imgB=None):
    """Compare two images using ORB and SIFT feature matching.

    Args:
        imgA: First input image (numpy array).
        imgB: Second input image (numpy array).
    """
    ORB_THRESH  = 12   # count of good ORB matches
    SIFT_THRESH = 12   # count of RANSAC inliers
    process_datacheck(dataset, results_dir, orb_thresh=ORB_THRESH, sift_thresh=SIFT_THRESH)
    
    if imgA and imgB:
        r = eval_pair(imgA, imgB, results_dir, "Custom_pair",
                      orb_thresh=ORB_THRESH, sift_thresh=SIFT_THRESH)
        print("\nCustom pair:")
        print("  ORB score (good matches):", r["orb"]["score"], " time:", f"{r['orb']['time_s']*1000:.1f} ms")
        print("  SIFT score (RANSAC inliers):", r["sift"]["score"], " time:", f"{r['sift']['time_s']*1000:.1f} ms")

if __name__ == "__main__":
    DATASET_PATH_UIA = r"data/input/uia"
    RESULTS_DIR_UIA  = r"data/output/uia"

    UIA_IMG1 = r"data/input/uia/same/uia-front-close.jpg"
    UIA_IMG2 = r"data/input/uia/same/uia-front-far.png"

    compare_images(DATASET_PATH_UIA, RESULTS_DIR_UIA, UIA_IMG1, UIA_IMG2)
    
    DATASET_PATH_FP = r"data/input/fingerprint"
    RESULTS_DIR_FP  = r"data/output/fingerprint"
    FP_IMG1 = r"data/input/fingerprint/same_1/101_6.tif"
    FP_IMG2 = r"data/input/fingerprint/same_1/101_7.tif"
    
    compare_images(DATASET_PATH_FP, RESULTS_DIR_FP, FP_IMG1, FP_IMG2)
