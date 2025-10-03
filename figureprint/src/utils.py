# fingerprint_compare_datacheck.py
import os, time, math, cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

def preprocess_otsu_binary(img_gray: np.ndarray) -> np.ndarray:
    """ Binarize grayscale image using Otsu's method.

    Args:
        img_gray (np.ndarray): input grayscale image.

    Returns:
        np.ndarray: binarized image.
    """
    _, img_bin = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return img_bin

def preprocess_enhanced_gray(img_gray: np.ndarray) -> np.ndarray:
    """ Enhance grayscale image using CLAHE and Gaussian blur. Alternative to otsu binarization for SIFT.

    Args:
        img_gray (np.ndarray): input grayscale image.

    Returns:
        np.ndarray: enhanced grayscale image.
    """
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(img_gray)
    g = cv.GaussianBlur(g, (3, 3), 0)
    return g

def read_gray(path: str) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def match_orb_knn(img1_gray: np.ndarray, img2_gray: np.ndarray, nfeatures=1000, ratio=0.7) -> Tuple[int, List, List, List, np.ndarray]:
    """Match two images using ORB features and KNN matching with ratio test. Preprocessing with Otsu's binarization.

    Args:
        img1_gray (np.ndarray): input image 1 (grayscale).
        img2_gray (np.ndarray): input image 2 (grayscale).
        nfeatures (int, optional): maximum number of ORB features to retain. Defaults to 1000.
        ratio (float, optional): Lowe's ratio test threshold. Defaults to 0.7.

    Returns:
        Tuple[int, List, List, List, np.ndarray]: _description_
    """
    img1 = preprocess_otsu_binary(img1_gray)
    img2 = preprocess_otsu_binary(img2_gray)

    orb = cv.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, kp1, kp2, [], None

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    vis = cv.drawMatches(img1, kp1, img2, kp2, good, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good), kp1, kp2, good, vis

def match_sift_flann_ransac(img1_gray: np.ndarray, img2_gray: np.ndarray, ratio=0.7, ransacReproj=3.0) -> Tuple[int, List, List, List, np.ndarray]:
    """Match two images using SIFT features, FLANN-based matcher, and RANSAC. Preprocessing with Otsu's binarization.

    Args:
        img1_gray (np.ndarray): input image 1 (grayscale).
        img2_gray (np.ndarray): input image 2 (grayscale).
        ratio (float, optional): Lowe's ratio test threshold. Defaults to 0.7.
        ransacReproj (float, optional): RANSAC reprojection threshold. Defaults to 3.0.

    Returns:
        Tuple[int, List, List, List, np.ndarray]:
            - inliers (int): number of inlier matches after RANSAC.
            - kp1 (List): keypoints in image 1.
            - kp2 (List): keypoints in image 2.
            - inlier_matches (List): list of inlier matches.
            - vis (np.ndarray): visualization image with inlier matches drawn.
    """
    g1 = preprocess_otsu_binary(img1_gray)
    g2 = preprocess_otsu_binary(img2_gray)
    # Better alternative:
    # g1 = preprocess_enhanced_gray(img1_gray)
    # g2 = preprocess_enhanced_gray(img2_gray)

    sift = cv.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    if des1 is None or des2 is None:
        return 0, kp1, kp2, [], None

    index_params = dict(algorithm=1, trees=5) 
    search_params = dict(checks=64)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    raw = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    if len(good) < 4:
        vis = cv.drawMatches(g1, kp1, g2, kp2, good, None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return 0, kp1, kp2, good, vis

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.estimateAffinePartial2D(pts1, pts2, method=cv.RANSAC, ransacReprojThreshold=ransacReproj)
    inliers = int(mask.sum()) if mask is not None else 0
    inlier_matches = [gm for gm, keep in zip(good, mask.ravel().tolist()) if keep] if mask is not None else []
    vis = cv.drawMatches(g1, kp1, g2, kp2, inlier_matches, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return inliers, kp1, kp2, inlier_matches, vis

# ---------- Evaluation helpers ----------
def eval_pair(img1_path: str, img2_path: str, outdir: str, pair_name: str,
              orb_thresh: int = 20, sift_thresh: int = 12) -> Dict[str, Dict]:
    """Evaluate a pair of images using ORB and SIFT feature matching.

    Args:
        img1_path (str): Path to the first input image.
        img2_path (str): Path to the second input image.
        outdir (str): Directory to save output visualizations.
        pair_name (str): Name identifier for the image pair.
        orb_thresh (int, optional): Threshold seed for ORB. Defaults to 20.
        sift_thresh (int, optional): Threshold seed for SIFT. Defaults to 12.

    Returns:
        Dict[str, Dict]: Dictionary containing scores, predictions, times, and keypoint counts for both ORB and SIFT.
    """
    os.makedirs(outdir, exist_ok=True)
    g1 = read_gray(img1_path)
    g2 = read_gray(img2_path)

    # Pipeline A
    t0 = time.perf_counter()
    orb_score, kp1a, kp2a, matches_a, vis_a = match_orb_knn(g1, g2)
    tA = time.perf_counter() - t0

    # Pipeline B
    t0 = time.perf_counter()
    sift_score, kp1b, kp2b, matches_b, vis_b = match_sift_flann_ransac(g1, g2)
    tB = time.perf_counter() - t0

    # Decisions
    orb_pred = 1 if orb_score > orb_thresh else 0
    sift_pred = 1 if sift_score > sift_thresh else 0
    
    if vis_a is not None:
        cv.imwrite(os.path.join(outdir, f"{pair_name}_ORB.png"), vis_a)
    if vis_b is not None:
        cv.imwrite(os.path.join(outdir, f"{pair_name}_SIFT.png"), vis_b)


    return {
        "orb": {"score": orb_score, "pred": orb_pred, "time_s": tA, "kpts": (len(kp1a), len(kp2a))},
        "sift": {"score": sift_score, "pred": sift_pred, "time_s": tB, "kpts": (len(kp1b), len(kp2b))}
    }

def process_datacheck(dataset_dir: str, outdir_base: str,
                      orb_thresh: int = 20, sift_thresh: int = 12) -> None:
    
    """Process a dataset directory for data check and comparison.

    Raises:
        RuntimeError: _no valid pairs processed. Check dataset_dir and folder structure._

    Returns:
        _None_ 
    """
    os.makedirs(outdir_base, exist_ok=True)
    y_true, y_orb, y_sift = [], [], []
    t_orb, t_sift = [], []
    k_orb, k_sift = [], []
    processed = 0

    for folder in sorted(os.listdir(dataset_dir)):
        fpath = os.path.join(dataset_dir, folder)
        if not os.path.isdir(fpath):
            continue

        # Grab all images (flat, not recursive)
        imgs = sorted([
            p for p in os.listdir(fpath)
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
        ])

        if len(imgs) < 2:
            print(f"[WARN] Skip {folder}: need ≥2 images, got {len(imgs)}")
            continue
        
        img1, img2 = os.path.join(fpath, imgs[0]), os.path.join(fpath, imgs[1])
        pair_name = os.path.basename(folder)

        res = eval_pair(img1, img2, os.path.join(outdir_base, folder), pair_name,
                        orb_thresh=orb_thresh, sift_thresh=sift_thresh)

        # Ground truth from folder name
        gt = 1 if "same" in folder.lower() else 0

        y_true.append(gt)
        y_orb.append(res["orb"]["pred"])
        y_sift.append(res["sift"]["pred"])
        t_orb.append(res["orb"]["time_s"])
        t_sift.append(res["sift"]["time_s"])
        k_orb.append(sum(res["orb"]["kpts"]))
        k_sift.append(sum(res["sift"]["kpts"]))
        processed += 1

        print(f"{folder:25s} | GT={gt} | "
              f"ORB: score={res['orb']['score']:3d} pred={res['orb']['pred']} time={res['orb']['time_s']*1000:.1f}ms | "
              f"SIFT: score={res['sift']['score']:3d} pred={res['sift']['pred']} time={res['sift']['time_s']*1000:.1f}ms")

    if processed == 0:
        raise RuntimeError("No valid pairs processed. Check dataset_dir and folder structure.")

    # Summary
    def acc(y, p): 
        return sum(int(a == b) for a, b in zip(y, p)) / len(y)

    print("\n--- Summary (Data Check & Comparison) ---")
    print(f"Pairs processed: {processed}")
    print(f"Accuracy ORB:  {acc(y_true, y_orb)*100:.1f}%   "
          f"avg time: {float(np.mean(t_orb))*1000:.1f} ms   avg keypoints: {float(np.mean(k_orb)):.0f}")
    print(f"Accuracy SIFT: {acc(y_true, y_sift)*100:.1f}%   "
          f"avg time: {float(np.mean(t_sift))*1000:.1f} ms   avg keypoints: {float(np.mean(k_sift)):.0f}")
