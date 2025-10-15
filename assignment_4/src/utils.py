import numpy as np
import cv2 as cv

LOWES_RATIO = 0.75



def harris_corner_detector(reference_image: np.ndarray, threshold:float=0.01):
    """
    Harris Corner Detector implementation.

    Parameters:
    - reference_image: 2D numpy array representing the grayscale image.
    - threshold: float, threshold for corner detection.

    Returns:
    - image: image with detected corners marked.
    """
    gray = cv.cvtColor(reference_image, cv.COLOR_BGR2GRAY)
    
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst, None)
    
    ret, dst = cv.threshold(dst, threshold*dst.max(), 255, 0)
    dst = np.uint8(dst)
    
    _, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    
    res = np.hstack((centroids, corners))
    res = np.intp(res)
    
    img = reference_image.copy()
    
    img[res[:,1],res[:,0]] = [0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]
    
    return img


def _to_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img

def _clahe(gray):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)


def align_images(image_to_align, reference_image, max_features, good_match_percent):
    """
    Aligns image_to_align to reference_image using feature matching.

    Parameters:
    - image_to_align: 2D numpy array representing the image to be aligned.
    - reference_image: 2D numpy array representing the reference image.
    - max_features: int, maximum number of features to detect.
    - good_match_percent: float, percentage of good matches to retain.

    Returns:
    - aligned_image: aligned version of image_to_align.
    - H: homography matrix used for alignment.
    """
    im1, im2 = image_to_align, reference_image
    g1, g2 = _to_gray(im1), _to_gray(im2)

    # Preprocess: light blur + CLAHE
    g1 = _clahe(cv.GaussianBlur(g1, (3,3), 0))
    g2 = _clahe(cv.GaussianBlur(g2, (3,3), 0))

    # SIFT (tuned)
    sift = cv.SIFT_create(nfeatures=max_features, contrastThreshold=0.02, edgeThreshold=10, sigma=1.6)
    kp1, d1 = sift.detectAndCompute(g1, None)
    kp2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise ValueError("Not enough keypoints/descriptors.")
    
    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    knn12 = flann.knnMatch(d1, d2, k=2)
    knn21 = flann.knnMatch(d2, d1, k=2)

    good12 = []
    for pair in knn12:
        if len(pair) == 2 and pair[0].distance < LOWES_RATIO * pair[1].distance:
            good12.append(pair[0])
    good21 = []
    for pair in knn21:
        if len(pair) == 2 and pair[0].distance < LOWES_RATIO * pair[1].distance:
            good21.append(pair[0])

    idx12 = {(m.queryIdx, m.trainIdx): m for m in good12}
    good = []
    for m in good21:
        key = (m.trainIdx, m.queryIdx)
        if key in idx12:
            good.append(idx12[key])

    if len(good) < 8:
        raise ValueError(f"Too few mutual good matches: {len(good)}")

    good.sort(key=lambda m: m.distance)
    good = good[:max(8, min(50, len(good)))]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    
    H, inliers = cv.findHomography(pts1, pts2, cv.RANSAC, ransacReprojThreshold=3.0, confidence=0.999)
    if H is None:
        raise ValueError("Homography could not be estimated.")

    h2, w2 = im2.shape[:2]
    aligned = cv.warpPerspective(im1, H, (w2, h2))
    
    return aligned, H