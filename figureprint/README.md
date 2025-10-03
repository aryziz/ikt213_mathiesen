# Figureprint matching

**Course:** IKT213

**Student:** Aryan Mathiesen  

**Date:** 03.10.2025

**Assignment:** Figureprint matching

## Description

This assignment is an answer to Figureprint matching 

## File Structure

```
figureprint/
├── README.md
├── src/
│   ├── __init__.py
│   ├── main.py         # Entry file
│   ├── utils.py        # Image utilities
├── data/
│   ├── input/
│   │   ├── fingerprint/
│   │   │   ├── same_1/                     # Dir containing original images
│   │   │   │   ├── 101_6.tif               # Sample same fingerprint image 1
│   │   │   │   ├── 101_7.tif               # Sample same fingerprint image 2
│   │   │   ├── ...                         # Same pattern as above; same_2, same_3, ...
│   │   │   ├── different_1/
│   │   │   │   ├── 101_6.tif               # Sample different fingerprint image 1
│   │   │   │   ├── 105_6.tif               # Sample different fingerprint image 2
│   │   │   ├── ...
│   │   ├── uia/                             # Same pattern as fingerprint dir
├── solutions/
│   ├── fingerprint/
│   │   ├── same_1/                         # Dir containing processed images
│   │   │   │   ├── same_1_ORB.png
│   │   │   │   ├── same_1_SIFT.png
│   │   │   ├── ...                         # Same pattern as above; same_2, same_3, ...
│   │   │   ├── different_1/
│   │   │   │   ├── different_1_ORB.png     # Matching on diff img using ORB
│   │   │   │   ├── different_1_SIFT.png    # Matching on diff img using SIFT
│   ├── uia/
│   │   ├── same/
│   │   │   ├── same_ORB.png                # Same pattern as above


```

**Note:** This assignment was completed individually and follows the course's academic integrity policy.