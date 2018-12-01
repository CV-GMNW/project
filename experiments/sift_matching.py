import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def sift_matching(img1, img2):
    FLANN_INDEX_LSH = 6

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000,nOctaveLayers=3,edgeThreshold=50,sigma=2)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    a = time.time()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = []
    good = []

    new_matches = []
    for i in range(len(matches)):
        if len(matches[i]) == 2:
          matchesMask.append([0,0])
          new_matches.append(matches[i])
    for i,m in enumerate(new_matches):
        if m[0].distance < 0.7 *m[1].distance:
              matchesMask[i]=[1,0]
              good.append(m[0])

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return src_pts, dst_pts

def sift_matching_experiments(img1,edgeThreshold):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000,nOctaveLayers=3,edgeThreshold=edgeThreshold,sigma=2)
    kp1, des1 = sift.detectAndCompute(img1, None)
    img=cv2.drawKeypoints(img1,kp1,None)
    plt.figure()
    plt.title('sift edgeThreshold =' + str(edgeThreshold))
    plt.imshow(img)
    plt.savefig('../sift testing/sift edgeThreshold =' + str(edgeThreshold) +'.png')

