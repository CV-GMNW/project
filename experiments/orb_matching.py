import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def orb_matching(img1, img2):
  FLANN_INDEX_LSH = 6

  # Initiate SIFT detector
  orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=50)

  # find the keypoints and descriptors with SIFT
  kp1, des1 = orb.detectAndCompute(img1,None)
  kp2, des2 = orb.detectAndCompute(img2,None)

  FLANN_INDEX_KDTREE = 1
  a = time.time()
  # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  index_params= dict(algorithm = FLANN_INDEX_LSH,
                     table_number = 6, # 12
                     key_size = 12,     # 20
                     multi_probe_level = 4) #2
  search_params = dict(checks = 50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  matches = flann.knnMatch(des1,des2,k=2)

  # Need to draw only good matches, so create a mask
  matchesMask = []
  good = []
  # ratio test as per Lowe's paper
  new_matches = []
  for i in range(len(matches)):
    if len(matches[i]) == 2:
      matchesMask.append([0,0])
      new_matches.append(matches[i])
  for i,m in enumerate(new_matches):
      if m[0].distance < 0.7 *m[1].distance:
          matchesMask[i]=[1,0]
          good.append(m[0])

  src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
  dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)


  return src_pts, dst_pts

def orb_matching_experiments(img1, nfeatures):
  orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=nfeatures)

  kp1, des1 = orb.detectAndCompute(img1,None)
  img=cv2.drawKeypoints(img1,kp1,None)
  plt.figure()
  plt.title('orb nfeatures =' + str(nfeatures))
  plt.imshow(img)
  plt.savefig('../orb testing/orb nfeatures =' + str(nfeatures) +'.png')