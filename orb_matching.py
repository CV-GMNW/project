import numpy as np
import cv2
from matplotlib import pyplot as plt


def orb_matching(img1, img2):
  MIN_MATCH_COUNT = 10
  FLANN_INDEX_LSH = 6

  # Initiate SIFT detector
  orb = cv2.ORB_create()

  # find the keypoints and descriptors with SIFT
  kp1, des1 = orb.detectAndCompute(img1,None)
  kp2, des2 = orb.detectAndCompute(img2,None)

  FLANN_INDEX_KDTREE = 0
  # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  index_params= dict(algorithm = FLANN_INDEX_LSH,
                     table_number = 6, # 12
                     key_size = 12,     # 20
                     multi_probe_level = 2) #2
  search_params = dict(checks = 50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  matches = flann.knnMatch(des1,des2,k=2)

  # for mat in matches:
  #   img_idx = mat.queryIdx
  #   img2_idx = mat.trainIdx
  #   (x1,y1) = kp1[img1_idx].pt
  #   (x2,y2) = kp2[img2_idx].pt

  #     # Append to each list
  #   list_kp1.append((x1, y1))
  #   list_kp2.append((x2, y2))

  # store all the good matches as per Lowe's ratio test.
  # good = []
  # for m,n in matches:
  #     if m.distance < 0.7*n.distance:
  #         good.append(m)

  # Need to draw only good matches, so create a mask
  matchesMask = [[0,0] for i in range(len(matches))]
  arr = np.asarray(matches)
  good = []
  # ratio test as per Lowe's paper
  for i,(m,n) in enumerate(matches):
      if m.distance < 0.5 *n.distance:
          matchesMask[i]=[1,0]
          good.append(m)

  # MIN_MATCH_COUNT = 1
  # if len(good) > MIN_MATCH_COUNT:
  src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
  dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

  return src_pts, dst_pts

if __name__ == '__main__':
  img1 = cv2.imread('ctd2.jpg', 0)
  img2 = cv2.imread('ctd1.jpg', 0)
  src1, dst = orb_matching(img1, img2)
  print(src1)