import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def surf_matching(img1, img2):
    FLANN_INDEX_LSH = 6

    # Initiate SURF detector
    sift = cv2.xfeatures2d.SURF_create(hessianThreshold=500,nOctaves=4,nOctaveLayers=2,extended=0,upright=0)

    # find the keypoints and descriptors with SURF
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

    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=(255, 0, 0),
    #                    matchesMask=matchesMask,
    #                    flags=0)
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    # print "time:"
    # print time.time()-a
    # plt.imshow(img3, ), plt.show()

    return src_pts, dst_pts

def surf_matching_experiments(img1,hessianThreshold):
    sift = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold,nOctaves=4,nOctaveLayers=2,extended=0,upright=0)
    kp1, des1 = sift.detectAndCompute(img1, None)
    img=cv2.drawKeypoints(img1,kp1,None)
    plt.figure()
    plt.title('sift hessianThreshold =' + str(hessianThreshold))
    plt.imshow(img)
    plt.savefig('./surf testing/surf hessianThreshold =' + str(hessianThreshold) +'.png')

if __name__ == '__main__':
    src1, dst = surf_matching('ctd1.jpg', 'ctd2.jpg')
    print(src1)
