import numpy as np
import cv2
from matplotlib import pyplot as plt


def surf_matching(im1_name, im2_name):
    MIN_MATCH_COUNT = 10
    FLANN_INDEX_LSH = 6

    img1 = cv2.imread(im2_name, 0)  # queryImage
    img2 = cv2.imread(im1_name, 0)  # trainImage

    # Initiate SURF detector
    sift = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SURF
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=(255, 0, 0),
    #                    matchesMask=matchesMask,
    #                    flags=0)
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    # plt.imshow(img3, ), plt.show()

    return src_pts, dst_pts


if __name__ == '__main__':
    src1, dst = surf_matching('ctd1.jpg', 'ctd2.jpg')
    print(src1)
