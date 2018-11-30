
import numpy as np
from orb_matching import orb_matching
from surf_matching import surf_matching
from sift_matching import sift_matching

def src_dest_2_correspondences(src_pnts, dest_pnts):

    return np.int32(np.reshape(np.concatenate((src_pnts, dest_pnts)), (-1, 2, 2)))

def get_correspondences(frames):
    correspondences = []
    for i in range(1, len(frames)):
        # src, dest = sift_matching(frames[i-1], frames[i])
        # src, dest = surf_matching(frames[i-1], frames[i])
        src, dest = orb_matching(frames[i-1], frames[i])
        correspondences.append(src_dest_2_correspondences(src, dest))

    return correspondences

# def get_correspondences(frames):
#     correspondences = np.zeros((len(frames), len(frames)))
#     for i in range(len(frames) - 1):
#         for j in range(1, len(frames)):
#             if j-i >= 1 and j-i <= 5:
#                 src, dest = orb_matching(frames[i-1], frames[i])
#                 correspondences[i][j] = src_dest_2_correspondences(src, dest)

#     return correspondences

