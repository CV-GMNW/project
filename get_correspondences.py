
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


def get_correspondences2(frames, dist=4, meth='sift'):
    corr_per_frame = [[]]
    for b in range(1, len(frames)):
        print '   ', (b+1), '/', len(frames)
        corr = []
        for a in range(b-1, max(b-1-dist,-1), -1):
            src, dest = None, None
            if meth == 'sift':
                src, dest = sift_matching(frames[a], frames[b])
            elif meth == 'surf':
                src, dest = surf_matching(frames[a], frames[b])
            else:
                src, dest = orb_matching(frames[a], frames[b])
            corr.append(src_dest_2_correspondences(src, dest))
        corr_per_frame.append(corr)
    # print corr_per_frame[2][0]
    return corr_per_frame
