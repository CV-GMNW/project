import numpy as np
from get_correspondences import *
from orb_matching import orb_matching
from video import *
from vid2img import *
from img2vid import *

def find_rel_locations(point_correspondences):
    rel_locs = [[0, 0]]

    # rel_locs = [[[0], [0]]]
    #
    # for i in range(len(frames) - 1):
    #     for j in range(1, len(frames)):
    #         if j-i >= 1 and j-i <= 5:


    for cli in range(len(point_correspondences)):
        avg_offset = [0, 0]
        n = len(point_correspondences[cli])
        for c in point_correspondences[cli]:
            x_offs = c[1][0] - c[0][0]
            y_offs = c[1][1] - c[0][1]
            avg_offset[0] += float(x_offs) / n
            avg_offset[1] += float(y_offs) / n
        rel_locs.append(avg_offset)

    return rel_locs

def mean_without_outliers(data, m=2.0):
    data = np.array(data)
    return np.mean(data[abs(data - np.mean(data)) < m * np.std(data)])

def find_locations(corrs_per_frame_2):
    locs = []

    for corrs in corrs_per_frame_2:
        averaged_loc = [0., 0.]
        num = len(corrs)
        # print num
        for j in range(len(corrs)):
            x_offsets = []
            y_offsets = []
            for c in corrs[j]:
                x_offsets.append(c[1][0] - c[0][0])
                y_offsets.append(c[1][1] - c[0][1])
            avg_x_offs = mean_without_outliers(x_offsets)
            avg_y_offs = mean_without_outliers(y_offsets)
            averaged_loc[0] += float(locs[-1-j][0] + avg_x_offs) / float(num)
            averaged_loc[1] += float(locs[-1-j][1] + avg_y_offs) / float(num)
        locs.append(averaged_loc)

    return np.int32(np.array(locs))

def place_on_black(frame, w, h, x, y):
    stitched_frame = np.zeros((h, w, 3))
    orig_w = np.shape(frame)[1]
    orig_h = np.shape(frame)[0]
    stitched_frame[y:y+orig_h, x:x+orig_w] = frame
    return stitched_frame

OUTPUT_SIZE = (0, 0)

def stitch_frames_METHOD_1(w_orig, h_orig, frames, point_correspondences):
    global OUTPUT_SIZE

    # 1. find frames locations relative to original
    print "  finding locations..."
    # rel_locs = find_rel_locations(point_correspondences)
    # cum_rel_locs = [rel_locs[0]]
    # for loc in rel_locs[1:]:
    #     cum_sum = [a + b for [a,b] in zip(cum_rel_locs[-1], loc)]
    #     cum_rel_locs.append(cum_sum)

    # cum_rel_locs = np.int32(np.array(cum_rel_locs))
    # cum_rel_locs = find_locations(point_correspondences)
    cum_rel_locs = np.int64(list(test(frames)))
    # print cum_rel_locs

    # 2. find size of frame needed
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    for rel_loc in cum_rel_locs:
        if rel_loc[0] < min_x:
            min_x = rel_loc[0]
        if rel_loc[0] > max_x:
            max_x = rel_loc[0]
        if rel_loc[1] < min_y:
            min_y = rel_loc[1]
        if rel_loc[1] > max_y:
            max_y = rel_loc[1]
    w_new = max_x + 1 - min_x + w_orig
    h_new = max_y + 1 - min_y + h_orig

    OUTPUT_SIZE = (w_new, h_new)
    print "  generating new frames..."

    # 3. position all frames in the new larger frame
    frame_positions = [(x - min_x, y - min_y) for (x, y) in cum_rel_locs]

    # 4. place frames
    for i in range(len(frames)):
        (pos_x, pos_y) = frame_positions[i]
        yield place_on_black(frames[i], w_new, h_new, pos_x, pos_y)


def test(frames):
    grays= []
    for f in frames:
        grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    transformation_matrix=np.matrix([[1, 0, 0], [0, 1, 0]])
    old_img = grays[0]
    cumulative_x_shift=0
    cumulative_y_shift=0
    inter_frame_delay=1
    rows,cols = frames[0].shape[0],frames[0].shape[1]
    for gray in range(len(grays)):
        new_img = grays[gray]
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000,nOctaveLayers=3,edgeThreshold=100,sigma=1)

        kp1, des1 = sift.detectAndCompute(old_img, None)
        kp2, des2 = sift.detectAndCompute(new_img, None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
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


        old = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        new = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        last_transformation_matrix = transformation_matrix
        transformation_matrix = cv2.estimateRigidTransform(new, old, False)


        if transformation_matrix is None:
            transformation_matrix=last_transformation_matrix
        x_shift=transformation_matrix[0,2]
        y_shift=transformation_matrix[1,2]


        cumulative_x_shift += x_shift
        cumulative_y_shift += y_shift

        x_total = cumulative_x_shift
        y_total = cumulative_y_shift

        # correction_transformation=np.matrix([[1, 0, -x_total], [0, 1, -y_total]])


        # stabilized_frame = cv2.warpAffine(frames[gray],correction_transformation,(cols+1000,rows+1000),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)

        # cv2.imshow('original frame',frames[gray])
        # cv2.imshow('stabilised frame',stabilized_frame)
        # cv2.waitKey(inter_frame_delay)

        yield [x_total, y_total]

        old_img = new_img


if __name__ == '__main__':
    vid = load_video('vid_utils/pano_shaky_3sec_small.mp4')

    print_video_characteristics(vid)
    print ""

    print "loading frames..."
    frames = list(vid.frames())
    print "finding corresponding points between frames..."
    # corr = get_correspondences(frames)
    # corr = get_correspondences2(frames, dist=3, meth='sift')
    print "stitching..."
    # test(frames)
    new_frames = stitch_frames_METHOD_1(vid.size()[0], vid.size()[1], frames, None)
    new_frames_list = [new_frame for new_frame in new_frames]

    # save_frames(new_frames_list, 'output')
    create_video_from_frames(new_frames_list, 'stitch1_output.avi', OUTPUT_SIZE[0], OUTPUT_SIZE[1], vid.fps())
    print "Done."

