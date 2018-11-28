import numpy as np


def src_dest_2_correspondences(src_pnts, dest_pnts):
    return np.concatenate(src_pnts.dest_pnts).reshape((len(src_pnts), 2, 2))


def find_rel_locations(point_correspondences):
    rel_locs = [(0, 0)]

    for cli in range(len(point_correspondences)):
        avg_offset = (0, 0)
        n = len(point_correspondences[cli])
        for c in point_correspondences[cli]:
            x_offs = c[1][0] - c[0][0]
            y_offs = c[1][1] - c[0][1]
            avg_offset += (x_offs / n, y_offs / n)
        rel_locs.append(avg_offset)

    return rel_locs


def place_on_black(frame, w, h, x, y):
    stitched_frame = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            stitched_frame[i + y][j + x] = frame[i][j]
    return stitched_frame


def stitch_frames_METHOD_1(w_orig, h_orig, frames, point_correspondences):
    # 1. find frames locations relative to original
    rel_locs = find_rel_locations(point_correspondences)
    cum_rel_locs = [rel_locs[0]]
    for loc in rel_locs[1:]:
        cum_sum = tuple([a + b for (a,b) in zip(cum_rel_locs[-1], loc)])
        cum_rel_locs.append(cum_sum)

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
    w_new = max_x - min_x + w_orig
    h_new = max_y - min_y + h_orig

    # 3. position all frames in the new larger frame
    frame_positions = [(x - min_x, y - min_y) for (x, y) in cum_rel_locs]

    # 4. place frames
    frames_new = []
    for i in range(len(frames)):
        (pos_x, pos_y) = frame_positions[i]
        frames_new.append(place_on_black(frames[i], w_new, h_new, pos_x, pos_y))

    return frames_new
