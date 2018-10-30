
def src_dest_2_correspondences(src_pnts, dest_pnts):
    # probably needed
    pass

def find_rel_locations(point_correspondences):
    rel_locs = [(0, 0)]

    for cli in range(len(point_correspondences)):
        avg_offset = (0, 0)
        for c in point_correspondences[cli]:
            x_offs = c[1][0] - c[0][0]
            y_offs = c[1][1] - c[0][1]
            avg_offset += (x_offs, y_offs) / len(point_correspondences[cli])
        rel_locs.append(avg_offset)

    return rel_locs

def cumulative(list_to_accum):
    cum = list_to_accum[0]
    for item in list_to_accum[1:]:
        cum.append(cum[-1] + item)
    return cum

def place_on_black(frame, w, h, x, y):
    pass

def stitch_frames_METHOD_1(w_orig, h_orig, frames, point_correspondences):
    # 1. find frames locations relative to original
    cum_rel_locs = cumulative(find_rel_locations(point_correspondences))

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

    # position all frames in the new larger frame
    frame_positions = cum_rel_locs - (min_x, min_y)

    # place frames
    frames_new = []
    for i in range(len(frames)):
        (pos_x, pos_y) = frame_positions[i]
        frames_new.append(place_on_black(frames[i], w_new, h_new, pos_x, pos_y))

    return frames_new
