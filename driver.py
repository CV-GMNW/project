import cv2
import sys
import os
import numpy as np
from get_correspondences import *
from video import *
from vid2img import *
from img2vid import *

def save_frames(frames, output_dir):
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    for image in frames:
        cv2.imwrite(output_dir + ("/frame%06i.jpg" % count), image)
        count += 1

def find_rel_locations(point_correspondences):
    rel_locs = [[0, 0]]

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


def place_on_black(frame, w, h, x, y):
    stitched_frame = np.zeros((h, w, 3))
    orig_w = np.shape(frame)[1]
    orig_h = np.shape(frame)[0]
    stitched_frame[y:y+orig_h, x:x+orig_w] = frame
    return stitched_frame


OUTPUT_SIZE = (0, 0)

def print_video_characteristics(video):
    print("Video properties:")
    print("    width: %ipx" % video.size()[0])
    print("    height: %ipx" % video.size()[1])
    print("    fps: %i" % video.fps())
    print("    length: %i frames (%.2f seconds)" % (video.num_frames(), video.length()))

def stitch_frames_METHOD_1(w_orig, h_orig, frames, point_correspondences):
    global OUTPUT_SIZE

    # 1. find frames locations relative to original
    print "  finding relative locations..."
    rel_locs = find_rel_locations(point_correspondences)
    cum_rel_locs = [rel_locs[0]]
    for loc in rel_locs[1:]:
        cum_sum = [a + b for [a,b] in zip(cum_rel_locs[-1], loc)]
        cum_rel_locs.append(cum_sum)

    cum_rel_locs = np.int32(np.array(cum_rel_locs))

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



if __name__ == "__main__":


    meth = sys.argv[1]
    video_path = sys.argv[2]
    output_dir = sys.argv[3]

    vid = load_video(video_path)

    print_video_characteristics(vid)
    print ""

    print "loading frames..."
    frames = list(vid.frames())
    print "finding corresponding points for every frame pair..."
    if meth == 'ORB' or meth == 'orb':
        corr = get_correspondences_orb(frames)
    elif meth == 'sift' or meth == 'SIFT':
        corr = get_correspondences_sift(frames)
    elif meth == 'surf' or meth == 'SURF':
        corr = get_correspondences_surf(frames)
    else:
        print "Invalid Input"
        quit()

    print "stitching..."
    new_frames = stitch_frames_METHOD_1(vid.size()[0], vid.size()[1], frames, corr)
    new_frames_list = [new_frame for new_frame in new_frames]

    
    # save_frames(new_frames_list, 'output')
    create_video_from_frames(new_frames_list, output_dir, OUTPUT_SIZE[0], OUTPUT_SIZE[1], vid.fps())
    print "Done."

    print_video_characteristics(vid)

    save_frames(vid.frames(), output_dir)

    print("Successfully created images from %i frames" % vid.num_frames())
    print("Saved to directory: \"" + output_dir + "\"")

