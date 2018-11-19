import cv2
import sys
import os
from video import *

def save_frames(frames, output_dir):
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    for image in frames:
        cv2.imwrite(output_dir + ("/frame%06i.jpg" % count), image)
        count += 1

def print_video_characteristics(video):
    print("Video properties:")
    print("    width: %ipx" % video.size()[0])
    print("    height: %ipx" % video.size()[1])
    print("    fps: %i" % video.fps())
    print("    length: %i frames (%.2f seconds)" % (video.num_frames(), video.length()))



if __name__ == "__main__":

    video_path = sys.argv[1]
    output_dir = sys.argv[2]

    video = load_video(video_path)

    print_video_characteristics(video)

    save_frames(video.frames(), output_dir)

    print("Successfully created images from %i frames" % video.num_frames())
    print("Saved to directory: \"" + output_dir + "\"")

