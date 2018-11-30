import cv2
import sys
import os
import numpy as np
from video import *


def load_frames_from_dir(frames_dir, ext=None):
    for f in os.listdir(frames_dir):
        if ext is None or f.endswith(ext):
            yield cv2.imread(os.path.join(frames_dir, f))

def create_video_from_frames(frames, output_path, w, h, fps):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for image in frames:
        writer.write(image.astype('uint8'))
    writer.release()




if __name__ == "__main__":

    frames_dir = sys.argv[1]
    fps = int(sys.argv[2])
    video_output_path = sys.argv[3]

    frames = list(load_frames_from_dir(frames_dir))
    create_video_from_frames(frames, video_output_path, fps)

    print("Successfully stitched %i frames into a video" % len(frames))
    print("Saved to file: \"" + video_output_path + "\"")

