import cv2
import sys
import os

video_path = sys.argv[1]
output_dir = sys.argv[2]

# Load video
vid = cv2.VideoCapture(video_path)

width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = vid.get(cv2.CAP_PROP_FPS)
num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

print("Video properties:")
print("    width: %ipx" % width)
print("    height: %ipx" % height)
print("    fps: %i" % fps)
print("    length: %i frames (%.2f seconds)" % (num_frames, num_frames / fps))

# create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

count = 0
success = True
while success:
    success, image = vid.read()
    if success:
        cv2.imwrite(output_dir + ("/frame%06i.jpg" % count), image)
        count += 1
        if count % fps == 0:
            print("Read %i seconds of video.." % (count / fps))

print("Successfully created images from %i frames" % count)
print("Saved to directory: \"" + output_dir + "\"")
