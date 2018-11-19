import cv2

class Video:
    def __init__(self, cv2_vid):
        self.cv2_vid = cv2_vid

    def size(self):
        return (self.cv2_vid.get(cv2.CAP_PROP_FRAME_WIDTH), self.cv2_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def fps(self):
        return self.cv2_vid.get(cv2.CAP_PROP_FPS)

    def num_frames(self):
        return self.cv2_vid.get(cv2.CAP_PROP_FRAME_COUNT)

    def frames(self):
        success = True
        while success:
            success, image = self.cv2_vid.read()
            if success:
                yield image

    def length(self): # Unit: seconds
        return self.num_frames() / self.fps()


def load_video(video_path):
    return Video(cv2.VideoCapture(video_path))

