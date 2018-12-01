from orb_matching import orb_matching_experiments
from surf_matching import surf_matching_experiments
from sift_matching import sift_matching_experiments
from video import *
from vid2img import *
from img2vid import *
from matplotlib import pyplot as plt

if __name__ == '__main__':
    vid = load_video('../project/pano_shaky_5sec_smol.mp4')

    print_video_characteristics(vid)
    print ""

    print "loading frames..."
    frames = list(vid.frames())

    print "orb matching experiments"
    nfeatures = [100,200,400,600,1000]
    for i in nfeatures:
        orb_matching_experiments(frames[0], i)

    print "surf_matching_experiments"
    hessianThreshold = [200,300,400,500,600]
    for i in hessianThreshold:
        surf_matching_experiments(frames[0], i)

    print "sift matching experiments"
    edgeThreshold = [5,10,20,30,50]
    for i in edgeThreshold:
        sift_matching_experiments(frames[0],i)
    plt.show()

