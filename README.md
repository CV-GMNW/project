# project

## Running code with anaconda
```
conda create -n cv python=2.7.15
source activate cv
pip install -r requirements.txt
```

## Usages

stabilize.py usage:

    python stabilize.py <input_video_path (mp4)> <output_video_path (avi)> <feature_detection_method(sift/surf/orb)>

vid2img.py usage:

    python vid2img.py <video_path> <output_dir>

img2vid.py usage:

    python img2vid.py <frame_images_dir> <fps> <output_video_path>
