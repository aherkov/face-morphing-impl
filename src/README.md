# Morphing

## Prerequisites:

```
pip3 install opencv-python
sudo apt install cmake
pip3 install dlib
```

## Necessary downloads

### Haar Cascade

https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

### Facial landmarks

https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2

## Guide

The program can be run using

```
python3 main.py --src PATH_TO_YOUR_SOURCE_IMAGE --dst PATH_TO_YOUR_DESTINATION_IMAGE
```

After the algorithm is done, the results can be found in the _morphs_ subdirectory

<br>
Parts of code were written using the following references:

https://learnopencv.com/face-morph-using-opencv-cpp-python/

https://learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python

http://www.codesofinterest.com/2016/10/getting-dlib-face-landmark-detection.html
