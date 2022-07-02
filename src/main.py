import argparse
import sys

import cv2

import morphing

parser = argparse.ArgumentParser(prog=sys.argv[0])
parser.add_argument('--src', '-s')
parser.add_argument('--dst', '-d')
args = parser.parse_args()

img_src, img_dst = cv2.imread(args.src), cv2.imread(args.dst)
img_src, img_dst = cv2.resize(img_src, (500, 500), interpolation=cv2.INTER_LINEAR), \
                   cv2.resize(img_dst, (500, 500), interpolation=cv2.INTER_LINEAR)

# depending if using linear or advanced morphing algorithm
use_advanced_algorithm = True
if use_advanced_algorithm:
    morphing_algorithm = morphing.AdvancedMorph(img_src, img_dst)
else:
    morphing_algorithm = morphing.LinearMorph(img_src, img_dst)

# number of morphing steps
n_steps = 5

morphs = morphing_algorithm.morph(n_steps)

for (i, img) in enumerate(morphs):
    cv2.imwrite("morphs/morph_step_" + str(i) + ".png", img)

print("morphing finished")
