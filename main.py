# Main.py
#
# Author(s): Akshay Paruchuri

import sys
import cv2
import numpy as np
from retina_transform import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong format: python retina_transform.py [image_path]")
        exit(-1)

    im_path = sys.argv[1]
    im = cv2.imread(im_path)
    # im = cv2.resize(im, (512, 320), cv2.INTER_CUBIC)

    # Make the fixation point the center of any given image
    # xc, yc = int(im.shape[1]/2), int(im.shape[0]/2)

    # Manually find the fixation point given an object of interest
    xc, yc = 186, 163

    # Sample compute coef
    # These are defaults that were used Zhibo Yang's original
    # usage of the foveat_img function
    # p = 15
    # k = 6
    # alpha = 3

    im, num_full_res_pixels = foveat_img(im, [(xc, yc)], 15, 3, 1.5)

    cv2.imwrite(im_path.split('.')[0]+'_RT.jpg', im)
