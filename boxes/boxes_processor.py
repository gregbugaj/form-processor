
# Add parent to the search path so we can reference the modules(craft, pix2pix) here without throwing and exception 
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import os
import numpy as np
import copy
import cv2
import numpy as np
from PIL import Image

from craft_text_detector import Craft

# import craft functions
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

from utils.resize_image import resize_image

def crop_poly_low(img, poly):
    """
        find region using the poly points
        create mask using the poly points
        do mask op to crop
        add white bg
    """
    # points should have 1*x*2  shape
    if len(poly.shape) == 2:
        poly = np.array([np.array(poly).astype(np.int32)])

    pts=poly
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg + dst

    return dst2

class BoxProcessor:
    def __init__(self, cuda: bool = False) -> None:
        print("Box processor [cuda={}]".format(cuda))
        self.cuda = cuda
        self.craft_net, self.refine_net = self.__load()

    def __load(self):
        # load models
        refine_net = load_refinenet_model(cuda=self.cuda)
        craft_net = load_craftnet_model(cuda=self.cuda)

        return craft_net, refine_net

    def process(self, snippet):
        print('Processing')
        # # read image
        # image = read_image(image)

        h=snippet.shape[0]
        w=snippet.shape[1]
        image=snippet
        # perform prediction
        prediction_result = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=True,
            long_size=w #1280
        )

        # cv_snip = np.array(img)                
        # snippet = cv2.cvtColor(cv_snip, cv2.COLOR_RGB2BGR)# convert RGB to BGR
        return snippet

