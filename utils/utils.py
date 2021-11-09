import os
import time

from PIL import Image
import cv2
import numpy as np

def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)   
    return dir   

def current_milli_time():
    return round(time.time() * 1000)   


def make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    if False:     
        print("The image size needs to be a multiple of 4. "
                "The loaded image size was (%d, %d), so it was adjusted to "
                "(%d, %d). This adjustment will be done to all images "
                "whose sizes are not multiples of 4" % (ow, oh, w, h))

    return img.resize((w, h), method)    

def make_power_2_cv2(img, base, method=Image.BICUBIC):

    print(f'InShape : {img.shape}')
    pil_snippet = Image.fromarray(img)
    pil_snippet = make_power_2(pil_snippet, base=base, method=method)
    cv_snip = np.array(pil_snippet)                
    cv_img = cv2.cvtColor(cv_snip, cv2.COLOR_RGB2BGR)# convert RGB to BGR

    print(f'OutShape : {cv_img.shape}')

    return cv_img


