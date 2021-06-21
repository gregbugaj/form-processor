
from operator import ne
import os
from utils.patches import get_patches, get_patches_2, plot_patches, plot_patches_2, reconstruct_from_patches_2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import numpy as np
import cv2
from tqdm import tqdm

from PIL import Image, ImageOps

def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)

def get_debug_image(h, w, img):
    #  expand shape from 1 channel to 3 channel
    mask = np.ones((h, w), dtype = np.uint8)
    # mask = mask[:, :, None] * np.ones(3, dtype = np.uint8)[None, None, :]
    # mask = mask * 255
    debug_img = np.ones((h, 2*w, 3), dtype = np.uint8) * 255
    debug_img[:h, :w] = img
    return debug_img
    ##cv2.line(debug_img, (0, h), (debug_img.shape[1], h), (255, 0, 0), 1)
    return debug_img
    

def process(img_path, dir_out, network_parameters):
    """
        Process document
    """
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    if not os.path.isfile(img_path):
        raise Exception("File not found '{}'".format(img_path))

    size_h = 286
    stride_h = 143
    
    size_w = 286
    stride_w = 143


    size_h = 256
    stride_h = 256
    
    size_w = 256
    stride_w = 256

    img = cv2.imread(img_path)
    org_img_size = img.shape
    patches = get_patches_2(img, size_h=size_h, stride_h=stride_h, size_w=size_w, stride_w=stride_w)
    
    print(len(patches))

    for i, patch in enumerate(tqdm(patches)):
        print('I = %s' % (i))
        masked = get_debug_image(size_h, size_w, patch)
        imwrite(os.path.join(dir_out, "org_%s.png" % (i)), patch)
        # imwrite(os.path.join(dir_out, "masked_%s.png" % (i)), masked)
        # break
        
    name = img_path.split("/")[-1]
    out_image = reconstruct_from_patches_2(patches, org_img_size, size_h=size_h, stride_h=stride_h, size_w=size_w, stride_w=stride_w)[0]
    imwrite(os.path.join(dir_out, "recon_{}.png".format(name)), out_image)
    

    
def _fragment(overlay, fragment, pos=(0,0)):
    # You may need to convert the color.
    fragment = cv2.cvtColor(fragment, cv2.COLOR_BGR2RGB)
    fragment_pil = Image.fromarray(fragment)
    overlay.paste(fragment_pil, pos)

def get_fragment(img, box):
    x,y,w,h  = box
    return img[y:y+h, x:x+w]

def save_fragment(fragment, fid, dir_out):
    filename =  '%s-%s.png' % ('fragment', fid)
    filename = os.path.join(dir_out, filename)
    print('File written : %s' % (filename))
    imwrite(filename, fragment)
    return filename

def process(img_path, dir_out, network_parameters):
    """
        Process form
    """
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    if not os.path.isfile(img_path):
        raise Exception("File not found '{}'".format(img_path))
    img = cv2.imread(img_path)
    shape = img.shape
    overlay = Image.new('RGB', (shape[1], shape[0]), color=(255,255,255,0))

    # box = [x,y, w, h]
    # x,y,w,h 
    # x,y are the coordinates for the top left corner of the box, and w,h are just the width and height

    fid = '001'
    box_01= [26, 154, 901, 134]
    fragment = get_fragment(img, box_01)
    save_fragment(fragment, fid, dir_out)     
    paste_fragment(overlay, fragment, (box_01[0], box_01[1]))

    fid = '002'
    box_02= [26, 280, 887, 134]
    fragment = get_fragment(img, box_02)
    save_fragment(fragment, fid, dir_out)     
    paste_fragment(overlay, fragment, (box_02[0], box_02[1]))

    fid = '003'
    box_03= [18, 1928, 1544, 265]
    fragment = get_fragment(img, box_03)
    save_fragment(fragment, fid, dir_out)     
    paste_fragment(overlay, fragment, (box_03[0], box_03[1]))

    savepath = os.path.join(dir_out, "%s.jpg" % 'overlay')
    overlay.save(savepath, format='JPEG', subsampling=0, quality=100)

if __name__ == '__main__':
    args = parse_args()
    args.network_param = './models/form_pix2pix/latest_net_G.pth'

    args.img_src = './assets/forms-single/PID_10_5_0_94371.tif'
    args.dir_out = './assets/cleaned-examples/set-x/cleaned'

    args.debug = False
    
    img_src = args.img_src 
    dir_out = args.dir_out 
    network_parameters = args.network_param

    process(img_path = img_src, dir_out = dir_out, network_parameters = network_parameters)
