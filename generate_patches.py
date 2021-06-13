
import os
from utils.patches import get_patches, get_patches_2, plot_patches, plot_patches_2
import matplotlib.pyplot as plt

# import utils.patches 
# from .. utils.patches import get_patches, get_patches_2, plot_patches, plot_patches_2
import numpy as np
import cv2
from tqdm import tqdm

def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)
    
def create_patches(dir_src, dir_out):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    size_h = 286
    stride_h = 143
    
    size_w = 286
    stride_w = 143

<<<<<<< HEAD:tools/generate_patches.py

    size_h = 364
    stride_h = 91
    
    size_w = 364
    stride_w = 91

    size_h = 512
    stride_h = 128
    
    size_w = 512
    stride_w = 128
=======
    # size_h = 256
    # stride_h = 256
    
    # size_w = 256
    # stride_w = 256
>>>>>>> 8e63d3f21dc09af99827d91edbd1b39dedffa631:generate_patches.py

    for filename in os.listdir(dir_src):
        try:
            img_path = os.path.join(dir_src, filename)
            print (img_path)
            img = cv2.imread(img_path)
            name = filename.split(".")[0]
            patches = get_patches_2(img, size_h=size_h, stride_h=stride_h, size_w=size_w, stride_w=stride_w, pad=False)
            # plot_patches_2(patches, org_img_size = org_img_size, size_h=size_h, stride_h=stride_h,  size_w=size_w, stride_w=stride_w)
            # plt.show()
            for i, patch in enumerate(tqdm(patches)):
                # Add target to each patch
                imwrite(os.path.join(dir_out, "%s_%s.png" % (name, i)), patch)
                # break
        except Exception as e:
            print(e)

if __name__ == '__main__':

    # parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    # parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    # create_patches(dir_src = './assets/forms', dir_out = './assets-gen/patches/forms')
    create_patches(dir_src = './assets-private', dir_out = '/home/greg/dev/pytorch-CycleGAN-and-pix2pix/datasets/eval/0001')
