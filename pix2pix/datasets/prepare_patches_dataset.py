from typing import Counter
import cv2
from resize_image import resize_image
import os
import glob
from PIL import Image, ImageOps

import multiprocessing as mp

from concurrent.futures.thread import ThreadPoolExecutor

help_msg = """
The processed images will be placed at --output_dir.
Example usage:

python ./prepare_patches_dataset.py  --input_dir ../datasets/patches --output_dir ../datasets/patches/ready

"""

def load_resized_img(path, size):
    return Image.open(path).convert('RGB')#.resize(size)
    # return Image.open(path).convert('RGB').resize(size)
    
def process(input_dir, output_dir, phase):
    save_phase = 'test' if phase == 'test' else 'train'

    savedir = os.path.join(output_dir, save_phase)
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir + 'A', exist_ok=True)
    os.makedirs(savedir + 'B', exist_ok=True)

    print("Directory structure prepared at %s" % output_dir)
    
    segmap_expr = os.path.join(input_dir, phase) + "/mask/*.png"
    segmap_paths = glob.glob(segmap_expr)
    segmap_paths = sorted(segmap_paths)

    target_expr = os.path.join(input_dir, phase) + "/image/*.png"
    target_paths = glob.glob(target_expr)
    target_paths = sorted(target_paths)

    print("Total masks  : %s" %  len(segmap_paths) )
    print("Total images : %s" %  len(target_expr) )

    # filter and only get 

    assert len(segmap_paths) == len(target_paths), \
        "%d images that match [%s], and %d images that match [%s]. Aborting." % (len(segmap_paths), segmap_expr, len(target_paths), target_expr)

    assert len(segmap_paths),"No Images found in the directory. Aborting."

    w = 700
    h = 350    

    # HICFA    
    w = 1024
    h = 1024

    # # box 33 : *.png
    # w = 1000
    # h = 160
    # box 33, box_2

    w = 1000
    h = 160
    
    # # diagnosis_code
    # w = 1680
    # h = 265

    # # service_lines
    # w = 2532
    # h = 1024

    # box 33
    w = 1024
    h = 256
    
    # # HICFAPhone07 UNET
    # w = 512
    # h = 160   
    
    # # HCFA05PatientAddressOne 
    # w = 1024
    # h = 256

    # # service_lines
    w = 1024
    h = 192   
    
    # Segmentation Mask
    w = 1700
    h = 2366

    # Segmentation Mask
    w = 1792
    h = 2494

    def __process(segmap_path, target_path, i, total):
        print(f'Starting process {total}: {i}')
        segmap = cv2.imread(segmap_path)
        target = cv2.imread(target_path)

        # no need to resize on this one
        if False:
            segmap = resize_image(segmap, (h, w), color=(255, 255, 255))                 
            target = resize_image(target, (h, w), color=(255, 255, 255))                 

        # convert colorspace from OpenCV to PIL 
        segmap = cv2.cvtColor(segmap, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        segmap = Image.fromarray(segmap)
        target = Image.fromarray(target)

        # segmap = load_resized_img(segmap_path, (w, h))
        # segmap = ImageOps.invert(segmap)
        # photo = load_resized_img(photo_path, (w, h))

        # h = segmap.size[1]
        # w = segmap.size[0]
        # data for pix2pix where the two images are placed side-by-side
        # sidebyside = Image.new('RGB', (512, 256))
        # sidebyside.paste(segmap, (256, 0))
        # sidebyside.paste(photo, (0, 0))
        
        sidebyside = Image.new('RGB', (w * 2, h))
        sidebyside.paste(segmap, (w, 0))
        sidebyside.paste(target, (0, 0))
        
        savepath = os.path.join(savedir, "%d.png" % i)
        sidebyside.save(savepath, format='PNG', subsampling=0, quality=100)
        # sidebyside.save(savepath, format='PNG', subsampling=0, quality=100)

        # data for cyclegan where the two images are stored at two distinct directories
        savepath = os.path.join(savedir + 'A', "%d_A.png" % i)
        target.save(savepath, format='PNG', subsampling=0, quality=100)
        savepath = os.path.join(savedir + 'B', "%d_B.png" % i)
        segmap.save(savepath, format='PNG', subsampling=0, quality=100)
    
        if i % (total // 10) == 0:
            print("%d / %d" %(i, total))

    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:      
        for i, (segmap_path, target_path) in enumerate(zip(segmap_paths, target_paths)):
            executor.submit(__process, segmap_path, target_path, i, len(segmap_paths)) 

    print('All tasks has been finished')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the input directory.')
    parser.add_argument('--output_dir', type=str, required=True,
                        default='./datasets/patches/ready',
                        help='Directory the output images will be written to.')
    opt = parser.parse_args()

    print(help_msg)
    
    print('Preparing Dataset for train phase')
    process(opt.input_dir, opt.output_dir, "train")

    print('Preparing Dataset for test phase')
    process(opt.input_dir, opt.output_dir, "test")

    print('Done')

    

