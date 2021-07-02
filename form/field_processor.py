
# Add parent to the search path so we can reference the module here without throwing and exception 
from logging import Handler, raiseExceptions
import os, sys

from numpy.core.fromnumeric import shape
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
import numpy as np
import json

from pix2pix.options.test_options import TestOptions
from pix2pix.data import create_dataset
from pix2pix.models import create_model
from pix2pix.util.visualizer import save_images
from pix2pix.util.util import tensor2im
 
from utils.utils import current_milli_time, ensure_exists
from utils.image_utils import imwrite

from utils.resize_image import resize_image

def resize_handler(image, args_dict):
    """
        handler for resizing images :
        args example : {'width': 1024, 'height': 256, 'anchor': 'center'}
    """
    width = args_dict['width']
    height = args_dict['height']
    anchor = args_dict['anchor']

    return resize_image(image, (height, width), color=(255, 255, 255))

class FieldProcessor:
    
    def __init__(self, work_dir, models:dict = None) -> None:
        print("Initializing Field processor")
        if models == None:
            raise Exception('Invalid argument exception for modeld')
        self.work_dir = work_dir 
        self.models = models

    def postprocess(self, src):
        """
            post process extracted image
            1) Remove leftover vertical lines
        """
        # Transform source image to gray if it is not already
        if len(src.shape) != 2:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        else:
            gray = src

        # Apply adaptiveThreshold at the bitwise_not of gray
        gray = cv2.bitwise_not(gray)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Create the images that will use to extract the horizontal and vertical lines
        thresh = np.copy(bw)
        image = src
        rows = thresh.shape[0]
        verticalsize = rows // 4

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        # segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        # image = cv2.bitwise_or(bw, detected_lines)
        # viewImage(image, 'image')

        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(image, [c], -1, (255,255,255), 2)

        # viewImage(image, 'image')
        # Repair image
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
        result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
        # viewImage(detected_lines, 'detected_lines')
        # viewImage(result, 'Snippet')
        return result

    def process(self, id, key, snippet)->None:
        """
            Process data field
        """
        print("Processing field : {}".format(key))
        opt, model, config = self.__setup(key)
   
        work_dir = ensure_exists(os.path.join(self.work_dir, id, 'fields', key))
        debug_dir = ensure_exists(os.path.join(self.work_dir, id, 'fields_debug', key))
        
        opt.dataroot = work_dir
        name = 'segmenation'

        # preprocessing
        handlers = {
            "resize":resize_handler
        }

        shape_before = snippet.shape
        if 'preprocess' in config:
            for pp_config in config['preprocess']:
                # {'id': 'prepare-data-for-unet', 'type': 'resize', 'width': 1024, 'height': 256, 'anchor': 'center'}
                print(f'Preprocessing : {pp_config}' )
                pp_id = pp_config['id']
                type = pp_config['type']
                args = pp_config['args']
                
                handler = handlers.get(type, lambda: f'Unknown handler type : {type}')
                print(f'Executing handler : {pp_id}')
                ret = handler(snippet, args)
                if len(ret) == 0:
                    raise Exception(f'Handler should return processed image but got None')
                snippet = ret

        shape_after = snippet.shape
        print('Shape info *************')
        print(f'Shape before : {shape_before}')
        print(f'Shape after : {shape_after}')

        # Debug 
        if True:
            image_name = '%s.png' % (key)
            save_path = os.path.join(work_dir, image_name)                   
            imwrite(save_path, snippet)

        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        
        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            # Debug 
            if True:
                for label, im_data in visuals.items():
                    image_numpy = tensor2im(im_data)
                    # Tensor is in RGB format OpenCV requires BGR
                    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
                    image_name = '%s_%s.png' % (name, label)
                    save_path = os.path.join(debug_dir, image_name)                   
                    imwrite(save_path, image_numpy)

            label='prediction'
            fake_im_data = visuals['fake']
            image_numpy = tensor2im(fake_im_data)
            # Tensor is in RGB format OpenCV requires BGR
            image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(debug_dir, image_name)

            # return self.postprocess(image_numpy)
            return image_numpy

    def __setup(self, key):
        """
            Model setup
        """
        name = self.models[key]
        config_file = os.path.join('./models/segmenter', name, 'config.json')

        if not os.path.exists(config_file):
            raise Exception(f'Config file not found : {config_file}')

        with open(config_file) as f:
            data = json.load(f)
        
        args = data['args']

        opt = TestOptions().parse(args)  # get test options
        # hard-code parameters for test
        opt.eval = False   # test code only supports num_threads = 0
        opt.num_threads = 0   
        opt.batch_size = 1    # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        
        return opt, model, data