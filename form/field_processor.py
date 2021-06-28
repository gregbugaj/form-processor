import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
import numpy as np
import time

# Add parent to the search path so we can reference the module here without throwing and exception 
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from utils.nms import nms, non_max_suppression_fast

from pix2pix.options.test_options import TestOptions
from pix2pix.data import create_dataset
from pix2pix.models import create_model
from pix2pix.util.visualizer import save_images
from pix2pix.util.util import tensor2im
 
def viewImage(image, name='Display'):
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)

def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)        

class FieldProcessor:
    
    def __init__(self, work_dir) -> None:
        print("Initializing Field processor")
        self.work_dir=work_dir
        
        # models can be shared
        self.models=dict()
        
        self.models['HCFA02'] = 'HCFA02'
        self.models['HCFA05_ADDRESS'] = 'HCFA02' # Reused
        self.models['HCFA05_CITY'] = 'HCFA02' # Reused
        self.models['HCFA05_STATE'] = 'HCFA02' # Reused
        self.models['HCFA05_ZIP'] = 'HCFA02' # Reused
        self.models['HCFA05_PHONE'] = 'HCFA02' # Reused

        self.models['HCFA33_BILLING'] = 'box33_pix2pix'
        self.models['HCFA21'] = 'diagnosis_code'

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
        # key = fragment['key']
        print("Processing field : {}".format(key))
        # snippet = fragment['snippet_overlay']
        opt, model = self.__setup(key)
   
        work_dir = os.path.join(self.work_dir, id, 'fields', key)
        debug_dir = os.path.join(self.work_dir, id, 'fields_debug', key)
       
        ensure_exists(work_dir)
        ensure_exists(debug_dir)

        print(f'debug_dir : {debug_dir}')
        opt.dataroot = work_dir
        name = 'segmenation'

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
        import json

        name = self.models[key]
        config_file = os.path.join('./models/segmenter', name, 'config.json')

        if not os.path.exists(config_file):
            raise Exception(f'Config file not found : {config_file}')

        with open(config_file) as f:
            data = json.load(f)
        
        args_default = data['args']
        print(data)

        if False:
            args_default = [
                '--dataroot', './data', 
                '--name', name,
                '--model', 'test',
                '--netG', 'resnet_9blocks',
                '--direction', 'AtoB',
                '--model', 'test',
                '--dataset_mode', 'single',
                '--gpu_id', '-1',
                '--norm', 'batch',
                '--preprocess', 'none',
                '--checkpoints_dir', './models/segmenter',
                '--input_nc', '1',
                '--output_nc', '1',
            ]

            # override model defaults       
            # TODO : Load config from files
            argsmap = dict()
            argsmap['HCFA02'] = args_default
            argsmap['HCFA33_BILLING'] = args_default
            argsmap['HCFA21'] = args_default

            if key == 'HCFA33_BILLING' or key == 'HCFA21':
            # if  key == 'HCFA21':
                args_default = argsmap[key]
                args_default.append('--norm')
                args_default.append('instance')
                args_default.append('--no_dropout')
                args_default.append('--input_nc')
                args_default.append('3')
                args_default.append('--output_nc')
                args_default.append('3')

        args = args_default
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
        
        return opt, model