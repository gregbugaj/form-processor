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

    def process(self, key, img_path, snippet)->None:
        """
            Process data field
        """
        print("Processing field : {}".format(key))
        opt, model = self.__setup(key)
   
        name = img_path.split('/')[-1]
        work_dir = os.path.join(self.work_dir, name, 'fields', key)
        debug_dir = os.path.join(self.work_dir, name, 'fields_debug', key)
       
        ensure_exists(work_dir)
        ensure_exists(debug_dir)

        opt.dataroot = work_dir
        name = 'segmenation'

        image_name = '%s.png' % (key)
        save_path = os.path.join(work_dir, image_name)                   
        imwrite(save_path, snippet)

        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

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

    def __setup(self, key):
        """
            Model setup
        """
        name = self.models[key]

        args = [
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
        ]

        opt = TestOptions().parse(args)  # get test options
        # hard-code parameters for test
        opt.eval = False   # test code only supports num_threads = 0
        opt.num_threads = 0   
        opt.batch_size = 1    # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

        print(opt)
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        return opt, model