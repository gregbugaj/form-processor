
# Add parent to the search path so we can reference the module here without throwing and exception 
from logging import Handler, raiseExceptions
import os, sys

from numpy.core.fromnumeric import shape
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch
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

import segmentation_models_pytorch as smp
import albumentations as albu

class Object(object):
    pass 


 # helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        print(f'shape : {image.shape}')
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


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

    def __process_pix2pix(self, key:str, snippet, opt, model, config) -> None:
        """process pix2pix form cleanup"""
        name = 'segmenation'
        debug_dir = ensure_exists(os.path.join(self.work_dir, id, 'fields_debug', key))
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
            image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)

            return image_numpy

        return None    
    
    def __process_smp(self, key:str, snippet, opt, model, config) -> None:
        """processing via SMP model"""        

        def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)): 
            ''' 
            Converts a torch Tensor into an image Numpy array 
            Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order 
            Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default) 
            ''' 
            tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp 
            tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1] 

            return tensor.numpy().astype(out_type)

        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype('float32')

        def get_preprocessing(preprocessing_fn):
            """Construct preprocessing transform
            
            Args:
                preprocessing_fn (callbale): data normalization function 
                    (can be specific for each pretrained neural network)
            Return:
                transform: albumentations.Compose
            """
            
            _transform = [
                albu.Lambda(image=preprocessing_fn),
                albu.Lambda(image=to_tensor),
            ]
            return albu.Compose(_transform)

        # load best saved checkpoint
        ENCODER = 'resnet34'
        ENCODER_WEIGHTS = 'imagenet'
        DEVICE = 'cpu' #cuda

        img_path = '/tmp/segmentaion-mask/pr_mask-snippet.png'
        cv2.imwrite(img_path, snippet)
        
        # print(f' ********** snippet shape : {snippet.shape}')
        # create segmentation model with pretrained encoder
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        preprocessing = get_preprocessing(preprocessing_fn)
        sample = preprocessing(image=snippet)
        image = sample['image']

        # print(f' ********** image shape : {image.shape}')
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = model.predict(x_tensor)        
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        # pr_mask = tensor2img(pr_mask) # normalized 
        pr_mask = 255-pr_mask*255 # convert 0...1 range into 0...255 range
        img_path = '/tmp/segmentaion-mask/pr_mask-01.png'
        cv2.imwrite(img_path, pr_mask)
        pr_mask = np.array(pr_mask).astype(np.uint8)

        # w = pr_mask.shape[0]
        # h = pr_mask.shape[1]
        # debug_img = get_debug_image(h, w, image_src, pr_mask)
        
        pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_RGB2BGR)
        img_path = '/tmp/segmentaion-mask/pr_mask-02.png'
        cv2.imwrite(img_path, pr_mask)
        visualize(image=snippet,predicted=pr_mask)

        return pr_mask

    def process(self, id, key, snippet)->None:
        """
            Process data field
        """
        print("Processing field : {}".format(key))
        opt, model, config = self.__setup(key)
        work_dir = ensure_exists(os.path.join(self.work_dir, id, 'fields', key))
        opt.dataroot = work_dir

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

        image_name = '%s.png' % (key)
        save_path = os.path.join(work_dir, image_name)                   
        imwrite(save_path, snippet)

        # TODO : Add postprocessing
        arch = config['arch']
        if arch == 'pix2pix':
            image_numpy = self.__process_pix2pix(key, snippet, opt, model, config)
        elif arch == 'smp':
            image_numpy = self.__process_smp(key, snippet, opt, model, config)

        name = 'segmenation'
        label = 'real'
        debug_dir = ensure_exists(os.path.join(self.work_dir, id, 'fields_debug', key))

        # Tensor is in RGB format OpenCV requires BGR
        # image_numpy = cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(debug_dir, image_name)
                       
        imwrite(save_path, image_numpy)

        return image_numpy

    def __setup(self, key):
        """
            Model setup
        """
        # TODO : Store models in cache so we don't have to reinitialize it

        name = self.models[key]
        config_file = os.path.join('./models/segmenter', name, 'config.json')

        if not os.path.exists(config_file):
            raise Exception(f'Config file not found : {config_file}')

        with open(config_file) as f:
            data = json.load(f)

        args = data['args']
        # arch can be [pix2pix:default, smp]
        arch = 'pix2pix'
        if 'arch' in data:
            arch = data['arch']
        else:
            data['arch'] = arch
        arch = arch.lower()
        
        if arch == 'pix2pix':
            print('Loading PIX2PIX model')
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
        elif arch == 'smp':
            print('Loading SMP model')
            DEVICE = 'cpu' # this will blowup if we use cuda with less than 8GB of mem for 1024x2056
            opt = Object()

            if 'model' in data:
                model_path = os.path.join('./models/segmenter', name, data['model'])
            else:
                model_path = os.path.join('./models/segmenter', name, 'model.pth')

            print(f'Laoding model : {model_path}')
            if not os.path.exists(model_path):
                raise Exception(f'File not found : {model_path}')

            # load best saved checkpoint
            checkpoint = torch.load(model_path)
            
            #Model can be saved directly as whole model or with state as {net, acc, epoch}
            if isinstance(checkpoint, dict):
                net = checkpoint['net']
                # model.load_state_dict(net)
                raise Exception('Not yet implemented')
            else:
                model = checkpoint.to(DEVICE)
            
            if isinstance(model, torch.nn.DataParallel):
                model = model.module #This is required as we are wrapping the network in DataParallel if we are using CUDA 
            
            # model.eval() 

            return opt, model, data            

        raise Exception(f'Unknown architecture : {arch}')