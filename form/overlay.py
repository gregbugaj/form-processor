# Add parent to the search path so we can reference the module here without throwing and exception
from PIL import Image
from utils.utils import ensure_exists
from utils.image_utils import imwrite, read_image, viewImage
from pix2pix.util.util import tensor2im
from pix2pix.util.visualizer import save_images
from pix2pix.models import create_model
from pix2pix.data import create_dataset
from pix2pix.options.test_options import TestOptions
import time
from shutil import copy, copyfile
import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir))


class FormOverlay:
    def __init__(self, work_dir, cuda: bool = False):
        print("Form overlay processor [cuda={}]".format(cuda))
        self.work_dir = work_dir
        self.opt, self.model = self.__setup()

    def __setup(self):
        """
            Model setup
        """
        args = [
            '--dataroot', './data',
            '--name', 'hicfa_mask_global',
            '--model', 'test',
            '--netG', 'global',
            '--direction', 'AtoB',
            '--model', 'test',
            '--dataset_mode', 'single',
            '--gpu_id', '-1',
            '--norm', 'instance',
            '--preprocess', 'none',
            '--checkpoints_dir', './models/overlay'
        ]

        opt = TestOptions().parse(args)
        # hard-code parameters for test
        opt.eval = False  
        opt.num_threads = 0  # test code only supports num_threads = 0
        opt.batch_size = 1  # test code only supports batch_size = 1
        opt.serial_batches = True
        opt.no_flip = True
        opt.no_dropout = True
        opt.display_id = -1

        model = create_model(opt)
        model.setup(opt)

        print('Model setup complete')
        return opt, model

    def __extract_segmenation_mask(self, img, dataroot_dir, work_dir, debug_dir):
        """
            Extract overlay segmentation mask for the image
        """
        model = self.model
        opt = self.opt
        opt.dataroot = dataroot_dir
        image_dir = work_dir
        name = 'overlay'

        # create a dataset given opt.dataset_mode and other options
        dataset = create_dataset(opt)
        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            # Debug
            if False:
                for label, im_data in visuals.items():
                    image_numpy = tensor2im(im_data)
                    # Tensor is in RGB format OpenCV requires BGR
                    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
                    image_name = '%s_%s.png' % (name, label)
                    save_path = os.path.join(debug_dir, image_name)
                    imwrite(save_path, image_numpy)
                    viewImage(image_numpy, 'Tensor Image')

            label = 'overlay'
            fake_im_data = visuals['fake']
            image_numpy = tensor2im(fake_im_data)
            # Tensor is in RGB format OpenCV requires BGR
            image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(image_dir, '%s_%s.png' % (name, label))

            # testing only
            imwrite(save_path, image_numpy)
            # viewImage(image_numpy, 'Prediction image')

            return image_numpy

    def blend_to_text(self, real_img, fake_img):
        """
            Blend real and fake(generated) images together to generate extracted text mask
        """
        real = read_image(real_img)
        fake = read_image(fake_img)

        print('Image shapes ***')
        # Sizes of input arguments do not match
        print(real.shape)
        print(fake.shape)

        if real.shape != fake.shape:
            raise Exception(f'Sizes of input arguments do not match(real, fake) : {real.shape} != {fake.shape}')

        blended_img = cv2.bitwise_or(real, fake)
        blended_img[blended_img >= 120] = [255]

        return blended_img

    def segment(self, documentId: str, img_path: str):
        """
            Form overlay segmentation 
        """
        if not os.path.exists(img_path):
            raise Exception('File not found : {}'.format(img_path))

        name = documentId
        work_dir = os.path.join(self.work_dir, name, 'work')
        debug_dir = os.path.join(self.work_dir, name, 'debug')
        dataroot_dir = os.path.join(self.work_dir, name, 'dataroot_overlay')

        ensure_exists(self.work_dir)
        ensure_exists(work_dir)
        ensure_exists(debug_dir)
        ensure_exists(dataroot_dir)

        dst_file_name = os.path.join(dataroot_dir, f'overlay_{name}.png')
        if not os.path.exists(dst_file_name):
            copyfile(img_path, dst_file_name)

        img = cv2.imread(dst_file_name)
        # viewImage(img, 'Source Image')

        segmask = self.__extract_segmenation_mask(
            img, dataroot_dir, work_dir, debug_dir)

        # Unable to segment return empty mask
        if np.array(segmask).size == 0:
            print('Unable to segment image')
            return None, None

        blended = self.blend_to_text(img, segmask)
        # viewImage(segmask, 'segmask')
        tm = time.time_ns()
        imwrite(os.path.join(debug_dir, 'overlay_{}.png'.format(tm)), segmask)

        # real, fake, blended
        return img, segmask, blended
