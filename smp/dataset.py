import os
import cv2
import matplotlib.pyplot as plt

import torch
import numpy as np
import segmentation_models_pytorch as smp

from torch.utils.data import Dataset as BaseDataset

def resize_image(image, desired_size, color=(255, 255, 255)):
    ''' Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------
    
    image: np.array
        The image to be resized.

    desired_size: (int, int)
        The (height, width) of the resized image

    Return
    ------

    image: np.array
        The image of size = desired_size
    '''

    size = image.shape[:2]
    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0])/size[0]
        ratio_h = float(desired_size[1])/size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x*ratio) for x in size])
        image = cv2.resize(image, (new_size[1], new_size[0]), interpolation = cv2.INTER_AREA)
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image

def get_debug_image(h, w, img, mask):
    #  expand shape from 1 channel to 3 channel
    if len(img.shape) == 2:
        img = img[:, :, None] * np.ones(3, dtype=int)[None, None, :]

    if len(mask.shape) == 2:
        mask = mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]

    debug_img = np.ones((2*h, w, 3), dtype = np.uint8) * 255

    debug_img[0:h, :] = img
    debug_img[h:2*h, :] = mask
    cv2.line(debug_img, (0, h), (debug_img.shape[1], h), (255, 0, 0), 1)
    return debug_img

class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            size = None
    ):
        self.ids_a = os.listdir(images_dir)
        self.ids_b = os.listdir(masks_dir)

        # not very good but for not it works
        self.ids_a.sort()
        self.ids_b.sort()

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids_a]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids_b]
        
        # convert str names to class values on masks
        # self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
        # size (w, h)
        if size == None:
            raise Exception('Invalid size')
        self.size = size
    
    def __getitem__(self, i):
        # print(f'self.images_fps[i] = {self.images_fps[i]}')
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        # cv2.imwrite('/tmp/segmentation-mask/%s.png' % (i), mask)

        def get_size(load_size, size):
            w, h = size
            new_w = load_size
            new_h = load_size * h // w
            return new_w, new_h
        
        size = self.size
        long_side = size[0]
        h = size[1]
        w  = long_side

        if False and np.random.choice([0, 0], p = [0.5, 0.5]) :
            size = (image.shape[1], image.shape[0]) # w,h
            new_size = get_size(long_side, size)

            image_resized = cv2.resize(image, (new_size[0], new_size[1]), interpolation = cv2.INTER_AREA)
            maks_resized = cv2.resize(mask, (new_size[0], new_size[1]), interpolation = cv2.INTER_AREA)
            # print(f'param : {size}  > {new_size}')
            image = resize_image(image_resized, (h, w))
            mask = resize_image(maks_resized, (h, w))
    
        image = resize_image(image, (h, w))
        mask = resize_image(mask, (h, w), color=(255, 255, 255))

        if True:
            # FIXME : Causes bolded artifacts
            # stacking will cause unwanted artifacts, as we only deal with single channel this is OK for now        
            # extract certain classes from mask (e.g. background)
            masks = [(mask == 255)]

            # dbg = get_debug_image(512,1536, mask, image)
            # cv2.imwrite(f'/tmp/mask/{i}.png', dbg)
            # cv2.imwrite(f'/tmp/mask/image_raw_{i}.png', image)
            # cv2.imwrite(f'/tmp/mask/mask_raw_{i}.png', mask)
            mask = np.stack(masks, axis=-1).astype('float')
            # cv2.imwrite(f'/tmp/mask/mask_stacked_{i}.png', mask*255)
        else:  # FIXME : THIS CAUSES THE LOSS TO BE NAN
            masks = [mask]
            mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # cv2.imwrite(f'/tmp/mask/image_augmentation_{i}.png', image)
        # cv2.imwrite(f'/tmp/mask/mask_augmentation_{i}.png', mask * 255)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # cv2.imwrite(f'/tmp/mask/image_preprocessing_{i}.png', image)
        # cv2.imwrite(f'/tmp/mask/mask_preprocessing{i}.png', mask * 255)

        # print('image / mask')
        # print(image.shape)
        # print(mask.shape)

        # cv2.imwrite(f'/tmp/mask/image_{i}.png', image)
        # cv2.imwrite(f'/tmp/mask/mask_{i}.png', mask * 255)
        return image, mask

    def __len__(self):
        return len(self.ids_a) # // 4
