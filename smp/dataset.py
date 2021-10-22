import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from codec import one_hot_encode

class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            class_rgb_values=None,
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

        self.class_rgb_values = class_rgb_values


    def __getitem__(self, i):
        print(f'self.images_fps[i] = {self.images_fps[i]}')

        # read images and masks
        # image = cv2.cvtColor(cv2.imread(self.images_fps[i]), cv2.COLOR_BGR2RGB)
        # mask = cv2.cvtColor(cv2.imread(self.masks_fps[i]), cv2.COLOR_BGR2RGB)

        # read images and masks
        image = cv2.imread(self.images_fps[i])[:,:,::-1]
        mask = cv2.imread(self.masks_fps[i])[:,:,::-1]

        cv2.imwrite(f'/tmp/mask/mask_stacked_{i}.png', mask)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        # cv2.imwrite(f'/tmp/mask/mask_stacked_{i}.png', mask)

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
