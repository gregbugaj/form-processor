import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
import numpy as np
import segmentation_models_pytorch as smp


from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
from tqdm import tqdm

from dataset import Dataset

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['form']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'
DEVICE = 'cpu'

# pixel_shuffle = torch.nn.PixelShuffle(2)
# input = torch.randn(1, 4, 8, 8)

# print(input)
# output = pixel_shuffle(input)
# print(output.size())
# os.exit(1)


# load best saved checkpoint
# checkpoint = torch.load('../smp/best_model@0.9914567287977759.pth', map_location=torch.device(DEVICE))
checkpoint = torch.load('../smp/best_model.pth', map_location=torch.device(DEVICE))

# checkpoint = torch.load('/home/greg/dev/form-processor/smp/best_model.pth')
best_model = checkpoint.to(DEVICE)
best_model = best_model.module # This is required as we are wrapping th network in DataParallel
best_model.eval()

# create segmentation model with pretrained encoder
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# DATA_DIR = '/home/greg/dev/unet-denoiser/data-HCFA02-TEST'
DATA_DIR = '/home/greg/dev/unet-denoiser/data-diagnosis_code-TEST'
DATA_DIR = '/home/greg/dev/pytorch-CycleGAN-and-pix2pix/datasets/diagnosis_code/eval'
DATA_DIR = '/home/greg/dev/unet-denoiser/data_HCFA21/'
DATA_DIR = '/home/greg/dev/unet-denoiser/data/'
DATA_DIR = '/tmp/form-segmentation/fields/HCFA21'
DATA_DIR = '/home/greg/HCFA21'
# DATA_DIR = '/home/greg/dev/unet-denoiser/data'

x_test_dir = os.path.join(DATA_DIR, 'train/image')
y_test_dir = os.path.join(DATA_DIR, 'train/mask') 

DATA_DIR = '/home/greg/dataset/data-hipa/forms/splitted/test'
x_test_dir = os.path.join(DATA_DIR, 'image')
y_test_dir = os.path.join(DATA_DIR, 'mask')    

DATA_DIR = '/home/gbugaj/data/training/optical-mark-recognition/hicfa/task_checkboxes-2021_10_18_16_09_24-cvat_for_images_1.1/output_split/test'
# DATA_DIR = '/home/greg/dataset/cvat/task_checkboxes_2021_10_18/output_split/test'
DATA_DIR = '/home/gbugaj/devio/unet-denoiser/data_hicfa_mask/test'
DATA_DIR = '/home/greg/dev/unet-denoiser/data_hicfa_text/test'
# DATA_DIR = '/home/greg/dev/pytorch-CycleGAN-and-pix2pix/datasets/hicfa_mask'
x_test_dir = os.path.join(DATA_DIR, 'image')
y_test_dir = os.path.join(DATA_DIR, 'mask')    



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


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(256, 1024)
        # albu.PadIfNeeded(min_height=1024, min_width=768)
        albu.PadIfNeeded(min_height=2496, min_width=1792) 
    ]
    return albu.Compose(test_transform)

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
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)): 
    ''' 
    Converts a torch Tensor into an image Numpy array 
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order 
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default) 
    ''' 
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp 
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1] 

# size=(1792, 2494)
# create test dataset
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
    size=(1792, 2496)
)

test_dataloader = DataLoader(test_dataset)

# (1024, 160) HICFA02
# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir, 
    classes=CLASSES,
    size=(1792, 2496)
)
    
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




for i in tqdm(range(len(test_dataset))):
    n = np.random.choice(len(test_dataset))
    n = i
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]

    gt_mask = gt_mask.squeeze()
    print('Shapes ***')    
    print(image.shape)

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    
    with torch.no_grad():
        pr_mask = best_model(x_tensor)
    # pr_mask = best_model.predict(x_tensor)
    
    continue
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
 
    # pr_mask = 255-pr_mask*255
    # gt_mask = 255-gt_mask*255

    pr_mask = pr_mask*255
    gt_mask = 255-gt_mask*255

    w = gt_mask.shape[1]
    h = gt_mask.shape[0]

    debug_img = get_debug_image(h, w, image_vis, pr_mask)
    
    cv2.imwrite('/tmp/segmentation-mask/{}_debug.png'.format(i), debug_img)
    cv2.imwrite('/tmp/segmentation-mask/{}_src.png'.format(i), image_vis)
    cv2.imwrite('/tmp/segmentation-mask/{}_mask.png'.format(i), pr_mask)
    

