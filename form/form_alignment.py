import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch
import cv2
import time
 
from utils.utils import current_milli_time, ensure_exists
from utils.image_utils import imwrite

from PIL import Image
import numpy as np

import segmentation_models_pytorch as smp
import albumentations as albu

from utils.resize_image import resize_image
from utils.visualize import visualize

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='alignment.log',
                    level=logging.INFO)

class Object(object):
    pass 

def showAndDestroy(label, image):
    cv2.imshow(label, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_best_point(point, point_candidates, max_distance=sys.maxsize):
    """Find best point by performing L2 norm (eucledian distance) calculation"""
    best = None
    for p in point_candidates:
        dist = np.linalg.norm(point - p)
        if dist <= max_distance:
            max_distance = dist
            best = p
    return max_distance, best

class FormAlignment:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.opt, self.model = self.__setup()

    def __setup(self):
        """
            Model setup
        """
        print('Loading SMP-Orientation model')

        DEVICE = 'cpu' # this will blowup if we use cuda with less than 8GB of mem for 1024x2056
        opt = Object()
        model_path = os.path.join('./models/orientation', 'best_model.pth')

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
        
        return opt, model

    def __process_smp(self, key:str, id:str, image, model) -> None:
        """processing via SMP model"""        
        debug_dir = ensure_exists(os.path.join(self.work_dir, id, 'orientation'))

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
        # FIXME : using 'cuda' causes "RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same"

        # create segmentation model with pretrained encoder
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        preprocessing = get_preprocessing(preprocessing_fn)
        sample = preprocessing(image=image)
        image = sample['image']

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = model.predict(x_tensor)        
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        # pr_mask = tensor2img(pr_mask) # normalized 
        pr_mask = 255-pr_mask*255 # convert 0...1 range into 0...255 range
        pr_mask = np.array(pr_mask).astype(np.uint8)
        pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_RGB2BGR)
        
        return pr_mask        

    def __segment(self, image, mask) -> np.ndarray:
        """Segment image"""
        start = time.time()
        segment = None

        # Extract ROI
        (cnts, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug = False

        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)
            print(f'approx = {len(approx)}')
            # allow for some holes 
            if len(approx) >= 4 and len(approx) <= 8:
                # compute the bounding box of the approximated contour and use the bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                area = cv2.contourArea(c)
                hull_area = cv2.contourArea(cv2.convexHull(c))
                solidity = area / float(hull_area)

                print(f'area : {area}')
                print(f'solidity : {solidity}')
                print(f'aspect_ratio : {aspect_ratio}')
                
                if area > 50000 and solidity > .95 and (aspect_ratio >= 0.7 and aspect_ratio <= 1.4):
                    # cv2.drawContours(image, [approx], -1, (0, 0, 255), 1)
                    # Keypoint order : (top-left, top-right, bottom-right, bottom-left)                
                    # Rearange points in in order to get correct perspective change
                    res = approx.reshape(-1, 2)
                    x, y, w, h = cv2.boundingRect(c)
                    _, top_left = find_best_point([y, x], res)
                    _, top_right = find_best_point([y, x + w], res)
                    _, bottom_right = find_best_point([y + h, x + w], res)
                    _, bottom_left = find_best_point([y + h, x], res)
                    # Expected size of the new image
                    width = w
                    height = h
                    cols = w
                    rows = h

                    src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
                    dst_pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    segment = cv2.warpPerspective(image, M, (cols, rows), flags=cv2.INTER_LINEAR)

                    if debug:
                        showAndDestroy('Extracted Segment', segment)

        dt = time.time() - start
        logging.info('Eval time %.3f sec' % dt)

        return segment

    def align(self, id:str, image) -> np.ndarray:
        """Extract and perfrom for alignment"""
        print(f'Aligning form id# : {id}')
        if isinstance(image, str):
            image = cv2.imread(image)
        debug_dir = ensure_exists(os.path.join(self.work_dir, id, 'orientation'))
        size = image.shape[:2]
        desired_size = (512, 512) # HxW

        image_src = cv2.resize(image, (desired_size[1], desired_size[0]), interpolation = cv2.INTER_AREA)
        mask = self.__process_smp(id, id, image_src, self.model)
        mask = 255 - mask
        mask = cv2.resize(mask, (size[1], size[0]), interpolation = cv2.INTER_AREA)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        segment = self.__segment(image, mask)

        if True:
            cv2.imwrite(os.path.join(debug_dir, 'adj_mask.png'), mask)
            cv2.imwrite(os.path.join(debug_dir, 'segment.png'), segment)

        return segment
