import os, sys
import types
import typing

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

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='omr.log',
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

class OpticalMarkRecognition:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.opt, self.model_checked, self.model_unchecked = self.__setup()

    def __load_model(self, model_name:str):

        DEVICE = 'cpu' # this will blowup if we use cuda with less than 8GB of mem for 1024x2056
        model_path = os.path.join('./models/omr', model_name)

        print(f'Loading model : {model_path}')
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
        
        return model

    def __setup(self):
        """
            Model setup
        """
        print('Loading OMR model')
        opt = Object()

        model_checked = self.__load_model('best_model-checked.pth')
        model_unchecked = self.__load_model('best_model-unchecked.pth')

        return opt, model_checked, model_unchecked

    def __process_smp(self, key:str, kid:str, image, model) -> None:
        """processing via SMP model"""        
        debug_dir = ensure_exists(os.path.join(self.work_dir, kid, 'omr'))

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

    def __segment(self, image, mask, color) -> typing.List[typing.Any]:
        """Segment image"""
        start = time.time()
        segment = None

        results = []
        # Extract ROI
        (cnts, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (cnts, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        debug = False

        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)
            # print(f'approx = {len(approx)}')
            # allow for some holes 

            if len(approx) >= 4 and len(approx) <= 12:
                # compute the bounding box of the approximated contour and use the bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                area = cv2.contourArea(c)
                hull_area = cv2.contourArea(cv2.convexHull(c))
                solidity = area / float(hull_area)

                if debug:
                    print(f'area : {area}')
                    print(f'solidity : {solidity}')
                    print(f'aspect_ratio : {aspect_ratio}')

                if solidity > .90 and (aspect_ratio >= 0.8 and aspect_ratio <= 1.2):
                    # cv2.drawContours(image, [approx], -1, (0, 0, 255), 1)
                    box = cv2.boundingRect(c)
                    x, y, w, h = box
                    results.append(box)
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

        if debug:
            showAndDestroy('Extracted Segment', image)

        dt = time.time() - start
        logging.info('Eval time %.3f sec' % dt)

        return results

    def __extract(self, kid, image, model):
        # FIXME : Size is hardcoded here this needs to be parametized
        size = image.shape[:2]
        desired_size = (1536, 1024) # HxW

        image_src = cv2.resize(image, (desired_size[1], desired_size[0]), interpolation = cv2.INTER_AREA)
        mask = self.__process_smp(kid, kid, image_src, model)
        mask = 255 - mask

        mask = cv2.resize(mask, (size[1], size[0]), interpolation = cv2.INTER_AREA)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        return mask


    def find_marks(self, kid:str, image) -> np.ndarray:
        """Extract and process checkmarks"""
        print(f'OMR kid# : {kid}')
        if isinstance(image, str):
            image = cv2.imread(image)
        debug_dir = ensure_exists(os.path.join(self.work_dir, kid, 'omr'))
        
        mask_checked = self.__extract(kid, image, self.model_checked)
        mask_unchecked = self.__extract(kid, image, self.model_unchecked)

        if True:
            cv2.imwrite(os.path.join(debug_dir, 'mask_checked.png'), mask_checked)
            cv2.imwrite(os.path.join(debug_dir, 'mask_unchecked.png'), mask_unchecked)

        def preprocess(img):
            #  expand shape from 1 channel to 3 channel
            # if len(img.shape) == 2:
            #     img = img[:, :, None] * np.ones(3, dtype=int)[None, None, :]
            if False:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_bw = img # 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')
                # img_bw = cv2.adaptiveThreshold(img, 255, cv2.THRESH_BINARY, 15, -2)
                # (T, img_bw) = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

                # Define the structuring elements
                se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
                se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

                # Perform closing then opening
                mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

                mask = np.dstack([mask, mask, mask]) / 255
                out = img * mask

                cv2.imshow('Output', out)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imwrite('output.png', out)

            # Creating kernel
            kernel = np.ones((5, 5), np.uint8)
            errode_img = cv2.erode(img, kernel) 

            kernel = np.ones((2, 2), np.uint8)
            dilate_img = cv2.dilate(errode_img, kernel, iterations=1)

            return dilate_img

        mask_checked = preprocess(mask_checked)    
        mask_unchecked = preprocess(mask_unchecked)    

        cv2.imwrite(os.path.join(debug_dir, 'mask_checked_preprocess.png'), mask_checked)
        cv2.imwrite(os.path.join(debug_dir, 'mask_unchecked_preprocess.png'), mask_unchecked)

        results = dict()
        results['checked'] =  self.__segment(image, mask_checked, color = (0, 255, 0))
        results['unchecked'] =  self.__segment(image, mask_unchecked, color = (0, 0, 255))
        
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(debug_dir, f'marked_{kid}.png'), image)

        return results
