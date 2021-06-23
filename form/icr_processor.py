# Add parent to the search path so we can reference the modules(craft, pix2pix) here without throwing and exception 
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


import numpy as np
import copy
import cv2
import numpy as np
from PIL import Image

import os

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from icr.utils import CTCLabelConverter, AttnLabelConverter, Averager
from icr.dataset import hierarchical_dataset, AlignCollate
from icr.model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils.resize_image import resize_image

def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)

def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)   
    return dir

class IcrProcessor:
    def __init__(self,work_dir:str = '/tmp/form-segmentation', cuda: bool = False) -> None:
        print("ICR processor [cuda={}]".format(cuda))
        self.cuda = cuda
        self.work_dir = work_dir
        self.__load()

    def __load(self):
        
        return None

    def process(self,id,key,image):
        """
            Process image via ICR
        """
        print('ICR processing : {}, {}'.format(id, key))

        debug_dir =  ensure_exists(os.path.join(self.work_dir,id,'icr', key, 'debug'))
        output_dir = ensure_exists(os.path.join(self.work_dir,id,'icr', key, 'output'))

        