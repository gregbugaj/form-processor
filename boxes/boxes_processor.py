import os
import numpy as np
import cv2

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw

from utils.resize_image import resize_image

class BoxProcessor:
    def __init__(self) -> None:
        print("Box processor")
        self.net, self.encoder = self.__load()

    def __load(self):
        tune_from = './models/text_detector/best_segmenter.pth'
        nms_thresh = 0.1
        cls_thresh = 0.4
        # -input_size=1280 --nms_thresh=0.1 --cls_thresh=0.4
        net = RetinaNet()
        net = net.cuda()

        # load checkpoint
        checkpoint = torch.load(tune_from)

        net.load_state_dict(checkpoint['net'])
        net.eval()
        
        encoder = DataEncoder(cls_thresh,nms_thresh)
        return net, encoder

    def process(self, snippet):
        print('Processing')
        
        print(self.net)
        encoder = self.encoder
        net = self.net

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])

        print('Loading image...')
        # 1280
        snippet = resize_image(snippet, (1024, 1024))
        img = Image.fromarray(snippet)
        shape = snippet.shape
        print(shape)
        w = shape[1]
        h = shape[0]
        img = img.resize((w,h))
        # w = shape[1]
        # h = shape[0]

        print(img)
        print('Predicting..')

        x = transform(img)
        x = x.unsqueeze(0)
        x = Variable(x)
        x = x.cuda()

        loc_preds, cls_preds = net(x)

        print('Decoding..')
        print(loc_preds)
        print(cls_preds)

        boxes, labels, scores = encoder.decode(loc_preds.data.squeeze(0), cls_preds.data.squeeze(0), (w,h))
        draw = ImageDraw.Draw(img)
       
        print(boxes)
        print(labels)
        print(scores)

        # boxes = boxes.data.numpy()
        # boxes = boxes.data.numpy()
        boxes = boxes.reshape(-1, 4, 2)

        for box in boxes:
            draw.polygon(np.expand_dims(box,0), outline=(0,255,0))
        # img.save("/tmp/form-segmentation/box.png")

        cv_snip = np.array(img)                
        snippet = cv2.cvtColor(cv_snip, cv2.COLOR_RGB2BGR)# convert RGB to BGR
        return snippet

