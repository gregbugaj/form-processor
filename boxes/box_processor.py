# -*- coding: utf-8 -*-
# Add parent to the search path so we can reference the modules(craft, pix2pix) here without throwing and exception 
import os, sys
from utils.nms import non_max_suppression_fast
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image
from utils.resize_image import resize_image

import time
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

import copy

import cv2
from skimage import io
import numpy as np
import craft.craft_utils
import craft.imgproc
import craft.file_utils
import json
import zipfile

from craft.craft import CRAFT
from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
         
    print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))

    return img.resize((w, h), method)

def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)

def crop_poly_low(img, poly):
    """
        find region using the poly points
        create mask using the poly points
        do mask op to crop
        add white bg
    """
    # points should have 1*x*2  shape
    if len(poly.shape) == 2:
        poly = np.array([np.array(poly).astype(np.int32)])

    pts=poly
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg + dst

    return dst2

def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)   
    return dir
        
def paste_fragment(overlay, fragment, pos=(0,0)):
    # You may need to convert the color.
    fragment = cv2.cvtColor(fragment, cv2.COLOR_BGR2RGB)
    fragment_pil = Image.fromarray(fragment)
    overlay.paste(fragment_pil, pos) 

def find_overlap(box, data):
    overalps=[]
    indexes=[]

    if len(data) == 0:
        return [], []

    x,y,w,h = box
    x1min = x
    x1max = x+w
    y1min = y
    y1max = y+h

    for i, bb in enumerate(data):
        _x,_y,_w,_h = bb
        x2min = _x
        x2max = _x+_w
        y2min = _y
        y2max = _y+_h
        if (x1min<x2max and x2min<x1max and y1min < y2max and y2min < y1max) :
            overalps.append(bb)
            indexes.append(i)

    return np.array(overalps), np.array(indexes)

def line_detection(src):
    """
        Detect lines 
    """
    print(f'Line detection : {src.shape}')
    cv2.imwrite('/home/greg/dev/form-processor/result/lines.png', src)
    # conversion required or we will get 'Failure to use adaptiveThreshold: CV_8UC1 in function adaptiveThreshold'
    src = src.astype('uint8')
    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src
    # Apply adaptiveThreshold at the bitwise_not of gray
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.THRESH_BINARY, 15, -2)
    cv2.imwrite('/home/greg/dev/form-processor/result/detected_lines.png', bw)

    # Create the images that will use to extract the horizontal lines
    thresh = np.copy(bw)
    image = src
    rows = thresh.shape[0]
    verticalsize = rows // 4

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
    detected_lines = cv2.dilate(thresh, kernel)

    # detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)

    cv2.imwrite('/home/greg/dev/form-processor/result/detected_linesXX.png', detected_lines)



def get_prediction(craft_net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size, mag_ratio, refine_net=None):
    net = craft_net
    show_time = True
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = craft.imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio= mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = craft.imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft.craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft.craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft.craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    render_img = score_link
    ret_score_text = craft.imgproc.cvt2HeatmapImg(render_img)
    if show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
    
    # line_detection(score_text * 255)
    return boxes, polys, ret_score_text


class BoxProcessor:
    def __init__(self, work_dir:str = '/tmp/form-segmentation', cuda: bool = False) -> None:
        print("Box processor [cuda={}]".format(cuda))
        self.cuda = cuda
        self.work_dir = work_dir
        self.craft_net, self.refine_net = self.__load()

    def __load(self):
        # load models
        args = Object()
        args.trained_model='./models/craft/craft_mlt_25k.pth'
        args.refiner_model='./models/craft/craft_refiner_CTW1500.pth'

        cuda = self.cuda
        refine = True
        # load net
        net = CRAFT()     # initialize

        print('Loading weights from checkpoint (' + args.trained_model + ')')
        if cuda:
            net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
        else:
            net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

        if cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False        

        net.eval()

        # LinkRefiner
        refine_net = None
        if refine:
            from craft.refinenet import RefineNet
            refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
            if cuda:
                refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
                refine_net = refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

            refine_net.eval()
            args.poly = True

        t = time.time()

        return net, refine_net

    def extract_bounding_boxes(self, id, key, image):
        """
            Extrac bouding boxes for specific image, try to predict line number representin each bouding box. 
            return box array, fragment array, line_number array,  prediciton results
        """
        print('Extracting bounding boxes : {}, {}'.format(id, key))
        try:
            debug_dir = ensure_exists(os.path.join(self.work_dir,id,'bounding_boxes', key, 'debug'))
            crops_dir = ensure_exists(os.path.join(self.work_dir,id,'bounding_boxes', key, 'crop'))
            output_dir = ensure_exists(os.path.join(self.work_dir,id,'bounding_boxes', key, 'output'))

            print(f'debug_dir : {debug_dir}')
            print(f'crops_dir : {crops_dir}')
            print(f'output_dir : {output_dir}')

            image = copy.deepcopy(image)

            bboxes, polys, score_text = get_prediction(
                image=image,
                craft_net=self.craft_net,
                refine_net=None,
                text_threshold=0.7,
                link_threshold=0.4,
                low_text=0.4,
                cuda=self.cuda,
                poly=True,
                canvas_size=1280, 
                mag_ratio=1.5
            )
            
            prediction_result = dict()
            prediction_result['bboxes'] = bboxes
            prediction_result['polys'] = polys
            prediction_result['heatmap'] = score_text
            
            regions = bboxes # prediction_result["boxes"]

            img_h=image.shape[0]
            img_w=image.shape[1]

            # line detection
            all_box_lines = []
            for idx, region in enumerate(regions):
                region = np.array(region).astype(np.int32).reshape((-1))
                region = region.reshape(-1, 2)
                poly = region.reshape((-1, 1, 2))
                box = cv2.boundingRect(poly)
                box = np.array(box).astype(np.int32)
                x,y,w,h = box
                h2 = (h / 2)
                box_line = [0, y+h/3, img_w, h/2]
                box_line = np.array(box_line).astype(np.int32)
                all_box_lines.append(box_line)
                # print(f' >  {cy} : {box} : {box_line}')

            all_box_lines = np.array(all_box_lines)
            y1 = all_box_lines[:,1]
            
            # sort boxes by the  y-coordinate of the bounding box
            idxs = np.argsort(y1)
            lines = []

            while len(idxs) > 0:
                last = len(idxs) - 1
                idx = idxs[last]
                box_line = all_box_lines[idx]
                overlaps, indexes = find_overlap(box_line, all_box_lines)
                overlaps = np.array(overlaps)
                min_x = overlaps[:, 0].min()
                min_y = overlaps[:, 1].min()
                max_w = overlaps[:, 2].max()
                max_h = overlaps[:, 3].max()
                box = [min_x, min_y, max_w, max_h]
                lines.append(box)
                idxs = np.delete(idxs, indexes)
			
            # reverse to get the right order
            lines = np.array(lines)[::-1]
            line_size = len(lines)
            
            print(f'Lines detected : {line_size}')
            result_folder = './result/'
            if not os.path.isdir(result_folder):
                os.mkdir(result_folder)

            # save score text
            filename = id
            mask_file = result_folder + "/res_" + filename + '_mask.jpg'
            cv2.imwrite(mask_file, score_text)

            # deepcopy image so that original is not altered
            image = copy.deepcopy(image)
            pil_image = Image.new('RGB', (image.shape[1], image.shape[0]), color=(0,255,0,0))

            rect_from_poly = []
            rect_line_numbers = []
            fragments = []
            ms = int(time.time() * 1000)

            for idx, region in enumerate(regions):
                print(f'--- Snippet ID {idx}')
                region = np.array(region).astype(np.int32).reshape((-1))
                region = region.reshape(-1, 2)
                poly = region.reshape((-1, 1, 2))

                box = cv2.boundingRect(poly)
                box = np.array(box).astype(np.int32)
                x,y,w,h = box
                snippet = crop_poly_low(image, poly)

                print(f' ** snippet shape : {snippet.shape}')
                # try to figure out line number
                _, line_indexes = find_overlap(box, lines)                
                line_number = -1
                if len(line_indexes) == 1:
                    line_number = line_indexes[0]+1

                fragments.append(snippet)
                rect_from_poly.append(box)
                rect_line_numbers.append(line_number)
                
                # export cropped region
                file_path = os.path.join(crops_dir, "%s_%s.jpg" % (ms, idx))
                cv2.imwrite(file_path, snippet)
                paste_fragment(pil_image, snippet, (x, y))

                # break    
            savepath = os.path.join(debug_dir, "%s.jpg" % ('txt_overlay'))
            pil_image.save(savepath, format='JPEG', subsampling=0, quality=100)

            # if True:
            #     return [], [], [], None
            # we can't return np.array here as t the 'fragments' will throw an error
            # ValueError: could not broadcast input array from shape (42,77,3) into shape (42,)
            return rect_from_poly, fragments, rect_line_numbers, prediction_result
        except Exception as ident:
            raise ident
            print(ident)

        return [], [], [], None

    def process_full_extraction(self,id,image):
        """
            Do full page text extraction
        """
        print('Processing full page extraction: {}'.format(id))

        debug_dir = ensure_exists(os.path.join(self.work_dir,id,'boxes_full'))
        crops_dir = ensure_exists(os.path.join(self.work_dir,id,'crops_full'))

        # # read image
        # image = read_image(image)
        h=image.shape[0]
        w=image.shape[1]
        image=copy.deepcopy(image)

        # perform prediction
        bboxes, polys, score_text = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.7,
            link_threshold=0.3,
            low_text=0.4,
            cuda=self.cuda,
            poly=True,
            canvas_size=1280, 
            mag_ratio=1.5
        )
        
        prediction_result = dict()
        prediction_result['bboxes'] = bboxes
        prediction_result['polys'] = polys
        prediction_result['heatmap'] = score_text
        
        # deepcopy image so that original is not altered
        image = copy.deepcopy(image)
        regions = bboxes

        # convert imaget to BGR color
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        file_path = os.path.join(debug_dir, "src.png")
        cv2.imwrite(file_path, image)
        
        pil_image = Image.new('RGB', (image.shape[1], image.shape[0]), color=(255,255,255,0))

        rect_from_poly=[]
        fragments=[]
        for i, region in enumerate(regions):
            region = np.array(region).astype(np.int32).reshape((-1))
            region = region.reshape(-1, 2)
            poly = region.reshape((-1, 1, 2))

            rect = cv2.boundingRect(poly)
            rect = np.array(rect).astype(np.int32)
            x,y,w,h = rect

            # if h < 15:
            #     continue
            
            snippet = crop_poly_low(image, poly)
            fragments.append(snippet)
            rect_from_poly.append(rect)
            # export cropped region
            file_path = os.path.join(crops_dir, "%s.jpg" % (i))
            cv2.imwrite(file_path, snippet)
            paste_fragment(pil_image, snippet, (x, y))

        savepath = os.path.join(debug_dir, "%s.jpg" % ('txt_overlay'))
        pil_image.save(savepath, format='JPEG', subsampling=0, quality=100)

        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return np.array(rect_from_poly), np.array(fragments), cv_img, prediction_result 

class Object(object):
    pass 

if __name__ == '__main__':

    img_path='/home/greg/dev/form-processor/craft-test/padded_snippet-HCFA02.jpg'
    image = cv2.imread(img_path)

    boxer = BoxProcessor(work_dir='/tmp/form-segmentation')
    boxer.extract_bounding_boxes('test', 'key', image)
    
    os.exit()

    args = Object()
    args.cuda=False
    args.refine = False

    args.trained_model='./models/craft/craft_mlt_25k.pth'
    args.refiner_model='./models/craft/craft_refiner_CTW1500.pth'
    args.test_folder='./craft-test'
    args.mag_ratio= 1.5
    args.canvas_size= 1280
    args.text_threshold= 0.7
    args.low_text= 0.4
    args.link_threshold= 0.4
    args.show_time= True
    args.poly= True

    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False        

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from craft.refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()


    """ For test images in a folder """
    image_list, _, _ = craft.file_utils.get_files(args.test_folder)

    result_folder = './result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = craft.imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        craft.file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))