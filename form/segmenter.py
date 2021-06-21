import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
import numpy as np
from shutil import copyfile
import time


# Add parent to the search path so we can reference the module here without throwing and exception 
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from utils.nms import nms, non_max_suppression_fast

from field_processor import FieldProcessor

from pix2pix.options.test_options import TestOptions
from pix2pix.data import create_dataset
from pix2pix.models import create_model
from pix2pix.util.visualizer import save_images
from pix2pix.util.util import tensor2im
 

from boxes.boxes_processor import BoxProcessor


# Don't change the order here as the field dictionary depends on it
hsv_color_ranges = [
            [[55, 58, 0], [86, 255, 255]],      # GREEN DARK  7fd99d
            [[123, 99, 206], [140, 255, 255]],  # Purple      a96df8
            [[0, 152, 240], [9, 255, 255]],     # Red         ff614e
            [[97, 188, 0], [179, 255, 178]],    # Blue        016aa4
            [[0, 220, 221 ], [30, 255, 255]],   # YELLOW      99624a
            [[0, 0, 120], [19, 255, 167]],      # Brown       99624a (hMin = 0 , sMin = 0, vMin = 120), (hMax = 19 , sMax = 255, vMax = 167)
            [[38, 154, 188], [51, 255, 255]],   # GREEN
            [[45, 187, 207], [179, 255, 255]],  # PINK
            [[0, 230, 160], [21, 255, 255]],    # ORANGE     (hMin = 0 , sMin = 230, vMin = 160), (hMax = 21 , sMax = 255, vMax = 255)
        ]

def viewImage(image, name='Display'):
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
def apply_filter(src_img):
    img = src_img.copy()
    blur = cv2.blur(img,(5,5))
    blur0= cv2.medianBlur(blur,5)
    blur1= cv2.GaussianBlur(blur0,(5,5),0)
    blur2= cv2.bilateralFilter(blur1,9,75,75)
    return blur2

def rgb_2_hsv(r,g,b):
    ## getting green HSV color representation
    col = np.uint8([[[r,g,b]]])
    hsv_col = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
    print(hsv_col)
    return hsv_col

def fixHSVRange(h, s, v):
    """
        Different applications use different scales for HSV. 
        For example gimp uses H = 0-360, S = 0-100 and V = 0-100. But OpenCV uses H: 0-179, S: 0-255, V: 0-255. 
    """
    # Normal H,S,V: (0-360,0-100%,0-100%)
    # OpenCV H,S,V: (0-180,0-255 ,0-255)
    return (180 * h / 360, 255 * s / 100, 255 * v / 100)

def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)

def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)        

def paste_fragment(overlay, fragment, pos=(0,0)):
    # You may need to convert the color.
    fragment = cv2.cvtColor(fragment, cv2.COLOR_BGR2RGB)
    fragment_pil = Image.fromarray(fragment)
    overlay.paste(fragment_pil, pos)

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

class FormSegmeneter:
    def __init__(self, work_dir, network):
        self.network = network
        self.work_dir = work_dir
        self.opt, self.model = self.__setup()

    def __setup(self):
        """
            Model setup
        """
        print('Model setup complete')
        args = [
            '--dataroot', './data', 
            '--name', 'hicfa_pix2pix',
            '--model', 'test',
            '--netG', 'unet_256',
            '--direction', 'AtoB',
            '--model', 'test',
            '--dataset_mode', 'single',
            '--gpu_id', '-1',
            '--norm', 'batch',
            '--load_size', '1024',
            '--crop_size', '1024',
            '--checkpoints_dir', './models/segmenter',
        ]

        opt = TestOptions().parse(args)  # get test options
        # hard-code parameters for test
        opt.eval = False   # test code only supports num_threads = 0
        opt.num_threads = 0   
        opt.batch_size = 1    # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

        print(opt)
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers

        return opt, model

    def __extract_segmenation_mask(self, img, dataroot_dir, work_dir, debug_dir):
        """
            Extract segmentation mask for the image
        """
        model = self.model
        opt = self.opt
        opt.dataroot = dataroot_dir
        image_dir = work_dir
        name = 'segmenation'

        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            # Debug 
            if False :
                for label, im_data in visuals.items():
                    image_numpy = tensor2im(im_data)
                    # Tensor is in RGB format OpenCV requires BGR
                    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
                    image_name = '%s_%s.png' % (name, label)
                    save_path = os.path.join(debug_dir, image_name)                   
                    imwrite(save_path, image_numpy)
                    viewImage(image_numpy, 'Tensor Image') 

            shape=img.shape
            alpha = 0.5  
            h=shape[0]
            w=shape[1]
            overlay_img = np.ones((h, w, 3), np.uint8) * 255 # white canvas
                           
            label='prediction'
            fake_im_data = visuals['fake']
            image_numpy = tensor2im(fake_im_data)
            # Tensor is in RGB format OpenCV requires BGR
            image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            overlay_img=image_numpy

            # testing only
            framed_img = cv2.resize(img, (1024,1024))
            overlay_img = cv2.addWeighted(overlay_img, alpha, framed_img, 1 - alpha, 0)

            imwrite(save_path, image_numpy)
            # viewImage(image_numpy, 'Prediction image') 
            # viewImage(overlay_img, 'Overlay') 

            save_path_overlay = os.path.join(image_dir, "%s.png" % 'overlay')
            imwrite(save_path_overlay, overlay_img)

            # create proper aspect ratio segmenation mask
            size = [1024,1024]
            desired_size = [w, h]
            ratio_w = float(desired_size[0])/size[0]
            ratio_h = float(desired_size[1])/size[1]
            ratio = min(ratio_w, ratio_h)

            # print('ratio_w = {}'.format(ratio_w))
            # print('ratio_h = {}'.format(ratio_h))
            # print('ratio = {}'.format(ratio))

            resized_overlay = cv2.resize(overlay_img, (shape[1], shape[0]))
            resized_mask = cv2.resize(image_numpy, (shape[1], shape[0]))
            size = resized_overlay.shape
            
            save_path_overlay = os.path.join(image_dir, "%s.png" % 'overlay_resized')
            imwrite(save_path_overlay, resized_overlay)
            
            resized_mask_path = os.path.join(image_dir, "%s.png" % 'resized_mask')
            imwrite(resized_mask_path, resized_mask)

            # return image_numpy
            return resized_mask

    def __fragment(self, img, hsv, layerId, fieldId):
        """
            Segment fragment 
        """
        print('layerId / id {} : {}'.format(layerId, fieldId))
        colid = fieldId
        low_color = np.array(hsv_color_ranges[colid][0],np.uint8)
        high_color = np.array(hsv_color_ranges[colid][1],np.uint8)
        mask = cv2.inRange(hsv, low_color, high_color)
        # viewImage(mask, 'mask')

        # Extract the area of interest
        # result_white = cv2.bitwise_and(img, img, mask=mask)
        # viewImage(result_white, 'result_white') 

        #  find Canny Edges
        edged = cv2.Canny(mask, 30, 200)
        
        # Use Blur
        # blur = cv2.GaussianBlur(edged, (3, 3), 0)
        blur = cv2.GaussianBlur(mask, (3, 3), 0)

        # viewImage(blur,'blur')
        # _, threshold = cv2.threshold(blur, 70, 255, 0)
        _, threshold = cv2.threshold(blur, 70, 255, 0)
        # viewImage(threshold, 'Threashold') 
 
        fragments_dir='/tmp/fragments'
        tm = time.time_ns()
        output_filename='threshold_%s.png' % (tm)
        imwrite(os.path.join(fragments_dir, output_filename), blur)

        contours, hierarchy =  cv2.findContours(blur,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours, -1, (255, 255, 0), 2)
        # viewImage(img,'Contours')

        font = cv2.FONT_HERSHEY_SIMPLEX
        all_boxes = []
        cls_scores = []
        all_pts = []
        drift = 5

        for cnt in contours:
            # (center(x, y), (width, height), angle of rotation)
            (x, y), (width, height), angle = rect = cv2.minAreaRect(cnt)
            # 90.0 deg https://theailearner.com/tag/cv2-minarearect/
            aspect_ratio = min(width, height) / max(width, height)
            # if angle < 90 - drift or angle > 90 + drift:
            #     continue
            # convert
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # Calculate the moments and get area
            # Trying to filter out small pieces
            M = cv2.moments(cnt)
            area = M['m00']
            if area > 50:
                x,y,w,h = cv2.boundingRect(cnt)
                bb = [x ,y, w, h]

                if False:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    try:
                        cv2.drawContours(img, [box], -1, (255, 0, 0), 2, 1)
                        org = [box[3][0]+5, box[3][1]-5]
                        label = 'id : {id}'.format(id=colid)
                        _=cv2.putText(img, label, org, font, .4, (0, 0, 0), 1, cv2.LINE_AA)
                    except Exception as e:
                        print(e)

                all_boxes.append(bb)
                all_pts.append(box)
                cls_scores.append(area)

        # important that we apply non-max suppression to the candiates
        if len(all_boxes) == 0:
            print('Unable to extract layerId / id {} : {}'.format(layerId, fieldId))
            return None, None

        # Testing NMS
        keepers = non_max_suppression_fast(np.array(all_boxes), .03)

        cls_scores = np.array(cls_scores) # prevent 'list indices must be integers or slices, not tuple'
        all_boxes = np.hstack((all_boxes, cls_scores [:, np.newaxis])).astype(np.float32, copy=False)
        
        keep = nms(all_boxes, 0.3)
        idx = keep[0]
        box = all_pts[idx]
        non_scored_box = all_boxes[idx][:4].astype('int32')  

        print('keep    : {}'.format(non_scored_box))
        print('keepers : {}'.format(keepers))

        if False:
            try:
                cv2.drawContours(img, [box], -1, (255, 0, 0), 2, 1)
                org = [box[3][0]+5, box[3][1]-5]
                label = 'id : {id}'.format(id=colid)
                _=cv2.putText(img, label, org, font, .4, (0, 0, 0), 1, cv2.LINE_AA)

            except Exception as e:
                print(e)
            viewImage(img, 'Final Fragment')
        # viewImage(img, 'Final Fragment')
        non_scored_box = all_boxes[idx][:4].astype('int32')  
        return box, non_scored_box

    def process(self, img_path):
        """
            Form segmentation 
        """
        if not os.path.exists(img_path):
            raise Exception('File not found : {}'.format(img_path))

        name = img_path.split('/')[-1]
        work_dir = os.path.join(self.work_dir, name, 'work')
        debug_dir = os.path.join(self.work_dir, name, 'debug')        
        dataroot_dir = os.path.join(self.work_dir, name, 'dataroot')
        fragments_dir = os.path.join(debug_dir, 'fragments')

        ensure_exists(self.work_dir)
        ensure_exists(work_dir)
        ensure_exists(debug_dir)
        ensure_exists(fragments_dir)
        ensure_exists(dataroot_dir)

        dst_file_name = os.path.join(dataroot_dir, name)
        if not os.path.exists(dst_file_name):
            copyfile(img_path, dst_file_name)

        img = cv2.imread(dst_file_name)
        # viewImage(img, 'Source Image') 
        font = cv2.FONT_HERSHEY_SIMPLEX
        segmask = self.__extract_segmenation_mask(img, dataroot_dir, work_dir, debug_dir)
        # viewImage(segmask, 'segmask') 
        # converting from BGR to HSV color space
        hsv = cv2.cvtColor(segmask, cv2.COLOR_BGR2HSV)
        # viewImage(hsv, 'HSV') 

        # causes issues due to conversion
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segmask = apply_filter(segmask)
        # viewImage(segmask, "Source Image") 

        tm = time.time_ns()
        imwrite(os.path.join(debug_dir, 'filtered_{}.png'.format(tm)), segmask)
        
        print("Processing segmentation")
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # viewImage(hsv, 'HSV') 

        layers = {
            'layer_1' : {
                'HCFA02': 0, 
                'HCFA05_ADDRESS': 1, 
                'HCFA05_CITY': 2,
                'HCFA05_STATE': 3,
                'HCFA05_ZIP': 4,
                'HCFA05_PHONE': 5,
                'HCFA21': 6,
                'HCFA24': 7,
                'HCFA33_BILLING': 8,
            },
        }

        fragments = dict()
        for lkey in layers.keys():
            print('Processing layer : {}'.format(lkey))
            layer = layers[lkey]
            for key in layer.keys():
                val = layer[key]
                box_rect, box = self.__fragment(segmask, hsv, lkey, val)     
                # it is possible to get bad segmask
                if box is None:
                    continue
                snippet = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                
                pil_snippet = Image.fromarray(snippet)
                
                pil_snippet = make_power_2(pil_snippet, base=4, method=Image.BICUBIC)
                cv_snip = np.array(pil_snippet)                
                snippet = cv2.cvtColor(cv_snip, cv2.COLOR_RGB2BGR)# convert RGB to BGR
                
                # expand the snippet to be framed in NxN border
                # try 8 px on each side
                shape = snippet.shape

                pad = 64

                pil_padded = Image.new('RGB', (shape[1] + pad, shape[0] + pad), color=(255,255,255,0))
                paste_fragment(pil_padded, snippet, (pad//2, pad//2))

                savepath = os.path.join(debug_dir, "%s-%s.jpg" % ('padded_snippet' , key))
                pil_padded.save(savepath, format='JPEG', subsampling=0, quality=100)

                cv_snip = np.array(pil_padded)                
                snippet = cv2.cvtColor(cv_snip, cv2.COLOR_RGB2BGR)# convert RGB to BGR

                frag = {
                    'layer':lkey,
                    'key':key,
                    'id':val,
                    'box': box,
                    'snippet':snippet
                }
                
                fragments[key] = frag
                try:
                    cv2.drawContours(segmask, [box_rect], -1, (255, 0, 0), 2, 1)
                    org = [box_rect[3][0]+5, box_rect[3][1]-5]
                    label = '{label} ({id})'.format(id=val, label=key)
                    _=cv2.putText(segmask, label, org, font, .4, (0, 0, 0), 1, cv2.LINE_AA)
                except Exception as e:
                    print(e)

        # viewImage(segmask, 'final')
        shape = img.shape
        overlay = Image.new('RGB', (shape[1], shape[0]), color=(255,255,255,0))

        # process extracted fragments
        for _key in fragments.keys():
            frag = fragments[_key]
            key = frag['key']
            _id = frag['id']
            box = frag['box'] # x,y,w,h
            snippet = frag['snippet'] # x,y,w,h
            # it is possible to get bad segmask
            if box is None:
                continue

            paste_fragment(overlay, snippet, (box[0], box[1]))

            tm = time.time_ns()
            output_filename='%s-%d-%s.png' % (key, _id, tm)
            imwrite(os.path.join(fragments_dir, output_filename), snippet)

        savepath = os.path.join(debug_dir, "%s-%s.jpg" % ('fragment_overlay' , tm))
        overlay.save(savepath, format='JPEG', subsampling=0, quality=100)
        
        return fragments

    def build_clean_fragments(self, img_path, fragments):
        """
            Build clean fragments document
            This is primarly for debug purposed
        """
        print('Building clean fragments')
        img = cv2.imread(img_path)
        shape = img.shape
        overlay = Image.new('RGB', (shape[1], shape[0]), color=(255,255,255,0))

        # process extracted fragments
        for _key in fragments.keys():
            frag = fragments[_key]
            key = frag['key']
            _id = frag['id']
            box = frag['box'] # x,y,w,h
            snippet = frag['snippet'] # Cliped snipped
            if 'clean' not in frag:
                continue

            clean = frag['clean'] # Cleaned snipped
            if box is None:
                continue
            paste_fragment(overlay, clean, (box[0], box[1]))

        tm = time.time_ns()
        name = img_path.split('/')[-1]
        debug_dir = os.path.join(self.work_dir, name, 'debug')
        savepath = os.path.join(debug_dir, "%s-%s.jpg" % ('clean_overlay' , tm))
        overlay.save(savepath, format='JPEG', subsampling=0, quality=100)

def segment(img_path):
    print('Segment')

    src = '/tmp/form-segmentation/PID_10_5_0_113174.tif/fields_debug/HCFA02/segmenation_fake.png'
    src = '/tmp/form-segmentation/PID_10_5_0_113174.tif/fields_debug/HCFA05_ADDRESS/segmenation_fake.png'
    src = '/tmp/form-segmentation/PID_10_5_0_113174.tif/fields_debug/HCFA05_PHONE/segmenation_fake.png'
    src = '/tmp/form-segmentation/PID_10_5_0_113174.tif/fields_debug/HCFA05_STATE/segmenation_fake.png'
    # snip = cv2.imread(src)

    # boxer = BoxProcessor()
    # boxer.process(snip)
    work_dir='/tmp/form-segmentation'

    segmenter = FormSegmeneter(work_dir, network="")
    fragments = segmenter.process(img_path)

    fp = FieldProcessor(work_dir='/tmp/form-segmentation')

    fragments['HCFA02']['clean'] = fp.process(img_path,fragments['HCFA02'])
    fragments['HCFA05_ADDRESS']['clean'] = fp.process(img_path,fragments['HCFA05_ADDRESS'])
    fragments['HCFA05_CITY']['clean'] = fp.process(img_path,fragments['HCFA05_CITY'])
    fragments['HCFA05_STATE']['clean'] = fp.process(img_path,fragments['HCFA05_STATE'])
    fragments['HCFA05_ZIP']['clean'] = fp.process(img_path,fragments['HCFA05_ZIP'])
    fragments['HCFA05_PHONE']['clean'] = fp.process(img_path,fragments['HCFA05_PHONE'])
    fragments['HCFA33_BILLING']['clean'] = fp.process(img_path,fragments['HCFA33_BILLING'])
    # fragments['HCFA21']['clean'] = fp.process(img_path,fragments['HCFA21'])
    
    clean_img=segmenter.build_clean_fragments(img_path, fragments)

    boxer = BoxProcessor()
    # boxer.process(snip)

    name = img_path.split('/')[-1]
    work_dir = os.path.join(work_dir, name)
    debug_dir = os.path.join(work_dir, 'boxes')

    ensure_exists(debug_dir)

    for key in fragments.keys():
        frag = fragments[key]
        print(frag)

        if 'clean' in frag:
            snip = frag['clean']
            print(snip.shape)
            detection = boxer.process(snip)

            tm = time.time_ns()
            name = img_path.split('/')[-1]
            savepath = os.path.join(debug_dir, "%s-%s.jpg" % ('clean_overlay' , tm))
            imwrite(savepath, detection)


if __name__ == '__main__':
    img_path ='/tmp/hicfa/PID_10_5_0_2787.original.redacted.tif'
    img_path ='/tmp/hicfa/PID_10_5_0_113174.tif'
    img_path ='/tmp/hicfa/PID_10_5_0_3101.original.tif'
    img_path ='/tmp/hicfa/PID_10_5_0_3103.original.tif'
    img_path ='/tmp/hicfa/PID_10_5_0_3104.original.tif'

    img_path ='/tmp/hicfa/task_3100-3199-2021_05_26_23_59_41-cvat for images 1.1/images/PID_10_5_0_3101.original.tif'
    img_path ='/tmp/hicfa/task_3100-3199-2021_05_26_23_59_41-cvat for images 1.1/images/PID_10_5_0_3102.original.tif'
    img_path ='/tmp/hicfa/task_3100-3199-2021_05_26_23_59_41-cvat for images 1.1/images/PID_10_5_0_3104.original.tif'
 
    segment(img_path)


    # /tmp/hicfa/task_3100-3199-2021_05_26_23_59_41-cvat for images 1.1/images
    # import glob
    # for name in glob.glob('/tmp/hicfa/task_3100-3199-2021_05_26_23_59_41-cvat for images 1.1/images/*.tif'):
    #     print(name)
    #     segment(name)
    #     break
    
    # segment(img_path)    
    # for i in range(100):
    #     segmenter.process(img_path)