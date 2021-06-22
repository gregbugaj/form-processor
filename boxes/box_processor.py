import numpy as np
import copy
import cv2
import numpy as np
from PIL import Image

# Add parent to the search path so we can reference the modules(craft, pix2pix) here without throwing and exception 
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from craft_text_detector import Craft

# import craft functions
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

from utils.resize_image import resize_image

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
class BoxProcessor:
    def __init__(self,work_dir:str = '/tmp/form-segmentation', cuda: bool = False) -> None:
        print("Box processor [cuda={}]".format(cuda))
        self.cuda = cuda
        self.work_dir = work_dir
        self.craft_net, self.refine_net = self.__load()

    def __load(self):
        # load models
        refine_net = load_refinenet_model(cuda=self.cuda)
        craft_net = load_craftnet_model(cuda=self.cuda)

        return craft_net, refine_net

    def extract_bounding_boxes(self,id,key, image):
        """
            Extrac bouding boxes for specific image
        """
        print('Extracting bounding boxes')

        debug_dir =  ensure_exists(os.path.join(self.work_dir,id,'bounding_boxes', key, 'debug'))
        crops_dir = ensure_exists(os.path.join(self.work_dir,id,'bounding_boxes', key, 'crop'))
        output_dir = ensure_exists(os.path.join(self.work_dir,id,'bounding_boxes', key, 'output'))

        h=image.shape[0]
        w=image.shape[1]
        image=copy.deepcopy(image)

        # perform prediction
        prediction_result = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.6,
            link_threshold=0.75,
            low_text=0.4,
            cuda=self.cuda,
            long_size=w 
        )

        image_results = copy.deepcopy(image)
        export_extra_results(
            image=image_results,
            regions=prediction_result["boxes"],
            heatmaps=prediction_result["heatmaps"],
            output_dir=output_dir
        )

        # output text only blocks
        # deepcopy image so that original is not altered
        image = copy.deepcopy(image)
        regions=prediction_result["boxes"]
        pil_image = Image.new('RGB', (image.shape[1], image.shape[0]), color=(255,255,255,0))

        rect_from_poly=[]
        fragments=[]
        for i, region in enumerate(regions):
            region = np.array(region).astype(np.int32).reshape((-1))
            region = region.reshape(-1, 2)
            poly = region.reshape((-1, 1, 2))

            box = cv2.boundingRect(poly)
            box = np.array(box).astype(np.int32)
            x,y,w,h = box
            
            snippet = crop_poly_low(image, poly)
            fragments.append(snippet)
            rect_from_poly.append(box)

            # export cropped region
            file_path = os.path.join(crops_dir, "%s.jpg" % (i))
            cv2.imwrite(file_path, snippet)
            paste_fragment(pil_image, snippet, (x, y))

        savepath = os.path.join(debug_dir, "%s.jpg" % ('txt_overlay'))
        pil_image.save(savepath, format='JPEG', subsampling=0, quality=100)

        return np.array(rect_from_poly), np.array(fragments), prediction_result["boxes"]    


    def process_full_extraction(self,id,image):
        """
            Do full page text extraction
        """
        print('Processing full page extraction: {}'.format(id))

        debug_dir =  ensure_exists(os.path.join(self.work_dir,id,'boxes_full'))
        crops_dir = ensure_exists(os.path.join(self.work_dir,id,'crops_full'))

        # # read image
        # image = read_image(image)
        h=image.shape[0]
        w=image.shape[1]
        image=copy.deepcopy(image)

        # perform prediction
        prediction_result = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=self.cuda,
            long_size=w #1280
        )

        # output text only blocks
        # deepcopy image so that original is not altered
        image = copy.deepcopy(image)
        regions=prediction_result["boxes"]

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
            if h < 15:
                continue
            
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
        return np.array(rect_from_poly), np.array(fragments), cv_img, prediction_result["boxes"]    
