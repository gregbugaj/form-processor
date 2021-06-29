
from form import processor
from operator import ne
import os
import cv2
import argparse
import numpy as np

from form.segmenter import FormSegmeneter
from boxes.box_processor import BoxProcessor
from form.numpyencoder import NumpyEncoder
from form.icr_processor import IcrProcessor
from form.field_processor import FieldProcessor

def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)   
    return dir   

class FormProcessor:
    def __init__(self, work_dir:str = '/tmp/form-segmentation', cuda: bool = False) -> None:
        print("Form processor [cuda={}]".format(cuda))
        self.work_dir = work_dir
        self.__load()

    def __load(self):
        print('Initializing processor')

        self.segmenter = FormSegmeneter(work_dir, network="")
        self.fp = FieldProcessor(work_dir)
        self.boxer = BoxProcessor(work_dir, cuda=False)
        self.icr = IcrProcessor(work_dir)

    def process(self, img_path):
        print(f'Processing image : {img_path}')
        work_dir =  self.work_dir

        id = img_path.split('/')[-1]
        debug_dir = ensure_exists(os.path.join(work_dir, id, 'work'))

        segmenter = self.segmenter
        fp = self.fp
        boxer = self.boxer
        icr = self.icr

        seg_fragments, img, segmask = segmenter.segment(id, img_path)
        rectangles, box_fragment_imgs, overlay_img, _ = boxer.process_full_extraction(id, img)
        segmenter.fragment_to_box_snippet(id, seg_fragments, overlay_img)

        print('-------- Image information -----------')

        print('img         : {}'.format(img.shape))
        print('overlay_img : {}'.format(overlay_img.shape))
        print('segmask     : {}'.format(segmask.shape))

        alpha = 0.5 

        shape=img.shape
        h=shape[0]
        w=shape[1]

        canvas_img = np.ones((h, w, 3), np.uint8) * 255 # white canvas
        canvas_img = cv2.addWeighted(canvas_img, alpha, segmask, 1 - alpha, 0)
        canvas_img = cv2.addWeighted(canvas_img, alpha, overlay_img, 1 - alpha, 0)

        file_path = os.path.join(debug_dir, "text_over_segmask.png")
        cv2.imwrite(file_path, canvas_img)

        # Same model

        # seg_fragments['HCFA02']['snippet_clean'] = fp.process(id, seg_fragments['HCFA02'])
        # seg_fragments['HCFA05_ADDRESS']['clean'] = fp.process(id,seg_fragments['HCFA05_ADDRESS'])
        # fragments['HCFA05_CITY']['clean'] = fp.process(img_path,fragments['HCFA05_CITY'])
        # fragments['HCFA05_STATE']['clean'] = fp.process(img_path,fragments['HCFA05_STATE'])
        # fragments['HCFA05_ZIP']['clean'] = fp.process(img_path,fragments['HCFA05_ZIP'])
        # fragments['HCFA05_PHONE']['clean'] = fp.process(img_path,fragments['HCFA05_PHONE'])
        
        # seg_fragments['HCFA33_BILLING']['snippet_clean'] = fp.process(id, seg_fragments['HCFA33_BILLING'])
        # fragments['HCFA21']['clean'] = fp.process(img_path,fragments['HCFA21'])
        # clean_img=segmenter.build_clean_fragments(id, img, seg_fragments)
        # boxes, fragments, _=boxer.extract_bounding_boxes(id, 'HCFA02', seg_fragments['HCFA02']['snippet_clean'])
        # boxer.extract_bounding_boxes(id, 'HCFA33_BILLING', seg_fragments['HCFA33_BILLING']['snippet_clean'])

        field = ['HCFA02', 'HCFA33_BILLING', 'HCFA05_ADDRESS', 'HCFA05_CITY', 'HCFA05_STATE', 'HCFA05_ZIP', 'HCFA05_PHONE']
        field = ['HCFA05_PHONE']
        
        result = []
        for field in field:
            print(f'Processing field : {field}')
        
            seg_fragments[field]['snippet_clean'] = fp.process(id, seg_fragments[field])
            snippet = seg_fragments[field]['snippet_clean']
            boxes, fragments, lines, _ = boxer.extract_bounding_boxes(id, field, snippet)

            icr_results = icr.icr_extract(id, field, snippet, boxes, fragments, lines)

            result.append(icr_results)

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Run evaluator')
    parser.add_argument('--img', dest='img_src', help='Image to evaluate', default='data/clean.png', type=str)
    parser.add_argument('--output', dest='dir_out', help='Output directory evaluate', default='./data/debug', type=str)
    parser.add_argument('--debug', dest='debug', help='Debug results', default=False, type=bool)

    return parser.parse_args() 

if __name__ == '__main__':
    args = parse_args()

    work_dir='/tmp/form-segmentation'

    img_path='/tmp/form-segmentation/a_013.png/fields_debug/HCFA33_BILLING/segmenation_fake.png'

    if False:
        img_path='/home/greg/tmp/snippets/009.png'
        

        id = img_path.split('/')[-1]
        debug_dir = ensure_exists(os.path.join(work_dir, id, 'work'))

        snippet = cv2.imread(img_path)

        fp = FieldProcessor(work_dir)
        icr = IcrProcessor(work_dir)

        # snippet_clean = fp.process(id, 'HCFA33_BILLING', snippet)
        # snippet_clean = fp.process(id, 'HCFA05_PHONE', snippet)

        boxer = BoxProcessor(work_dir, cuda=False)
        boxes, img_fragments, lines, _= boxer.extract_bounding_boxes(id, 'field', snippet)
        icr.icr_extract(id, 'HCFA05_PHONE', snippet, boxes, img_fragments, lines)

    if True:
        processor = FormProcessor(work_dir='/tmp/form-segmentation', cuda=False)

    if False:
        work_dir='/tmp/form-segmentation'
        boxer = BoxProcessor(work_dir, cuda=False)

        import glob
        for name in glob.glob('/home/greg/tmp/single/*.jpg'):
            print(name)
            id = name.split('/')[-1]
            debug_dir = ensure_exists(os.path.join(work_dir, id, 'work'))
            snip = read_image(name)
            if snip is None:
                continue
            boxes, fragments, lines, _= boxer.extract_bounding_boxes(id, 'field', snip)
            icr(id, snip, boxes, fragments)

    if False:
        import glob
        # for name in glob.glob('/tmp/hicfa/*.tif'):
        # for name in glob.glob('/home/greg/tmp/task_3100-3199-2021_05_26_23_59_41-cvat/images/*.tif'):
        for name in glob.glob('/home/greg/tmp/task_3100-3199-2021_05_26_23_59_41-cvat/images/PID_10_5_0_3129.original.tif'):
            try:
                print(name)
                segment(name)
                break
            except Exception as ident:
                print(ident)
