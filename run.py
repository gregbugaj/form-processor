
import json
import time
from form import processor
from operator import ne
import os
import cv2
import argparse
import numpy as np

import hashlib

from form.segmenter import FormSegmeneter
from boxes.box_processor import BoxProcessor
from form.numpyencoder import NumpyEncoder
from form.icr_processor import IcrProcessor
from form.field_processor import FieldProcessor

def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)   
    return dir   

def current_milli_time():
    return round(time.time() * 1000)   
class FormProcessor:
    def __init__(self, work_dir:str = '/tmp/form-segmentation', cuda: bool = False) -> None:
        print("Form processor [cuda={}]".format(cuda))
        self.work_dir = work_dir
        self.__load()

    def __load(self):
        print('Initializing processor')
        m0 = current_milli_time()
        self.segmenter = FormSegmeneter(work_dir)
        self.fproc = FieldProcessor(work_dir)
        self.boxer = BoxProcessor(work_dir, cuda=False)
        self.icr = IcrProcessor(work_dir)
        m1 = current_milli_time()-m0

        print('Form processor initialized in {} ms'.format(m1))

    def process(self, img_path):
        m0 = current_milli_time()
        print(f'Processing image : {img_path}')
        work_dir =  self.work_dir

        id = img_path.split('/')[-1]
        debug_dir = ensure_exists(os.path.join(work_dir, id, 'work'))

        segmenter = self.segmenter
        fproc = self.fproc
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

        # All models need to be rebuild
        # fields = ['HCFA02', 'HCFA33_BILLING', 'HCFA05_ADDRESS', 'HCFA05_CITY', 'HCFA05_STATE', 'HCFA05_ZIP', 'HCFA05_PHONE']
        fields = ['HCFA33_BILLING']
        fields = ['HCFA02']
        
        print(f'All fields : {fields}')
        meta = {
            'imageSize': {'width': img.shape[1], 'height': img.shape[0]},
            'id':id
        }

        field_results = []

        for field in fields:
            print(f'Processing field : {field}')

            icr_results = {}
            failed = False
            try:
                snippet = seg_fragments[field]['snippet']
                seg_fragments[field]['snippet_clean'] = fproc.process(id, field, snippet)
                snippet = seg_fragments[field]['snippet_clean']            
                boxes, fragments, lines, _ = boxer.extract_bounding_boxes(id, field, snippet)
                icr_results = icr.icr_extract(id, field, snippet, boxes, fragments, lines)
            except Exception as ident:
                failed = True
                print(f'Field failed : {field}')
                print(ident)

            data = {
                'field':field,
                'icr':icr_results,
                'failed':failed
            }

            field_results.append(data)

        m1 = current_milli_time()-m0

        result =  {
            'meta': meta,
            'eval_time':m1,
            'fields': field_results,
        }

        print(f'Eval time : {m1}')
        return result

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Run evaluator')
    parser.add_argument('--img', dest='img_src', help='Image to evaluate', default='data/form.png', type=str)
    parser.add_argument('--output_dir', dest='output_dir', help='Item result directory', default='./data/output', type=str)
    parser.add_argument('--work_dir', dest='debug', help='Debug directory results', default='./data/debug', type=str)

    return parser.parse_args() 

def main(img_path, output_dir, work_dir, cuda):
    print('Main')
    print(f'cuda       = {cuda}')
    print(f'img_path   = {img_path}')
    print(f'output_dir = {output_dir}')
    print(f'work_dir   = {work_dir}')

    processor = FormProcessor(work_dir=work_dir, cuda=cuda)
    results = processor.process(img_path)

    print(results) 

    file_path = os.path.join(output_dir, "results.json")
    print(f'Saving results to : {file_path}')
    with open(file_path, 'w') as f:
        json.dump(results, f,  sort_keys=True,  separators=(',', ': '), ensure_ascii=False, indent=4, cls=NumpyEncoder)


if __name__ == '__main__':
    args = parse_args()

    args.img_src = '/home/greg/tmp/hicfa/PID_10_5_0_3101.original.tif'
    args.work_dir = '/tmp/form-segmentation'

    img_path = args.img_src
    work_dir = args.work_dir

    id = img_path.split('/')[-1]
    output_dir = ensure_exists(os.path.join(work_dir, id, 'result'))

    main(img_path=img_path, output_dir=output_dir, work_dir=work_dir, cuda=False)