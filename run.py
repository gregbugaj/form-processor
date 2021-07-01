
import json
import time

from PIL import Image
from utils.overlap import find_overlap
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

from utils.image_utils import paste_fragment

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
        self.field_processor = FieldProcessor(work_dir)
        self.box_processor = BoxProcessor(work_dir, cuda=False)
        self.icr_processor = IcrProcessor(work_dir)
        m1 = current_milli_time()-m0

        print('Form processor initialized in {} ms'.format(m1))

    def apply_heuristics(self, id, key, seg_box, overlay_img):
        """
            Apply heuristics to the overlay
            Currently this only applies to fields that can be segmented into multiple lines without cleaning
        """
        print(f'Applying heuristics to segmentaion box : {seg_box}')
        work_dir =  self.work_dir

        try:
            if seg_box is None:
                return False, None

            box_processor = self.box_processor
            img = overlay_img
            snippet = img[seg_box[1]:seg_box[1]+seg_box[3], seg_box[0]:seg_box[0]+seg_box[2]]
            snippet = cv2.cvtColor(snippet, cv2.COLOR_RGB2BGR)# convert RGB to BGR
            boxes, fragments, lines, _ = box_processor.extract_bounding_boxes(id, key, snippet)

            # header/dataline
            # 1-based dataline indexes
            pil_image = Image.new('RGB', (snippet.shape[1], snippet.shape[0]), color=(255,255,255,0))
            max_line_number = 0

            data_line_indexes = [1] 
            all_box_lines = []

            for i in range(len(boxes)):
                box, fragment, line = boxes[i], fragments[i], lines[i]
                if line == 2:
                    paste_fragment(pil_image, fragment, (box[0], box[1]))
                    all_box_lines.append(box)

                if line > max_line_number:
                    max_line_number = line
                    
            # We are only applying this to header/dataline 
            if len(all_box_lines) == 0 or max_line_number != 2:
                return False, None

            all_box_lines = np.array(all_box_lines)
            min_x = all_box_lines[:, 0].min()
            min_y = all_box_lines[:, 1].min()
            max_w = all_box_lines[:, 2].max()
            max_h = all_box_lines[:, 3].max()

            box = [min_x, min_y, max_w, max_h]

            A1 = (min_x + max_w) * max_h
            A2 = seg_box[2] * seg_box[3]
            ar = A1/A2
            lr = max_h / seg_box[3]

            print(f'Target box : {box}')
            print(f'Area box   : {A1}')
            print(f'Area seg   : {A2}')
            print(f'Area ratio : {ar}')

            if ar < 0.05 or ar > 0.30 or lr < 0.30:
                return False, None


            debug_dir = ensure_exists(os.path.join(work_dir, id, 'heuristics'))
            savepath = os.path.join(debug_dir, "%s.png" % (key))

            pil_image.save(savepath, format='PNG', subsampling=0, quality=100)
            cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            return True, cv_img
        except Exception as ident:
            print(ident)
        return False, None

    def process(self, img_path):
        m0 = current_milli_time()
        print(f'Processing image : {img_path}')
        work_dir =  self.work_dir

        id = img_path.split('/')[-1]
        debug_dir = ensure_exists(os.path.join(work_dir, id, 'work'))

        segmenter = self.segmenter
        field_processor = self.field_processor
        box_processor = self.box_processor
        icr_processor = self.icr_processor

        seg_fragments, img, segmask = segmenter.segment(id, img_path)
        overlay_boxes, box_fragment_imgs, overlay_img, _ = box_processor.process_full_extraction(id, img)
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
        fields = ['HCFA02', 'HCFA02']
        
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
                fragment = seg_fragments[field]
                snippet = fragment['snippet']
                box = fragment['box']
                applied, snippet_heuristic = self.apply_heuristics(id, field, box, overlay_img)    

                if applied:
                    snippet = snippet_heuristic
                else:
                    snippet = field_processor.process(id, field, snippet)

                fragment['snippet_clean'] = snippet
                boxes, fragments, lines, _ = box_processor.extract_bounding_boxes(id, field, snippet)
                icr_results = icr_processor.icr_extract(id, field, snippet, boxes, fragments, lines)
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
    args.img_src = '/home/greg/tmp/hicfa/PID_10_5_0_3103.original.tif'
    args.work_dir = '/tmp/form-segmentation'

    img_path = args.img_src
    work_dir = args.work_dir

    id = img_path.split('/')[-1]
    output_dir = ensure_exists(os.path.join(work_dir, id, 'result'))

    main(img_path=img_path, output_dir=output_dir, work_dir=work_dir, cuda=True)