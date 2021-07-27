
from form.form_alignment import FormAlignment
import os
import sys
from utils.visualize import visualize
import cv2
import argparse
import numpy as np
import json
import time

from PIL import Image

from form.segmenter import FormSegmeneter
from boxes.box_processor import BoxProcessor
from form.numpyencoder import NumpyEncoder
from form.icr_processor import IcrProcessor
from form.field_processor import FieldProcessor

from utils.image_utils import imwrite, paste_fragment
from utils.utils import current_milli_time, ensure_exists

# logging
import logging
from logging.handlers import RotatingFileHandler
from logging import handlers

LOGFILE = 'processor.log'
log = logging.getLogger('FP.Segmenter')
log.setLevel(logging.INFO)
format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
log.addHandler(ch)

fh = handlers.RotatingFileHandler(LOGFILE, maxBytes=(1048576 * 5), backupCount=7)
fh.setFormatter(format)
log.addHandler(fh)

class FormProcessor:
    def __init__(self, work_dir:str = '/tmp/form-segmentation', config:object=None, cuda: bool = False) -> None:
        log.info("Form processor [cuda={}]".format(cuda))
        log.info("Work dir : %s", work_dir)
        log.info("Config   : %s", config)
        self.work_dir = work_dir
        self.config = config
        self.cuda = cuda
        self.__load()

    def __load(self):
        log.info('Initializing processor')
        # All models need to be rebuild
        segmenter_models = dict()
        for field_config in self.config['fields']:
            segmenter_models[field_config['field']] = field_config['segmenter']

        m0 = current_milli_time()
        work_dir =  self.work_dir
        
        self.form_align = FormAlignment(work_dir)
        self.segmenter = FormSegmeneter(work_dir)
        self.field_processor = FieldProcessor(work_dir, segmenter_models)
        self.box_processor = BoxProcessor(work_dir, cuda=self.cuda)
        self.icr_processor = IcrProcessor(work_dir, cuda=self.cuda)
        m1 = current_milli_time()-m0

        log.info('Form processor initialized in {} ms'.format(m1))

    def apply_heuristics(self, id, key, seg_box, overlay_img, heuristics_config):
        """
            Apply heuristics to the overlay
            Currently this only applies to fields that can be segmented into multiple lines without cleaning
        """
        # TODO : Use heuristics config to get the necessary parameters rather than having them hardcoded in here
        print(f'Applying heuristics to segmentaion box : {seg_box}')
        work_dir =  self.work_dir

        try:
            if seg_box is None:
                return False, None

            box_processor = self.box_processor
            img = overlay_img
            
            # allow for small padding around the component, this padding is not this same as margin
            pad_x = 20
            pad_y = 8
            snippet = img[seg_box[1]:seg_box[1]+seg_box[3] + pad_y, seg_box[0]-pad_x:seg_box[0]+seg_box[2]+pad_x*2]
            snippet = cv2.cvtColor(snippet, cv2.COLOR_RGB2BGR)# convert RGB to BGR
            boxes, fragments, lines, _ = box_processor.extract_bounding_boxes(id, key, snippet)

            # header/dataline
            # 1-based dataline indexes
            pil_image = Image.new('RGB', (snippet.shape[1], snippet.shape[0]), color=(255,255,255,0))
            max_line_number = 0
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
            ar = A1 / A2
            lr = max_h / seg_box[3]

            if ar < 0.05 or ar > 0.30 or lr < 0.30:
                return False, None

            if min_x < 2 or (min_y + max_h) - 2 > seg_box[1]+seg_box[3]:
                return False, None

            if True:
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
        work_dir =  self.work_dir
        id = img_path.split('/')[-1]
        log.info('[%s] Processing image : %s', id, img_path)
        debug_dir = ensure_exists(os.path.join(work_dir, id, 'work'))
        dataroot_dir = ensure_exists(os.path.join(self.work_dir, id, 'dataroot'))

        config = self.config
        form_align = self.form_align
        segmenter = self.segmenter
        field_processor = self.field_processor
        box_processor = self.box_processor
        icr_processor = self.icr_processor
        
        aligned_segment = form_align.align(id, img_path)
        segment_path = os.path.join(dataroot_dir, 'aligned_segment.png')

        cv2.imwrite(segment_path, aligned_segment)
        seg_fragments, img, segmask = segmenter.segment(id, segment_path)

        # Extract boxes and turn them into boxes
        # overlay_boxes, box_fragment_imgs, overlay_img, _ = box_processor.process_full_extraction(id, img)
        # segmenter.fragment_to_box_snippet(id, seg_fragments, overlay_img)
        overlay_img = img

        m1 = current_milli_time()
        log.info('[%s] Segmentation completed in : %s(ms)', id, m1-m0)
        
        alpha = 0.5 
        shape=img.shape
        h = shape[0]
        w = shape[1]

        canvas_img = np.ones((h, w, 3), np.uint8) * 255 # white canvas
        canvas_img = cv2.addWeighted(canvas_img, alpha, segmask, 1 - alpha, 0)
        canvas_img = cv2.addWeighted(canvas_img, alpha, overlay_img, 1 - alpha, 0)

        if True:
            file_path = os.path.join(debug_dir, "text_over_segmask.png")
            cv2.imwrite(file_path, canvas_img)

        # All models need to be rebuild
        fields = config['fields']
        field_results = []
        s = current_milli_time()

        for field_config in fields:
            log.info('[%s] [%s] Start field processing', id, field_config)
            seg_name = field_config['segmenter']
            field = field_config['field']
            heuristics = field_config['heuristics']

            if not field_config['enabled']:
                log.info('[%s] [%s] Field disabled', id, field)
                continue
            
            margin = [0,0,0,0] # L,T,R,B
            if 'margin' in field_config:
                m = field_config['margin']
                m = m.split(',')
                margin = [int (k) for k in m]

            L,T,R,B = margin 
            print(f'Margin >> {margin}')
            
            icr_results = {}
            failed = False
            heuristics_applied = False
            
            try:
                m0 = current_milli_time()
                fragment = seg_fragments[field]
                snippet = fragment['snippet']
                box = fragment['box']

                # Heuristics is applied to original image rather than image
                # TODO : Dynamicly call heuristics method rather than hardcode it here
                if heuristics['enabled']:
                    heuristics_applied, heuristics_snippet = self.apply_heuristics(id, field, box, overlay_img, heuristics)    
                    m1 = current_milli_time()
                    log.info('[%s] [%s] heuristics applied, time : %s, %s(ms)', id, field, heuristics_applied, m1-m0)
                    
                if heuristics_applied:
                    fields_aggro_dir = ensure_exists(os.path.join(self.work_dir, 'fields_heuristics', field))
                    fields_clean_aggro_dir = ensure_exists(os.path.join(self.work_dir, 'fields_clean_heuristics', field))
                    # fields_debug_aggro_dir = ensure_exists(os.path.join(self.work_dir, 'fields_debug', field))
        
                    imwrite(os.path.join(fields_aggro_dir, '%s.png' % (id)), snippet)
                    imwrite(os.path.join(fields_clean_aggro_dir, '%s.png' % (id)), heuristics_snippet)

                    snippet = heuristics_snippet
                    snippet_margin = heuristics_snippet
                else:
                    # if we don't have a segmenation specified then we skip it
                    if seg_name == '':
                        log.info('[%s] [%s] Skipping segmenation', id, field)
                    else:
                        m0 = current_milli_time()
                        # apply margin to the box and snippet                
                        seg_box = [max(0, box[0]+L), box[1], box[2], box[3]+B]
                        snippet_margin = img[seg_box[1]:seg_box[1]+seg_box[3], seg_box[0]:seg_box[0]+seg_box[2]]
                        box = seg_box
                        snippet = field_processor.process(id, field, snippet_margin)

                        m1 = current_milli_time()
                        log.info('[%s] [%s] Field processor time : %s(ms)', id, field, m1-m0)

                fragment['snippet_clean'] = snippet
                m0 = current_milli_time()
                boxes, fragments, lines, _ = box_processor.extract_bounding_boxes(id, field, snippet)
                m1 = current_milli_time()
                log.info('[%s] [%s] Box processor time : %s(ms)', id, field, m1 - m0)

                m0 = current_milli_time()
                icr_results, icr_overlay = icr_processor.icr_extract(id, field, snippet, boxes, fragments, lines)

                debug_dir = ensure_exists(os.path.join(work_dir, id, 'work_figures'))
                save_path = os.path.join(debug_dir, '%s.png' % (field))
                visualize(imgpath=save_path, snippet=snippet_margin, cleaned=snippet, icr=icr_overlay)

                m1 = current_milli_time()
                log.info('[%s] [%s] ICR processor time : %s(ms)', id, field, m1 - m0)
                log.info('[%s] [%s] Field eval time : %s(ms)', id, field, m1 - s)
            except Exception as ident:
                failed = True
                print(f'Field failed : {field}')
                log.error('Failed processing field', ident)

            data = {
                'field':field,
                'heuristics_applied':heuristics_applied,
                'icr':icr_results,
                'failed':failed
            }

            field_results.append(data)

        e = current_milli_time()
        log.info('[%s] [] Eval time : %s(ms)', id, e-s)
        return {
            'meta': {'imageSize': {'width': img.shape[1], 'height': img.shape[0]}, 'id':id},
            'eval_time':m1,
            'fields': field_results,
        }

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Run evaluator')
    parser.add_argument('--img', dest='img_src', help='Image to evaluate', default='data/form.png', type=str)
    parser.add_argument('--output_dir', dest='output_dir', help='Item result directory', default='./data/output', type=str)
    parser.add_argument('--work_dir', dest='debug', help='Debug directory results', default='./data/debug', type=str)
    parser.add_argument('--config', dest='config', help='Field configuration file', default='./config.json', type=str)

    return parser.parse_args() 

def main(config_path, img_path, output_dir, work_dir, cuda):
    print('Main')
    print(f'cuda        = {cuda}')
    print(f'config_path = {config_path}')
    print(f'img_path    = {img_path}')
    print(f'output_dir  = {output_dir}')
    print(f'work_dir    = {work_dir}')

    if not os.path.exists(config_path):
        raise Exception(f'config file not found : {config_path}')

    with open(config_path) as f:
        config = json.load(f)

    processor = FormProcessor(work_dir=work_dir, config=config, cuda=cuda)
    results = processor.process(img_path)
    file_path = os.path.join(output_dir, "results.json")

    print(f'Saving results to : {file_path}')
    with open(file_path, 'w') as f:
        json.dump(results, f,  sort_keys=True,  separators=(',', ': '), ensure_ascii=False, indent=4, cls=NumpyEncoder)

if __name__ == '__main__':
    args = parse_args()
    args.img_src = '/media/greg/XENSERVER-6/27ofStateFarm100/272943_0031516168746_001.tif'
    # args.work_dir = '/tmp/form-segmentation'
    args.config = './config.json'
    # args.config = './config-single.json'

    img_path = args.img_src
    work_dir = args.work_dir
    config_path = args.config

    id = img_path.split('/')[-1]
    output_dir = ensure_exists(os.path.join(work_dir, id, 'result'))
    main(config_path=config_path, img_path=img_path, output_dir=output_dir, work_dir=work_dir, cuda=False)