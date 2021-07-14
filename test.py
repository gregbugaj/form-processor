
import json
import os
import cv2
import argparse
import numpy as np

from run import FormProcessor
from boxes.box_processor import BoxProcessor
from form.icr_processor import IcrProcessor

if __name__ == '__main__':

    work_dir='/tmp/form-segmentation'
    img_path='/home/greg/tmp/hicfa/PID_10_5_0_3101.original.tif'

    if False:
        img_path = '/tmp/form-segmentation/PID_10_5_0_3104.original.tif/fields_debug/HCFA24/segmenation_real.png'
        img_path = '/tmp/form-segmentation/PID_10_5_0_3101.original.tif/fields_debug/HCFA24/segmenation_real.png'

        snippet = cv2.imread(img_path)
        id = 'PID_10_5_0_3101'
        key = 'HCFA24'
        # snippet = cv2.cvtColor(snippet, cv2.COLOR_RGB2BGR)# convert RGB to BGR
        boxer = BoxProcessor(work_dir, cuda=False)
        boxes, img_fragments, lines, _= boxer.extract_bounding_boxes(id, 'field', snippet)

        icr = IcrProcessor(work_dir)
        icr.icr_extract(id, key, snippet, boxes, img_fragments, lines)

    if False:
        processor = FormProcessor(work_dir='/tmp/form-segmentation', cuda=False)
        processor.process(img_path)

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


    if True:

        img_path='/home/greg/tmp/hicfa/PID_10_5_0_3112.original.tif'
        img_path='/home/greg/tmp/hicfa/PID_10_5_0_3128.original.tif'
        work_dir='/tmp/form-segmentation'

        snippet = cv2.imread(img_path)
        boxer = BoxProcessor(work_dir, cuda=False)
        boxer.process_full_extraction('id', snippet, .3)
