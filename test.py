
from utils.image_utils import read_image
from form.field_processor import FieldProcessor
from form.form_alignment import FormAlignment
import json
import os
from utils.utils import current_milli_time, ensure_exists
import cv2
import argparse
import numpy as np
import glob

from run import FormProcessor
from boxes.box_processor import BoxProcessor
from form.icr_processor import IcrProcessor

from form.segmenter import FormSegmeneter

if __name__ == '__main__':

    work_dir='/tmp/form-segmentation'
    img_path='/home/greg/tmp/hicfa/PID_10_5_0_3100.original.tif'
    img_path='/home/greg/dataset/data-hipa/forms/hcfa-allstate/269692_202006290005214_001.tif'
    img_path='/tmp/form-segmentation/269692_202006290005214_001.tif/dataroot/aligned_segment.png'
    img_path='/home/greg/tmp/aligned_segment_scaled.png'
    img_path='/media/greg/XENSERVER-6/27ofStateFarm100/272943_0031516168746_001.tif'
    img_path='/tmp/form-segmentation/aligned_segment.png'
    img_path='/home/greg/dataset/data-hipa/forms/hcfa-allstate/269688_202006290005126_001.tif'

    if False:
        image = cv2.imread(img_path)
        work_dir='/tmp/segmentation-mask'
        m0 = current_milli_time()
        id = img_path.split('/')[-1]
        segmenter = FormAlignment(work_dir)
        m1 = current_milli_time()-m0
        
        print('Form processor initialized in {} ms'.format(m1))
        m0 = current_milli_time()
        segmask = segmenter.align(id, image)
        m1 = current_milli_time()- m0
        print('Time {} ms'.format(m1))

    if False:
        work_dir='/tmp/segmentation-mask'
        m0 = current_milli_time()
        id = img_path.split('/')[-1]
        segmenter = FormSegmeneter(work_dir)
        m1 = current_milli_time()-m0
        
        print('Form processor initialized in {} ms'.format(m1))
        m0 = current_milli_time()
        seg_fragments, img, segmask = segmenter.segment(id, img_path)
        m1 = current_milli_time()- m0
        print('Time {} ms'.format(m1))

        
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

    if True:
        config_path = 'config-single.json'
        with open(config_path) as f:
            config = json.load(f)

        processor = FormProcessor(work_dir=work_dir, config=config, cuda=False)
        # for name in glob.glob('/home/greg/dev/assets-private/27ofStateFarm100/273429_0031517628960_001.tif'):
        for name in glob.glob('/home/greg/dataset/data-hipa/forms/hcfa-allstate/*.tif'):
            try:
                print(name)
                results = processor.process(name)
            except Exception as ident:
                print(ident)

        # processor = FormProcessor(work_dir=work_dir, config=config, cuda=False)
        # results = processor.process(img_path)

    if False:
        img_path='/tmp/form-segmentation/272944_0031516168976_001.tif/fields_debug/HCFA02/segmenation_real.png'

        id = img_path.split('/')[-1]
        debug_dir = ensure_exists(os.path.join(work_dir, id, 'work'))
        snippet = cv2.imread(img_path)

        icr = IcrProcessor(work_dir)
        boxer = BoxProcessor(work_dir, cuda=False)
        boxes, img_fragments, lines, _= boxer.extract_bounding_boxes(id, 'field', snippet)
        icr.icr_extract(id, 'HCFA05_PHONE', snippet, boxes, img_fragments, lines)


    if False:
        work_dir='/tmp/form-segmentation'
        boxer = BoxProcessor(work_dir, cuda=False)
        icr = IcrProcessor(work_dir)

        import glob
        for name in glob.glob('/tmp/form-segmentation/272944_0031516168976_001.tif/fields_debug/HCFA02/*.jpg'):
            print(name)
            id = name.split('/')[-1]
            debug_dir = ensure_exists(os.path.join(work_dir, id, 'work'))
            snip = read_image(name)
            if snip is None:
                continue
            boxes, fragments, lines, _= boxer.extract_bounding_boxes(id, 'field', snip)
            icr(id, snip, boxes, fragments)

    if False:
        # for name in glob.glob('/tmp/hicfa/*.tif'):
        # for name in glob.glob('/home/greg/tmp/task_3100-3199-2021_05_26_23_59_41-cvat/images/*.tif'):
        for name in glob.glob('/home/greg/tmp/task_3100-3199-2021_05_26_23_59_41-cvat/images/PID_10_5_0_3129.original.tif'):
            try:
                print(name)
                segment(name)
                break
            except Exception as ident:
                print(ident)


    if False:

        img_path='/home/greg/tmp/hicfa/PID_10_5_0_3112.original.tif'
        img_path='/home/greg/tmp/hicfa/PID_10_5_0_3128.original.tif'
        work_dir='/tmp/form-segmentation'

        snippet = cv2.imread(img_path)
        boxer = BoxProcessor(work_dir, cuda=False)
        boxer.process_full_extraction('id', snippet, .3)
