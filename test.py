
from form import overlay
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
from form.overlay import FormOverlay
from form.optical_mark_recognition import OpticalMarkRecognition

def imwrite(path, img):
    try:
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    except Exception as ident:
        print(ident)

if __name__ == '__main__':

    if False:
        from PIL import Image
        image = Image.open('/home/gbugaj/data/rms-asp/149495857/PID_576_7188_0_149495857_converted.tif')
        image.save('/home/gbugaj/data/rms-asp/149495857/PID_576_7188_0_149495857_converted_pillow.tif', compression='group4')


        import numpy as np
        from PIL import Image
        img = Image.new("1", (800, 1280), (255))

        imlist = []
        img_dir = '/home/gbugaj/data/rms-asp/149495857/clean/'

        for _path in glob.glob(os.path.join(img_dir,'*.tif')  ):
            src_img_path = os.path.join(img_dir, _path)
            print(src_img_path)
            src = Image.open(_path)
            imlist.append(src)

        img.save("/home/gbugaj/data/rms-asp/149495857/converted.tif", compression="group4", save_all=True,  append_images=imlist)
                    


        os.exit()
    work_dir='/tmp/form-segmentation'
    img_path='/home/greg/tmp/hicfa/PID_10_5_0_3100.original.tif'

    overlay_processor = FormOverlay(work_dir=work_dir)
    img_dir = '/home/gbugaj/devio/pytorch-CycleGAN-and-pix2pix/results/hicfa_mask_pp/test_latest/images'
    img_dir = '/home/gbugaj/data/rms-asp/149512505/PID_1038_7836_0_149512505/'


    # for _path in glob.glob(os.path.join(img_dir,'*20220228.215.8047.63150004D.TIF*')  ):
    for _path in glob.glob(os.path.join(img_dir,'*.tif')  ):
    # for _path in glob.glob(os.path.join(img_dir,'*real*')):
        try:
            docId = _path.split('/')[-1].split('.')[0]
            docId = _path.split('/')[-1]
            print(f'DocumentId : {docId}')

            if os.path.exists(f'{img_dir}/clean/{docId}'):
                continue

            src_img_path = os.path.join(img_dir, _path)
            real,fake,blended = overlay_processor.segment(docId, src_img_path)

            stacked = np.hstack((real, fake, blended))
            print('Saving document')
            # image_process_path = f'/tmp/segmentation-mask/stacked_{docId}.jpg'
            image_process_path = f'/home/gbugaj/data/rms-asp/149512505/stacked/stacked_{docId}.png'
            imwrite(image_process_path, stacked)

            # image_process_path = f'/home/gbugaj/data/rms-asp/149495857/clean/{docId}.tif'
            image_process_path = f'/home/gbugaj/data/rms-asp/149512505/clean/{docId}' # This will have the .tif extension
            imwrite(image_process_path, blended)

        except Exception as ident:
            # raise ident
            print(ident)



    if False :
        print('OMR Detection')
        omr = OpticalMarkRecognition(work_dir=work_dir)
        
        for _path in glob.glob('/home/gbugaj/data/training/optical-mark-recognition/hicfa/task_checkboxes-2021_10_18_16_09_24-cvat_for_images_1.1/output_split/test/image/*.png'):
            try:
                kid = _path.split('/')[-1]
                results = omr.find_marks(kid, _path)
                print(results)
                break
            except Exception as ident:
                print(ident)
    
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
        img_path = '/tmp/form-segmentation/272943_0031516168746_001.tif/fields_debug/HCFA02/segmenation_real.png'

        snippet = cv2.imread(img_path)
        id = '272943_0031516168746_001'
        key = 'HCFA24'
        boxer = BoxProcessor(work_dir, cuda=False)
        boxes, img_fragments, lines, _= boxer.extract_bounding_boxes(id, 'field', snippet)

        icr = IcrProcessor(work_dir)
        icr.recognize(id, key, snippet, boxes, img_fragments, lines)

    if False:
        config_path = 'config-single.json'
        with open(config_path) as f:
            config = json.load(f)

        processor = FormProcessor(work_dir=work_dir, config=config, cuda=False)
        # for name in glob.glob('/home/greg/dev/assets-private/27ofStateFarm100/*.tif'):
        # for name in glob.glob('/home/gbugaj/data/private/HCFA-AllState/*.tif'):
        # for name in glob.glob('/home/greg/dataset/data-hipa/forms/hcfa-allstate/*.tif'):
        # for name in glob.glob('/media/gbugaj/XENSERVER-6/ImagesForPartAIssues/*.tif'):
        for name in glob.glob('/home/gbugaj/data/private/HCFA-StateFarm/*.tif'):
            try:
                print(name)
                results = processor.process(name)
                # break
            except Exception as ident:
                print(ident)

    if False:
        img_path = '/media/greg/XENSERVER-6/models-prod/bounding-boxes/PID_10_5_0_155085.tif.png'
        img_path = '/tmp/form-segmentation/samples/001.png'

        if not os.path.exists(img_path):
            raise Exception(f'File not found : {img_path}')

        id = img_path.split('/')[-1]
        debug_dir = ensure_exists(os.path.join(work_dir, id, 'work'))
        snippet = cv2.imread(img_path)

        icr = IcrProcessor(work_dir)
        boxer = BoxProcessor(work_dir, cuda=False)
        boxes, img_fragments, lines, _ = boxer.extract_bounding_boxes(id, 'field', snippet)
        # icr.recognize(id, 'HCFA05_PHONE', snippet, boxes, img_fragments, lines)

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
        img_path = '/home/greg/tmp/hicfa/PID_10_5_0_3112.original.tif'
        img_path = '/home/greg/tmp/hicfa/PID_10_5_0_3128.original.tif'
        work_dir = '/tmp/form-segmentation'

        snippet = cv2.imread(img_path)
        boxer = BoxProcessor(work_dir, cuda=False)
        boxer.process_full_extraction('id', snippet, .3)



