
import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

from utils.image_utils import paste_fragment
from boxes.box_processor import BoxProcessor
from form.icr_processor import IcrProcessor
from form.segmenter import FormSegmeneter
from utils.utils import current_milli_time
from utils.visualize import visualize
from form.form_alignment import FormAlignment

from boxes.box_processor import BoxProcessor
from form.icr_processor import IcrProcessor
from utils.utils import current_milli_time


def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        raise ident
        print(ident)

def blend_to_text(real_img, fake_img):
    """
        Blend real and fake(generated) images together to generate extracted text
    """

    print(f'real_img : {real_img}')
    print(f'fake_img : {fake_img}')

    real = cv2.imread(real_img, cv2.IMREAD_GRAYSCALE)
    fake = cv2.imread(fake_img, cv2.IMREAD_GRAYSCALE)

    blended_img = cv2.bitwise_or(real, fake)
    blended_img[blended_img >= 120] = [255]

    return blended_img


def process(img_original_path, img_processed_path):
    work_dir = '/tmp/form-segmentation'

    snippet = cv2.imread(img_processed_path)
    id = img_original_path.split('/')[-1] # Segmenter requires the ID to be a filename 
    key = id

    if True:
        boxer = BoxProcessor(work_dir, cuda=False)
        boxes, img_fragments, lines, _ = boxer.extract_bounding_boxes(
            id, 'field', snippet)

        icr = IcrProcessor(work_dir)
        icr.recognize(id, key, snippet, boxes, img_fragments, lines)

    return 
    m0 = current_milli_time()
    segmenter = FormSegmeneter(work_dir)
    m1 = current_milli_time()-m0

    print('Form processor initialized in {} ms'.format(m1))
    m0 = current_milli_time()
    seg_fragments, img, segmask = segmenter.segment(id, img_original_path)
    m1 = current_milli_time() - m0
    print('Time {} ms'.format(m1))

    blended_segmentation = cv2.addWeighted(snippet, .5,  segmask, .7, 0)
    imwrite('/tmp/segmentation-mask/blended_img_segmentation.png', blended_segmentation)


def eval_dir(img_dir):
    import glob
    work_dir='/tmp/segmentation-mask'
    dir_out='/tmp/segmentation-mask/all'

    for _path in glob.glob(os.path.join(img_dir,'*_real.png')):
        kid = _path.split('/')[-1]
        origina_img = os.path.join(img_dir, _path)
        fake_img = os.path.join(img_dir, _path.replace('_real', '_fake'))

        print(origina_img)
        print(fake_img)
        
        real = cv2.imread(origina_img, cv2.IMREAD_GRAYSCALE)
        fake = cv2.imread(fake_img, cv2.IMREAD_GRAYSCALE)

        blended_img = blend_to_text(origina_img, fake_img)

        stacked = np.hstack((real, fake, blended_img))
        image_process_path = f'/tmp/segmentation-mask/stacked_{kid}.tif'
        imwrite(image_process_path, stacked)

        # process('/tmp/segmentation-mask/PID_10_5_0_155061_real.png', image_process_path)
        # process('/tmp/segmentation-mask/PID_10_5_0_154648_real.png', '/tmp/segmentation-mask/PID_10_5_0_154648_real.png')
        process('/tmp/segmentation-mask/PID_10_5_0_155061_real.png', '/tmp/segmentation-mask/PID_10_5_0_155061_real.png')

        break

def process(img_dir):
    import glob
    work_dir='/tmp/segmentation-mask'
    dir_out='/tmp/segmentation-mask/all'
    segmenter = FormAlignment(work_dir)

    for _path in glob.glob(os.path.join(img_dir,'*.*')):
        try:
            kid = _path.split('/')[-1]
            print(kid)
            img_path = os.path.join(img_dir, _path)
            image = cv2.imread(img_path)
            m0 = current_milli_time()
            
            m0 = current_milli_time()
            segmask = segmenter.align(kid, image)
            
            segmask_path = os.path.join(dir_out, "%s.png" % (kid))
            print(segmask_path)
            
            imwrite(segmask_path, segmask)
            m1 = current_milli_time()- m0
            print('Time {} ms'.format(m1))

            # break
        except Exception as ident:
            print(ident)


if __name__ == '__main__':

    eval_dir('/home/gbugaj/devio/pytorch-CycleGAN-and-pix2pix/results/hicfa_mask_pp/test_latest/images')

    # process(os.path.expanduser('~/tmp/hicfa_mask/blended.png'))
    # process(os.path.expanduser('~/tmp/hicfa_mask/final.png'))
    # process(
    #     '/home/gbugaj/devio/pytorch-CycleGAN-and-pix2pix/results/hicfa_mask_pp_1/test_latest/images/PID_10_5_0_3101.original_real_real.png',
    #     os.path.expanduser('/tmp/segmentation-mask/blended_img.png'))

    if False:
        main(os.path.expanduser('/home/gbugaj/devio/pytorch-CycleGAN-and-pix2pix/results/hicfa_mask_pp_1/test_latest/images/PID_10_5_0_3101.original_real_real.png'),
             os.path.expanduser(
                 '/tmp/segmentation-mask/PID_10_5_0_3101.original_fake.png'),
             )
