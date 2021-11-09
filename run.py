
import os
import sys
import argparse
import json

from form.numpyencoder import NumpyEncoder
from form.form_processor_deux import FormProcessor
from preprocess import imwrite
from utils.image_utils import read_image

from utils.utils import current_milli_time, ensure_exists, make_power_2, make_power_2_cv2

# logging
import logging
from logging import handlers

LOGFILE = 'processor.log'
log = logging.getLogger('FP.Segmenter')
log.setLevel(logging.INFO)
format = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
log.addHandler(ch)

fh = handlers.RotatingFileHandler(
    LOGFILE, maxBytes=(1048576 * 5), backupCount=7)
fh.setFormatter(format)
log.addHandler(fh)


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Run evaluator')
    parser.add_argument('--img', dest='img_src',
                        help='Image to evaluate', default='data/form.png', type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='Item result directory', default='./data/output', type=str)
    parser.add_argument('--work_dir', dest='debug',
                        help='Debug directory results', default='./data/debug', type=str)
    parser.add_argument('--config', dest='config',
                        help='Field configuration file', default='./config.json', type=str)

    return parser.parse_args()


def main(config_path, img_path, output_dir, work_dir, cuda):
    print('Main')
    print(f'cuda        = {cuda}')
    print(f'config_path = {config_path}')
    print(f'img_path    = {img_path}')
    print(f'output_dir  = {output_dir}')
    print(f'work_dir    = {work_dir}')

    if not os.path.exists(img_path):
        raise Exception(f'image file not found : {img_path}')

    if not os.path.exists(config_path):
        raise Exception(f'config file not found : {config_path}')

    with open(config_path) as f:
        config = json.load(f)
   
    processor = FormProcessor(work_dir=work_dir, config=config, cuda=cuda)
    results = processor.process(img_path)
    file_path = os.path.join(output_dir, "results.json")

    print(f'Saving results to : {file_path}')
    with open(file_path, 'w') as f:
        json.dump(results, f,  sort_keys=True,  separators=(
            ',', ': '), ensure_ascii=False, indent=4, cls=NumpyEncoder)


if __name__ == '__main__':
    args = parse_args()
    args.img_src = '/media/greg/XENSERVER-6/27ofStateFarm100/272943_0031516168746_001.tif'
    args.img_src = '/home/gbugaj/data/private/HCFA-StateFarm/247611_0031267713836_001.tif'
    args.work_dir = '/tmp/form-segmentation'

    args.config = './config.json'
    args.config = './config-single.json'

    img_path = args.img_src
    work_dir = args.work_dir
    config_path = args.config

    id = img_path.split('/')[-1]
    output_dir = ensure_exists(os.path.join(work_dir, id, 'result'))
    main(config_path=config_path, img_path=img_path,
         output_dir=output_dir, work_dir=work_dir, cuda=False)
