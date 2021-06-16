
from operator import ne
import os
import cv2
import argparse
from form.processor import process
from form.segmenter import FormSegmeneter


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Run evaluator')
    parser.add_argument('--network-param', dest='network_param', help='Network parameter filename',default='unet_best.params', type=str)
    parser.add_argument('--img', dest='img_src', help='Image to evaluate', default='data/clean.png', type=str)
    parser.add_argument('--output', dest='dir_out', help='Output directory evaluate', default='./data/debug', type=str)
    parser.add_argument('--debug', dest='debug', help='Debug results', default=False, type=bool)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.network_param = './models/form_pix2pix/latest_net_G.pth'
    img_path ='./assets/forms-seg/001_fake.png'
    # img_path ='./assets/forms-seg/baseline.jpg'
    # img_path ='./assets/forms-seg/001_fake_green.jpg'
    
    segmenter = FormSegmeneter(network="")
    segmenter.process(img_path)    

    if False:
        # dev
        args.img_src = './assets-private/forms-raw/PID_10_5_0_94371.tif'
        args.img_src = './assets-private/forms-raw/requested/PID_10_5_0_94372.tif'
        args.dir_out = './assets-gen/cleaned-examples/set-001/test'
        args.debug = False
        
        img_src = args.img_src 
        dir_out = args.dir_out 
        network_parameters = args.network_param
        
        process(img_path = img_src, dir_out = dir_out, network_parameters = network_parameters)
