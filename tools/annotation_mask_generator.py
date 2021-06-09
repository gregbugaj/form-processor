
import os

import numpy as np
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
import random

from resize_image import resize_image

def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)

def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def fourChannels(img):
  height, width, channels = img.shape
  if channels < 4:
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return new_img

  return img

def create_mask(dir_src, dir_dest, cvat_annotation_file):
    print("dir_src         = {}".format(dir_src))
    print("dir_dest        = {}".format(dir_dest))
    print("xml : {}".format(cvat_annotation_file))

    colormap=dict()
    colormap['HCFA21'] = (0, 255, 0, 150)

    dir_dest_image = os.path.join(dir_dest, 'image')
    dir_dest_mask = os.path.join(dir_dest, 'mask')
    dir_dest_overlay = os.path.join(dir_dest, 'overlay')

    ensure_exists(dir_dest_image)
    ensure_exists(dir_dest_mask)
    ensure_exists(dir_dest_overlay)

    data = {}  
    data['ds'] = []  
    strict = False
    xmlTree = ET.parse(cvat_annotation_file)

    for element in xmlTree.findall("image"):
        name = element.attrib['name']
        polygons = element.findall("polygon")
        boxes = element.findall("box")
        points = []
        filename = name.split('/')[-1]

        print("pol, box = {}, {}".format(len(polygons), len(boxes)))
        if len(boxes) > 0:
            for box_node in boxes:
                    label = box_node.attrib['label']
                    xtl = float(box_node.attrib['xtl'])
                    ytl = float(box_node.attrib['ytl'])
                    xbr = float(box_node.attrib['xbr'])
                    ybr = float(box_node.attrib['ybr'])
                    w = xbr - xtl
                    h = ytl - ybr
                    xtr = xtl + w 
                    ytr = ytl
                    xbl = xtl 
                    ybl = ytl - h 
                    print('filename = {} label = {}'.format(filename, label))
                    points = ['{},{}'.format(xtl, ytl), '{},{}'.format(xtr, ytr), '{},{}'.format(xbr, ybr), '{},{}'.format(xbl, ybl)]
                    data['ds'].append({'name': filename, 'points': points, 'color':colormap[label]}) 
                    break
        else:
            for polygon_node in polygons:
                label = polygon_node.attrib['label']
                points = polygon_node.attrib['points'].split(';')
                
                size = len(points)
                if strict and size > 0 and size != 4:
                    raise ValueError("Expected 4 point got : %s " %(size))
                if size  == 4:
                    data['ds'].append({'name': filename, 'points': points, 'color':colormap[label]}) 
                break

    filenames = os.listdir(dir_src)

    print('Total annotations : %s '% (len(data['ds'])))
    print('Total files : %s '% (len(filenames)))
    
    for row in data['ds']:
        print(row)
        filename = row['name']
        points = row['points']
        color_display = row['color']

        if filename in filenames:
            path = os.path.join(dir_src, filename)
            img = cv2.imread(path)
            points = [[float(seg) for seg in pt.split(',')] for pt in points]
            
            # Polygon corner points coordinates 
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            h, w, c = img.shape
            
            isClosed = True
            thickness = 1
            
            mask_img = np.ones((h, w, c), np.uint8) * 255 # white canvas
            mask_img = cv2.polylines(mask_img, [pts],  isClosed, color_display, thickness) 
            mask_img = cv2.fillPoly(mask_img, [pts], color_display) # white mask
            
            alpha = 0.5  # Transparency factor.

            overlay_img = np.ones((h, w, c), np.uint8) * 255 # white canvas
            # overlay_img = fourChannels(overlay_img)
            overlay_img = cv2.polylines(overlay_img, [pts],  isClosed, color_display, thickness) 
            overlay_img = cv2.fillPoly(overlay_img, [pts], color_display) # white mask

            # Following line overlays transparent over the image
            overlay_img = cv2.addWeighted(overlay_img, alpha, img.copy(), 1 - alpha, 0)

            rsize = (1024, 1024)
            mask_img = resize_image(mask_img, rsize)
            img = resize_image(img, rsize)
            overlay_img = resize_image(overlay_img, rsize)

            path_dest_mask = os.path.join(dir_dest_mask,  "{}.png".format(filename.split('.')[0]))
            path_dest_img = os.path.join(dir_dest_image, "{}.png".format(filename.split('.')[0]))
            path_dest_overlay = os.path.join(dir_dest_overlay, "{}.png".format(filename.split('.')[0]))

            imwrite(path_dest_mask, mask_img)
            imwrite(path_dest_img, img)
            imwrite(path_dest_overlay, overlay_img)
            # break

if __name__ == '__main__':
    # parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    # parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
  
    root_src = './assets-private/task_3100-3199-2021_05_26_23_59_41-cvat'
    root_src = './assets-private/task_3200-3299-2021_05_27_00_43_52-cvat'
    root_src = './assets-private/task_3300-3399-2021_05_27_14_23_55-cvat'
    root_src = './assets-private/task_3400-3499-2021_05_27_14_28_26-cvat'
    dir_src = os.path.join(root_src, 'images')
    dir_dest  = os.path.join(root_src, 'output')
    cvat_annotation_file=os.path.join(root_src, 'annotations.xml') 

    create_mask(dir_src , dir_dest, cvat_annotation_file)
