
import os

import numpy as np
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
import random
import glob
import json

from resize_image import resize_image
from split_dir import split_dir

# http://medialab.github.io/iwanthue/

def rgb(hex):
    h=hex.replace('#','')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
 
def imwrite(path, img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

def augment_image(img, mask, pts, count):
    import random
    import string
    """Augment imag and mask"""
    import imgaug as ia
    import imgaug.augmenters as iaa
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    def get_random_string(length):
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str

    # for i in range(upper):
    #     cv2.putText(img, tx1, (int((x2-x1) // 3 * (1 + random.uniform(1, 2))) , y1 - y1 // (2 + i)), font, random.uniform(.4, .9), (0, 0, 0),2, cv2.LINE_AA)
    seq_shared = iaa.Sequential([
        
        # sometimes(iaa.Affine(
        #     scale={"x": (0.8, 1.0), "y": (0.8, 1.0)},
        #     cval=(255),
        # )),

        sometimes(iaa.Affine(
            shear=(-1, 1),
            cval=(255),
        )),

        sometimes(iaa.Rotate((-1, 1), cval=(255)))
    ])

    seq = iaa.Sequential([
        sometimes(iaa.SaltAndPepper(0.03, per_channel=False)),

        # Blur each image with varying strength using
        # gaussian blur (sigma between 0 and 3.0),
        # average/uniform blur (kernel size between 2x2 and 7x7)
        # median blur (kernel size between 1x1 and 5x5).
    #    sometimes(iaa.OneOf([
    #         iaa.GaussianBlur((0, 2.0)),
    #         iaa.AverageBlur(k=(2, 7)),
    #         iaa.MedianBlur(k=(1, 3)),
    #     ])),

    ], random_order=True)

    masks = []
    images = [] 
    for i in range(count):
        seq_shared_det = seq_shared.to_deterministic()
        image_aug = seq(image = img)
        image_aug = seq_shared_det(image = image_aug)
        mask_aug = seq_shared_det(image = mask)

        masks.append(mask_aug)
        images.append(image_aug)
        # cv2.imwrite('/tmp/imgaug/%s.png' %(i), image_aug)
        # cv2.imwrite('/tmp/imgaug/%s_mask.png' %(i), mask_aug)

    return images, masks

def create_mask(dir_src, dir_dest, cvat_annotation_file):
    print("dir_src         = {}".format(dir_src))
    print("dir_dest        = {}".format(dir_dest))
    print("annotations     = {}".format(cvat_annotation_file))

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

    index = 0 

    for element in xmlTree.findall("image"):
        name = element.attrib['name']
        polygons = element.findall("polygon")
        boxes = element.findall("box")
        filename = name.split('/')[-1]

        points = []
        colors = []
        labels = []
        colormap = dict()

        colormap['checked'] = (0, 255, 0)
        colormap['unchecked'] = (255, 0, 0)
        
        if len(polygons)  == 0 and len(boxes) == 0:
            continue

        print("pol, box = {}, {}".format(len(polygons), len(boxes)))
        if len(boxes) > 0:
            for box_node in boxes:
                    label = box_node.attrib['label']

                    # if label == 'checked':
                    #     continue

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
                    pts = ['{},{}'.format(xtl, ytl), '{},{}'.format(xtr, ytr), '{},{}'.format(xbr, ybr), '{},{}'.format(xbl, ybl)]
                    
                    points.append(pts)
                    colors.append(colormap[label])
                    labels.append(label)

        # else:
        for polygon_node in polygons:
            label = polygon_node.attrib['label']
            pts = polygon_node.attrib['points'].split(';')
            size = len(pts)

            if strict and size > 0 and size != 4:
                raise ValueError("Expected 4 point got : %s " %(size))
            if size  == 4:
                points.append(pts)
                labels.append(label)
        
        if len(points) == 0:
            continue
        
        row = {'name': filename, 'points': points, 'color':colors, "labels":labels}
        data['ds'].append(row) 
        index = index+1

    print('Total annotations : %s '% (len(data['ds'])))
    rsize = (1024, 1536)
    # rsize = (512, 768)
    # rsize = (768, 1024)
    
    for row in data['ds']:
        filename = row['name']
        points_set = row['points']
        label_set = row['labels']
        color_set = row['color']
        filenames = os.listdir(dir_src)
        path = os.path.join(dir_src, filename)
        print(path)

        if filename in filenames:
            path = os.path.join(dir_src, filename)
            img = cv2.imread(path)
            h, w, c = img.shape
            isClosed = True
            thickness = 2

            index = 0 
            alpha = 0.5  # Transparency factor.
            mask_img = np.ones((h, w, c), np.uint8) * 255 # white canvas    
            overlay_img = np.ones((h, w, c), np.uint8) * 255 # white canvas
            
            print(points_set)
            for points  in points_set:
                # color_display = (0,255,0)
                color_display = color_set[index]
                label = label_set[index]
                points = [[float(seg) for seg in pt.split(',')] for pt in points]
                # Polygon corner points coordinates 
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                mask_img = cv2.fillPoly(mask_img, [pts], color_display) # white mask
                # mask_img = cv2.polylines(mask_img, [pts],  isClosed, (0,0,0), thickness) 

                overlay_img = cv2.polylines(overlay_img, [pts],  isClosed, color_display, thickness) 
                overlay_img = cv2.fillPoly(overlay_img, [pts], color_display) # white mask
                # 4,1,2
                org = [pts[3][0][0]+10, pts[3][0][1]-10]
                txt_label = '{id} : {key}'.format(id=index, key=label)
                _ = cv2.putText(overlay_img, txt_label, org, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                index=index+1                
            # Following line overlays transparent over the image
            overlay_img = cv2.addWeighted(overlay_img, alpha, img.copy(), 1 - alpha, 0)
            
            # DO NOT RESIZE
            mask_img = cv2.resize(mask_img, rsize)
            img = cv2.resize(img, rsize)

            path_dest_mask = os.path.join(dir_dest_mask,  "{}.png".format(filename.split('.')[0]))
            path_dest_img = os.path.join(dir_dest_image, "{}.png".format(filename.split('.')[0]))
            path_dest_overlay = os.path.join(dir_dest_overlay, "{}.png".format(filename.split('.')[0]))

            imwrite(path_dest_mask, mask_img)
            imwrite(path_dest_img, img)
            imwrite(path_dest_overlay, overlay_img)

            # Apply transformations to the image
            aug_images, aug_masks = augment_image(img, mask_img, pts, 10)

            assert len(aug_images) == len(aug_masks)

            # # Add originals
            # aug_images.append(img)
            # aug_masks.append(mask_img)

            index = 0
            for a_i, a_m in zip(aug_images, aug_masks):
                img = a_i
                mask_img = a_m
                path_dest_mask = os.path.join(dir_dest_mask,  "{}_{}.png".format(filename.split('.')[0], index))
                path_dest_img = os.path.join(dir_dest_image, "{}_{}.png".format(filename.split('.')[0], index))

                imwrite(path_dest_mask, mask_img)
                imwrite(path_dest_img, img)
                index = index + 1
            
if __name__ == '__main__':
    root_src = '/home/gbugaj/data/training/optical-mark-recognition/hicfa/task_checkboxes-2021_10_18_16_09_24-cvat_for_images_1.1'
    # root_src = '/home/greg/dataset/cvat/task_checkboxes_2021_10_18'

    dir_src = os.path.join(root_src, 'images', 'HCFA-AllState')
    dir_dest = os.path.join(root_src, 'output')
    dir_dest_split = os.path.join(root_src, 'output_split')
    cvat_annotation_file=os.path.join(root_src, 'annotations.xml') 

    create_mask(dir_src, dir_dest, cvat_annotation_file)
    split_dir(dir_dest, dir_dest_split)
