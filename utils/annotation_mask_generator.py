
import os

import numpy as np
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
import random
import glob

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
        
    # # add some text to emulate text writing on document
    # x1 = pts[0][0][0]
    # x2 = pts[1][0][0]
    # y1 = pts[0][0][1]
    # y2 = pts[-1][0][1]

    # tx1 = get_random_string(12)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    
    # upper = int(random.uniform(0, .4) * 10)
    # for i in range(upper):
    #     cv2.putText(img, tx1, (int((x2-x1) // 3 * (1 + random.uniform(1, 2))) , y1 - y1 // (2 + i)), font, random.uniform(.4, .9), (0, 0, 0),2, cv2.LINE_AA)
    seq_shared = iaa.Sequential([
        
        # sometimes(iaa.Affine(
        #     scale={"x": (0.8, 1.0), "y": (0.8, 1.0)},
        #     cval=(255),
        # )),

        sometimes(iaa.Affine(
            shear=(-2, 2),
            cval=(255),
        )),

        sometimes(iaa.Rotate((-2, 2), cval=(255)))

        # iaa.CropAndPad(
        #     percent=(-0.07, 0.2),
        #     # pad_mode=ia.ALL,
        #     pad_mode=["edge"],
        #     pad_cval=(150, 200)
        # )
    ])

    seq = iaa.Sequential([
        # sometimes(iaa.SaltAndPepper(0.03, per_channel=False)),
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


def create_mask(dir_src, dir_dest, cvat_annotation_file, remap_dir):
    print("dir_src         = {}".format(dir_src))
    print("dir_dest        = {}".format(dir_dest))
    print("xml : {}".format(cvat_annotation_file))

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

    # extract color map
    # h = input('Enter hex: ').lstrip('#')
    # print('RGB =', tuple(int(h[i:i+2], 16) for i in (0, 2, 4)))

    # <name>HCFA01</name>
    # <color>#2900f8</color>

    colormap=dict()
    # colormap['HCFA21'] = (0, 255, 0, 150)
    for element in xmlTree.findall("meta/task/labels/label"):
        name_node = element.find("name")
        color_node = element.find("color")
        name = name_node.text
        color = color_node.text.lstrip('#')
        color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        colormap[name] = color
        # print(name, color)
        # print()

    index = 0 
    for element in xmlTree.findall("image"):
        name = element.attrib['name']
        mappped_images = element.attrib['mapped_images']
        print(mappped_images)
        polygons = element.findall("polygon")
        boxes = element.findall("box")
        points = []
        colors = []
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
                    pts = ['{},{}'.format(xtl, ytl), '{},{}'.format(xtr, ytr), '{},{}'.format(xbr, ybr), '{},{}'.format(xbl, ybl)]
                    points.append(pts)
                    colors.append(colormap[label])
                    break
        # else:
        for polygon_node in polygons:
            label = polygon_node.attrib['label']
            pts = polygon_node.attrib['points'].split(';')
            
            size = len(pts)
            if strict and size > 0 and size != 4:
                raise ValueError("Expected 4 point got : %s " %(size))
            if size  == 4:
                points.append(pts)
                colors.append(colormap[label])
        
        data['ds'].append({'name': filename, 'points': points, 'color':colors, 'mapped':1}) 

        for filename in mappped_images.split(',') :
            data['ds'].append({'name': filename, 'points': points, 'color':colors, 'mapped':0}) 
        index = index+1

    filenames = os.listdir(dir_src)

    print('Total annotations : %s '% (len(data['ds'])))
    print('Total files : %s '% (len(filenames)))
    
    rsize = (512, 512)

    for row in data['ds']:
        print(row)
        filename = row['name']
        points_set = row['points']
        color_set = row['color']
        mapped = row['mapped']

        if mapped == 1:
            parts = element.attrib['name'].split('.')
            name = '.'.join(parts[:-1])
            img_dir = os.path.join(remap_dir, name)
            dir_src = img_dir
            filenames = os.listdir(dir_src)

        if filename in filenames:
            path = os.path.join(dir_src, filename)
            img = cv2.imread(path)
            h, w, c = img.shape
            isClosed = True
            thickness = 6

            index = 0 
            alpha = 0.5  # Transparency factor.
            mask_img = np.ones((h, w, c), np.uint8) * 255 # white canvas    
            overlay_img = np.ones((h, w, c), np.uint8) * 255 # white canvas
            
            for points, color_display in zip(points_set, color_set):
                color_display = color_set[index]
                points = [[float(seg) for seg in pt.split(',')] for pt in points]
                # Polygon corner points coordinates 
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))

                
                mask_img = cv2.fillPoly(mask_img, [pts], color_display) # white mask
                mask_img = cv2.polylines(mask_img, [pts],  isClosed, (0,0,0), thickness) 

                overlay_img = cv2.polylines(overlay_img, [pts],  isClosed, color_display, thickness) 
                overlay_img = cv2.fillPoly(overlay_img, [pts], color_display) # white mask

                # path_dest_mask = os.path.join(dir_dest_mask,  "{}_{}.jpg".format(filename.split('.')[0], index))
                # imwrite(path_dest_mask, mask_img)

                index=index+1                
            # # overlay_img = fourChannels(overlay_img)

            # Following line overlays transparent over the image
            overlay_img = cv2.addWeighted(overlay_img, alpha, img.copy(), 1 - alpha, 0)
            
            mask_img = cv2.resize(mask_img, rsize)
            img = cv2.resize(img, rsize)

            # mask_img = resize_image(mask_img, rsize)
            # img = resize_image(img, rsize)
            
            # overlay_img = resize_image(overlay_img, rsize)

            path_dest_mask = os.path.join(dir_dest_mask,  "{}.jpg".format(filename.split('.')[0]))
            path_dest_img = os.path.join(dir_dest_image, "{}.jpg".format(filename.split('.')[0]))
            path_dest_overlay = os.path.join(dir_dest_overlay, "{}.png".format(filename.split('.')[0]))

            imwrite(path_dest_mask, mask_img)
            imwrite(path_dest_img, img)
            imwrite(path_dest_overlay, overlay_img)

            # Apply transformations to the image
            aug_images, aug_masks = augment_image(img, mask_img, pts, 2)

            assert len(aug_images) == len(aug_masks)
            # Add originals
            aug_images.append(img)
            aug_masks.append(mask_img)

            index = 0
            for a_i, a_m in zip(aug_images, aug_masks):
                img = a_i
                mask_img = a_m
                path_dest_mask = os.path.join(dir_dest_mask,  "{}_{}.jpg".format(filename.split('.')[0], index))
                path_dest_img = os.path.join(dir_dest_image, "{}_{}.jpg".format(filename.split('.')[0], index))
                path_dest_overlay = os.path.join(dir_dest_overlay, "{}_{}.png".format(filename.split('.')[0], index))

                imwrite(path_dest_mask, mask_img)
                imwrite(path_dest_img, img)
                imwrite(path_dest_overlay, overlay_img)

                index = index + 1

def remap(dir_src, dir_dest, cvat_annotation_file, remap_dir):
    print('Remaping XML annotations to image')

    print("dir_src         = {}".format(dir_src))
    print("dir_dest        = {}".format(dir_dest))
    print("xml : {}".format(cvat_annotation_file))

    ensure_exists(dir_dest)
    xmlTree = ET.parse(cvat_annotation_file)

    for element in xmlTree.findall("image"):
        parts = element.attrib['name'].split('.')
        name = '.'.join(parts[:-1])
        img_dir = os.path.join(remap_dir, name)
        print(img_dir)
        if os.path.exists(img_dir):
            found_images = glob.glob(img_dir+'/*.png', recursive=False)
            names = [img.split('/')[-1] for img in found_images]
            element.set('mapped_images', ','.join(names))


    remap_file = os.path.join(dir_src, 'remap.xml')
    print(remap_file)
    with open(remap_file, 'wb') as f:
        xmlTree.write(f, encoding='utf-8')
        
    print(xmlTree)
if __name__ == '__main__':

    root_src = '../assets-private/cvat/task_3100-3199-2021_05_26_23_59_41-cvat'
    # root_src = '../assets-private/cvat/task_3200-3299-2021_05_27_00_43_52-cvat'
    # root_src = '../assets-private/cvat/task_3300-3399-2021_05_27_14_23_55-cvat' ## TEST SET
    # root_src = '../assets-private/cvat/task_3400-3499-2021_05_27_14_28_26-cvat'
    root_src = '../assets-private/cvat/task_3000-3099-2021_06_02_21_43_46-cvat-ALL'
    root_src = 'e'


    dir_src = os.path.join(root_src, 'images')
    dir_dest  = os.path.join(root_src, 'output')
    cvat_annotation_file=os.path.join(root_src, 'annotations.xml') 
    cvat_annotation_file=os.path.join(root_src, 'remap.xml') 
    remap_src = os.path.join(root_src, 'remap')
    # remap(root_src , dir_dest, cvat_annotation_file, remap_src)
    create_mask(dir_src , dir_dest, cvat_annotation_file, remap_src)