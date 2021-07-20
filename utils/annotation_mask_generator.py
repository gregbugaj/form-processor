
import os

import numpy as np
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
import random
import glob

from resize_image import resize_image
from split_dir import split_dir

# http://medialab.github.io/iwanthue/

def rgb(hex):
    h=hex.replace('#','')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

COLOR_1 = rgb('#7fd99d')
COLOR_2 = rgb('#a96df8')
COLOR_3 = rgb('#ff614e')
COLOR_4 = rgb('#016aa4')
COLOR_5 = rgb('#FFFF00')
COLOR_6 = rgb('#99624a')
COLOR_7 = rgb('#a1d743')
COLOR_8 = rgb('#dc199b')
COLOR_9 = rgb('#bf6900')
COLOR_10 = rgb('#510051')
COLOR_11 = rgb('#464646 ')
COLOR_12 = rgb('#770B20')
COLOR_13 = rgb('#DC143C')
COLOR_14 = rgb('#66669C')
COLOR_15 = rgb('#BE9999')
COLOR_16 = rgb('#FAAA1E')
COLOR_17 = rgb('#DCDC00')
COLOR_18 = rgb('#6B8E23')
COLOR_19 = rgb('#4682B4')
COLOR_20 = rgb('#DC143C')
COLOR_21 = rgb('#770B20')


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
        # sometimes(iaa.SaltAndPepper(0.03, per_channel=False)),
        # Blur each image with varying strength using
        # gaussian blur (sigma between 0 and 3.0),
        # average/uniform blur (kernel size between 2x2 and 7x7)
        # median blur (kernel size between 1x1 and 5x5).
       sometimes(iaa.OneOf([
            iaa.GaussianBlur((0, 2.0)),
            iaa.AverageBlur(k=(2, 7)),
            iaa.MedianBlur(k=(1, 3)),
        ])),

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

    accepted = dict()
    colormap = dict()

    if True:
        accepted["HCFA01a"] = True
        accepted["HCFA02"] = True
        accepted["HCFA03"] = True
        accepted["HCFA04"] = True
        accepted["HCFA06"] = True
        accepted["HCFA08"] = True

        accepted["HCFA05_ADDRESS"] = True
        accepted["HCFA05_CITY"] = True
        accepted["HCFA05_ZIP"] = True
        accepted["HCFA05_PHONE"] = True
        accepted["HCFA05_STATE"] = True

        accepted["HCFA07_ADDRESS"] = True
        accepted["HCFA07_CITY"] = True
        accepted["HCFA07_STATE"] = True
        accepted["HCFA07_ZIP"] =  True
        accepted["HCFA07_PHONE"] = True

        # accepted["HCFA21"] = True
        # accepted["HCFA24"] = True
        # accepted["HCFA33_BILLING"] = True

        colormap["HCFA01a"] = COLOR_1
        colormap["HCFA02"] = COLOR_2
        colormap["HCFA03"] = COLOR_3
        colormap["HCFA04"] = COLOR_4
        colormap["HCFA06"] = COLOR_5
        colormap["HCFA08"] = COLOR_16

        colormap["HCFA05_ADDRESS"] = COLOR_6
        colormap["HCFA05_CITY"] = COLOR_7
        colormap["HCFA05_STATE"] = COLOR_8
        colormap["HCFA05_ZIP"] =  COLOR_9
        colormap["HCFA05_PHONE"] = COLOR_10

        colormap["HCFA07_ADDRESS"] = COLOR_11
        colormap["HCFA07_CITY"] = COLOR_12
        colormap["HCFA07_STATE"] = COLOR_13
        colormap["HCFA07_ZIP"] =  COLOR_14
        colormap["HCFA07_PHONE"] = COLOR_15

    if False:
        accepted["HCFA33_BILLING"] = True
        accepted["HCFA33a_NPI"] = True
        accepted["HCFA33b_NONNPI"] = True

        accepted["HCFA32_SERVICE"] = True
        accepted["HCFA32a_NPI"] = True
        accepted["HCFA32b_NONNPI"] = True

        accepted["HCFA31"] = True
        accepted["HCFA25"] = True
        accepted["HCFA26"] = True
        accepted["HCFA27"] = True
        accepted["HCFA28"] = True
        accepted["HCFA29"] = True
        accepted["HCFA30"] = True
    
        # accepted["HCFA21"] = True
        # accepted["HCFA24"] = True
        print(colormap)

        colormap["HCFA33_BILLING"] = COLOR_1
        colormap["HCFA33a_NPI"] = COLOR_2
        colormap["HCFA33b_NONNPI"] = COLOR_3

        ## Only  HCFA32_SERVICE is present
        colormap["HCFA32_SERVICE"] = COLOR_4
        colormap["HCFA32a_NPI"] = COLOR_5
        colormap["HCFA32b_NONNPI"] = COLOR_6

        colormap["HCFA31"] = COLOR_7
        colormap["HCFA25"] = COLOR_8
        colormap["HCFA26"] = COLOR_9
        colormap["HCFA27"] = COLOR_10
        colormap["HCFA28"] = COLOR_15
        colormap["HCFA29"] = COLOR_12
        colormap["HCFA30"] = COLOR_13


    if False:
        accepted["HCFA24"] = True
        accepted["HCFA23"] = True
        accepted["HCFA22"] = True
        accepted["HCFA21"] = True
        accepted["HCFA20"] = True
        accepted["HCFA19"] = True
        accepted["HCFA18"] = True
        accepted["HCFA17a_"] = True
        accepted["HCFA17_NAME"] = True
        accepted["HCFA17a_NONNPI"] = True
        accepted["HCFA17b_NPI"] = True

        accepted["HCFA14"] = True
        accepted["HCFA15"] = True
        accepted["HCFA16"] = True

        # accepted["HCFA21"] = True
        # accepted["HCFA24"] = True
        print(colormap)

        colormap["HCFA24"] = COLOR_1
        colormap["HCFA23"] = COLOR_2
        colormap["HCFA22"] = COLOR_3
        colormap["HCFA21"] = COLOR_4
        colormap["HCFA20"] = COLOR_5
        colormap["HCFA19"] = COLOR_6
        colormap["HCFA18"] = COLOR_7
        colormap["HCFA17a_"] = COLOR_8
        colormap["HCFA17_NAME"] = COLOR_9
        colormap["HCFA17a_NONNPI"] = COLOR_10
        colormap["HCFA17b_NPI"] = COLOR_11
        colormap["HCFA14"] = COLOR_12
        colormap["HCFA15"] = COLOR_13
        colormap["HCFA16"] = COLOR_14


    # colormap["HCFA21"] = COLOR_16
    # colormap["HCFA24"] = COLOR_17
    # colormap["HCFA33_BILLING"] = COLOR_18

    for element in xmlTree.findall("image"):
        name = element.attrib['name']

        mappped_images = ''       
        if 'mapped_images' in element.attrib:
            mappped_images = element.attrib['mapped_images']
        
        polygons = element.findall("polygon")
        boxes = element.findall("box")
        filename = name.split('/')[-1]
    
        points = []
        colors = []

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
                    if label not in accepted:
                        continue
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
                if label not in accepted:
                    continue
                points.append(pts)
                colors.append(colormap[label])
        
        data['ds'].append({'name': filename, 'points': points, 'color':colors, 'mapped': 0 if mappped_images == '' else 1}) 

        for filename in mappped_images.split(',') :
            data['ds'].append({'name': filename, 'points': points, 'color':colors, 'mapped':0}) 

        index = index+1

    print('Total annotations : %s '% (len(data['ds'])))
    rsize = (1024, 1024)
    
    for row in data['ds']:
        filename = row['name']
        points_set = row['points']
        color_set = row['color']
        mapped = row['mapped']

        if mapped == 1:
            parts = filename.split('.')
            name = '.'.join(parts[:-1])
            img_dir = os.path.join(remap_dir, name)
            dir_src = img_dir
        
        filenames = os.listdir(dir_src)
        path = os.path.join(dir_src, filename)
        print(path)

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
                # mask_img = cv2.polylines(mask_img, [pts],  isClosed, (0,0,0), thickness) 

                overlay_img = cv2.polylines(overlay_img, [pts],  isClosed, color_display, thickness) 
                overlay_img = cv2.fillPoly(overlay_img, [pts], color_display) # white mask

                # path_dest_mask = os.path.join(dir_dest_mask,  "{}_{}.jpg".format(filename.split('.')[0], index))
                # imwrite(path_dest_mask, mask_img)

                index=index+1                
            # Following line overlays transparent over the image
            overlay_img = cv2.addWeighted(overlay_img, alpha, img.copy(), 1 - alpha, 0)
            
            mask_img = cv2.resize(mask_img, rsize)
            img = cv2.resize(img, rsize)

            path_dest_mask = os.path.join(dir_dest_mask,  "{}.jpg".format(filename.split('.')[0]))
            path_dest_img = os.path.join(dir_dest_image, "{}.jpg".format(filename.split('.')[0]))
            path_dest_overlay = os.path.join(dir_dest_overlay, "{}.png".format(filename.split('.')[0]))

            imwrite(path_dest_mask, mask_img)
            imwrite(path_dest_img, img)
            imwrite(path_dest_overlay, overlay_img)

            # Apply transformations to the image
            aug_images, aug_masks = augment_image(img, mask_img, pts, 1)

            assert len(aug_images) == len(aug_masks)

            # # Add originals
            # aug_images.append(img)
            # aug_masks.append(mask_img)

            index = 0
            for a_i, a_m in zip(aug_images, aug_masks):
                img = a_i
                mask_img = a_m
                path_dest_mask = os.path.join(dir_dest_mask,  "{}_{}.jpg".format(filename.split('.')[0], index))
                path_dest_img = os.path.join(dir_dest_image, "{}_{}.jpg".format(filename.split('.')[0], index))

                imwrite(path_dest_mask, mask_img)
                imwrite(path_dest_img, img)
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
        print(f'Remap dir : {img_dir}')
        if os.path.exists(img_dir):
            found_images = glob.glob(img_dir+'/*.png', recursive=False)
            names = [img.split('/')[-1] for img in found_images]
            element.set('mapped_images', ','.join(names))

    remap_file = os.path.join(dir_src, 'remap.xml')
    print(remap_file)
    with open(remap_file, 'wb') as f:
        xmlTree.write(f, encoding='utf-8')


def validate_annoations(cvat_annotation_file):
    print('Validating XML annotations')
    print("xml : {}".format(cvat_annotation_file))
    xmlTree = ET.parse(cvat_annotation_file)
    
    has_dupes = False
    for element in xmlTree.findall("image"):
        name = element.attrib['name']
        polygons = element.findall("polygon")
        boxes = element.findall("box")
        labelmap = dict()
        for polygon_node in polygons:
            label = polygon_node.attrib['label']
            if label in labelmap:
                msg = f'Duplicate label detected  [{name} : {label}]'
                print (msg)
                has_dupes = True
            labelmap[label] = True
        
    if has_dupes:
        raise Exception('Validation failed')
    print('Validation completed')        

if __name__ == '__main__':
    root_src = '../assets-private/cvat/task_3100-3199-2021_05_26_23_59_41-cvat'
    # root_src = '../assets-private/cvat/task_3200-3299-2021_05_27_00_43_52-cvat'
    # root_src = '../assets-private/cvat/task_3300-3399-2021_05_27_14_23_55-cvat' ## TEST SET
    # root_src = '../assets-private/cvat/task_3400-3499-2021_05_27_14_28_26-cvat'

    root_src = '../assets-private/cvat/task_redactedAllFields'
    root_src = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/task_redacted-wide-2021_06_23_19_21_27-cvat'
    # root_src = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/task_redacted_20-2021_06_22_18_19_51-cvat'
    root_src = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/task_redacted-wide-2021_06_24_22_14_17-cvat_box33'
    root_src = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/task_redacted_20-2021_06_22_18_19_51-cvat'
    root_src = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/hicfa-forms'

    dir_src = os.path.join(root_src, 'images')
    dir_dest  = os.path.join(root_src, 'output')
    dir_dest_split  = os.path.join(root_src, 'output_split')
    cvat_annotation_file=os.path.join(root_src, 'annotations.xml') 
    remap_src = os.path.join(root_src, 'remap')
    
    validate_annoations(cvat_annotation_file)

    remap(root_src , dir_dest, cvat_annotation_file, remap_src)

    cvat_annotation_file=os.path.join(root_src, 'remap.xml') 

    create_mask(dir_src, dir_dest, cvat_annotation_file, remap_src)
    split_dir(dir_dest, dir_dest_split)

# python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/hicfa_form --output_dir ./datasets/hicfa_form/ready