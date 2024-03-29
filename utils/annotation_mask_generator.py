
import os
import string

import numpy as np
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
import random
import glob
import json
from PIL import ImageFont, ImageDraw, Image  

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


def drawTrueTypeTextOnImage(cv2Image, text, xy, size):
    """
    Print True Type fonts using PIL and convert image back into OpenCV
    """
    # print(f'text : {xy}, {text}')
    # Pass the image to PIL  
    pil_im = Image.fromarray(cv2Image)  
    draw = ImageDraw.Draw(pil_im)  
    # use a truetype font  
    # /usr/share/fonts/truetype/freefont/FreeSerif.ttf"
    # FreeMono.ttf
    # FreeSerif.ttf
    # "FreeSerif.ttf","FreeSerifBold.ttf",
    # fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf"]) 
    # fontPath = os.path.join("/usr/share/fonts/truetype/freefont", fontFace)

    # fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf", "oldfax.ttf", "FreeMonoBold.ttf", "FreeSans.ttf", "Old_Rubber_Stamp.ttf"]) 
    fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf", "FreeMonoBold.ttf", "FreeSans.ttf"]) 
    fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf", "FreeSans.ttf", "ColourMePurple.ttf", "Pelkistettyatodellisuutta.ttf" ,"SpotlightTypewriterNC.ttf"]) 
    fontPath = os.path.join("./assets/fonts/truetype", fontFace)
    fontPath = os.path.join("/home/greg/dev/unet-denoiser/assets/fonts/truetype", fontFace)

    font = ImageFont.truetype(fontPath, size)
    size_width, size_height = draw.textsize(text, font)

    # text has to be within the bounds otherwise return same image
    x = xy[0]
    y = xy[1]
    
    img_h = cv2Image.shape[0]
    img_w = cv2Image.shape[1]

    adj_y = y + size_height
    adj_w = x + size_width
    
    # print(f'size : {img_h},  {adj_y},  {size_width}, {size_height} : {xy}')
    if adj_y > img_h or adj_w > img_w:
        return False, cv2Image, (0, 0)

    draw.text(xy, text, font=font,  fill='#000000')  
    # Make Numpy/OpenCV-compatible version
    cv2Image = np.array(pil_im)
    return True, cv2Image, (size_width, size_height)

def get_word_list():
    f = open('text.txt', encoding='utf-8', mode="r")
    text = f.read()
    f.close()
    lines_list = str.split(text, '\n')
    while '' in lines_list:
        lines_list.remove('')

    lines_word_list = [str.split(line) for line in lines_list]
    words_list = [words for sublist in lines_word_list for words in sublist]

    return words_list

words_list = get_word_list()
word_count = 0

def get_text():
    global word_count, words_list
    # text to be printed on the blank image
    num_words = np.random.randint(1, 6)
    
    # renew the word list in case we run out of words 
    if (word_count + num_words) >= len(words_list):
        print('===\nrecycling the words_list')
        words_list = get_word_list() 
        word_count = 0

    print_text = ''
    for _ in range(num_words):
        index = np.random.randint(0, len(words_list))
        print_text += str.split(words_list[index])[0] + ' '
        #print_text += str.split(words_list[word_count])[0] + ' '
        word_count += 1
    print_text = print_text.strip() # to get rif of the last space
    return print_text

def print_lines(img):
    def getUpperOrLowerText(txt):
        if np.random.choice([0, 1], p = [0.5, 0.5]) :
            return txt.upper()
        return txt.lower()    

    # get a line of text
    if np.random.choice([0, 1], p = [0.5, 0.5]) :
        txt = get_text()
    else:
        letters = string.digits 
        c = np.random.randint(1, 9)
        txt = (''.join(random.choice(letters) for i in range(c)))

    if np.random.choice([0, 1], p = [0.5, 0.5]):
        txt = txt.upper()

    # # txt = fake.name()
    # letters = string.ascii_lowercase 
    # c = np.random.randint(1, 4)
    # txt = (''.join(random.choice(letters) for i in range(c)))

    txt = getUpperOrLowerText(txt)

    trueTypeFontSize = np.random.randint(32, 62)
    valid, img, _ = drawTrueTypeTextOnImage(img, txt, (np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0])), trueTypeFontSize)
    
    return valid, img


def print_lines_aligned(img, boxes):
    def getUpperOrLowerText(txt):
        if np.random.choice([0, 1], p = [0.5, 0.5]) :
            return txt.upper()
        return txt.lower()    

    print(f'img : {img.shape}')
    def make_txt():
        # get a line of text
        txt = get_text()

        if np.random.choice([0, 1], p = [0.8, 0.2]) :
            txt = get_text()
            # txt = fake.name()
        else:
            # txt =  get_phone()  #fake.address()
            letters = string.digits 
            c = np.random.randint(1, 9)
            txt = (''.join(random.choice(letters) for i in range(c)))
        
        if np.random.choice([0, 1], p = [0.5, 0.5]):
            txt = txt.upper()

        return getUpperOrLowerText(txt)
   
    trueTypeFontSize = np.random.randint(24, 32)
    xy = (np.random.randint(0, img.shape[1] / 10), np.random.randint(10, img.shape[0] / 20))
    xy = (np.random.randint(0, 10), np.random.randint(10,  20))

    w = img.shape[1]
    h = img.shape[0]
    start_y = xy[1]

    while True:
        m_h = np.random.randint(90, 120)
        start_x = xy[0]
        while True:
            txt = make_txt()    
            pos = (start_x, start_y)
            valid, img, wh = drawTrueTypeTextOnImage(img, txt, pos, trueTypeFontSize)
            txt_w =  wh[0] + np.random.randint(60, 120)
            # print(f' {start_x}, {start_y} : {valid}  : {wh}')
            start_x = start_x + txt_w
            if wh[1] > m_h:
                m_h = wh[1]
            if start_x > w:
                break
        start_y = start_y +  np.random.randint(m_h//2, m_h*1.5)
        if start_y > h:
            break
        
    box = [xy[0], xy[1], wh[0], wh[1]]
    boxes.append(box)

    return True, img

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

        img_= img.copy()
        valid, img_ = print_lines_aligned(img_, [])
 
        image_aug = seq(image = img_)
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
    config_path  = '../mapping.json'

    if not os.path.exists(config_path):
        raise Exception(f'config file not found : {config_path}')

    with open(config_path) as f:
        config = json.load(f)

    cmap = config['colors']
    fmap = config['fields']

    if len(cmap) != len(fmap):
        raise Exception(f'Color map and Field map should be of same size')

    for color, field in zip(cmap, fmap):
        accepted[field] = True
        colormap[field] = color

    # FIXME : Bad mask from labeled data
    accepted['HCFA17a_NONNPI'] = False
    accepted['HCFA17b_NPI'] = False
    accepted['HCFA17a_'] = False
    # accepted['HCFA17_NAME'] = False

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
        labels = []

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
                    if label not in accepted or accepted[label] == False:
                        continue
                    
                    points.append(pts)
                    colors.append(colormap[label])
                    labels.append(label)

                    break
        # else:
        for polygon_node in polygons:
            label = polygon_node.attrib['label']
            pts = polygon_node.attrib['points'].split(';')
            size = len(pts)

            if strict and size > 0 and size != 4:
                raise ValueError("Expected 4 point got : %s " %(size))
            if size  == 4:
                if label not in accepted or accepted[label] == False:
                    continue
                points.append(pts)
                colors.append(colormap[label])
                labels.append(label)
        
        data['ds'].append({'name': filename, 'points': points, 'color':colors, 'mapped': 0 if mappped_images == '' else 1, "labels":labels}) 

        for filename in mappped_images.split(',') :
            data['ds'].append({'name': filename, 'points': points, 'color':colors, 'mapped':0, "labels":labels}) 

        index = index+1

    print('Total annotations : %s '% (len(data['ds'])))
    rsize = (1024, 1024)
    
    for row in data['ds']:
        filename = row['name']
        points_set = row['points']
        color_set = row['color']
        label_set = row['labels']
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
                _=cv2.putText(overlay_img, txt_label, org, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
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
            aug_images, aug_masks = augment_image(img, mask_img, pts, 10)
            # augment original
            valid, img = print_lines_aligned(img, [])

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
    # remap(root_src , dir_dest, cvat_annotation_file, remap_src)
    cvat_annotation_file=os.path.join(root_src, 'remap.xml') 

    create_mask(dir_src, dir_dest, cvat_annotation_file, remap_src)
    split_dir(dir_dest, dir_dest_split)

# python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/hicfa_form --output_dir ./datasets/hicfa_form/ready