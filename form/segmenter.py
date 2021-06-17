import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
import numpy as np

# Add parent to the search path so we can reference the module here without throwing and exception 
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from utils.nms import nms

hsv_color_ranges = [
            [[55, 58, 0], [86, 255, 255]],     # GREEN DARK  7fd99d
            [[123, 99, 206], [140, 255, 255]], # Purple      a96df8
            [[0, 152, 240], [9, 255, 255]],    # Red         ff614e
            [[97, 188, 0], [179, 255, 178]],   # Blue        016aa4
            [[25, 195, 0], [51, 255, 255]],    # YELLOW      99624a
            [[0, 74, 0], [13, 170, 158]],      # Brown       99624a
            [[0, 240, 124], [25, 255, 255]],   # ORANGE            
            [[111, 180, 90], [179, 255, 255]], # PINK
            [[38, 154, 188], [51, 255, 255]],  # GREEN
        ]

def viewImage(image, name='Display'):
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_filter(src_img):
    img = src_img.copy()
    blur = cv2.blur(img,(5,5))
    blur0= cv2.medianBlur(blur,5)
    blur1= cv2.GaussianBlur(blur0,(5,5),0)
    blur2= cv2.bilateralFilter(blur1,9,75,75)
    return blur2

def rgb_2_hsv(r,g,b):
    ## getting green HSV color representation
    col = np.uint8([[[r,g,b]]])
    hsv_col = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
    print(hsv_col)
    return hsv_col

def fixHSVRange(h, s, v):
    """
        Different applications use different scales for HSV. 
        For example gimp uses H = 0-360, S = 0-100 and V = 0-100. But OpenCV uses H: 0-179, S: 0-255, V: 0-255. 
    """
    # Normal H,S,V: (0-360,0-100%,0-100%)
    # OpenCV H,S,V: (0-180,0-255 ,0-255)
    return (180 * h / 360, 255 * s / 100, 255 * v / 100)

def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)

class FormSegmeneter:
    def __init__(self, network):
        self.network = network
        print("Initialzed")

    def __extract_segmenation_mask(self, img):
        """
            Extract segmentation mask for the image
        """
        return img

    def __fragment(self, img, hsv, layerId, fieldId):
        """
            Segment fragment 
        """
        print('layerId / id {} : {}'.format(layerId, fieldId))
        # id = hsv_color_ranges[val]
        # print(id)
        print('Fragment')
        colid = fieldId
        low_color = np.array(hsv_color_ranges[colid][0],np.uint8)
        high_color = np.array(hsv_color_ranges[colid][1],np.uint8)

        mask = cv2.inRange(hsv, low_color, high_color)
        # viewImage(mask, 'mask')

        # Extract the area of interest
        # result_white = cv2.bitwise_and(img, img, mask=mask)
        # viewImage(result_white, 'result_white') 

        #  find Canny Edges
        edged = cv2.Canny(mask, 30, 200)
        # Use Blur
        blur = cv2.GaussianBlur(edged, (3, 3), 0)
        # viewImage(blur,'blur')

        _, threshold = cv2.threshold(blur, 70, 255, 0)
        # viewImage(threshold, 'Threashold') 
        
        contours, hierarchy =  cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours, -1, (255, 255, 0), 2)
        # viewImage(img,'Contours')

        print("Contours len = " + str(len(contours)))
        font = cv2.FONT_HERSHEY_SIMPLEX

        all_boxes = []
        cls_scores = []
        all_pts = []
        drift = 5
        for cnt in contours:
            # (center(x, y), (width, height), angle of rotation)
            (x, y), (width, height), angle = rect = cv2.minAreaRect(cnt)
            # 90.0 deg https://theailearner.com/tag/cv2-minarearect/
            aspect_ratio = min(width, height) / max(width, height)
            if angle < 90 - drift or angle > 90 + drift:
                continue
            # convert
            box = cv2.boxPoints(rect)
            box = np.int0(box)
             
            # Calculate the moments and get area
            # Trying to filter out small pieces
            M = cv2.moments(cnt)
            area = M['m00']
            # print(area)
            if area > 50:
                all_boxes.append([x ,y, width, height])
                all_pts.append(box)
                cls_scores.append(area)

        # important that we apply non-max suppression to the candiates
        cls_scores = np.array(cls_scores) # prevent 'list indices must be integers or slices, not tuple'
        all_boxes = np.hstack((all_boxes, cls_scores [:, np.newaxis])).astype(np.float32, copy=False)

        keep = nms(all_boxes, 0.3)
        idx = keep[0]
        box = all_pts[idx]

        if False:
            try:
                cv2.drawContours(img, [box], -1, (255, 0, 0), 2, 1)
                org = [box[3][0]+5, box[3][1]-5]
                label = 'id : {id}'.format(id=colid)
                _=cv2.putText(img, label, org, font, .4, (0, 0, 0), 1, cv2.LINE_AA)
            except Exception as e:
                print(e)
            viewImage(img, 'final')

        non_scored_box = all_pts[idx][:4]

        print('################ : {}'.format(non_scored_box))
        return box, non_scored_box

    def process(self, img_path):
        """
            Form segmentation 
        """

        img = cv2.imread(img_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        segmask = self.__extract_segmenation_mask(img)

        # converting from BGR to HSV color space
        hsv = cv2.cvtColor(segmask, cv2.COLOR_BGR2HSV)
        viewImage(hsv, 'HSV') 

        # return
        # img = cv2.imread(img_path)
        # causes issues due to conversion
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segmask = apply_filter(segmask)
        viewImage(segmask, "Source Image") 

        print("Processing segmentation")
        h = segmask.shape[0]
        w = segmask.shape[1]

        print("{} : {}".format(h, w))
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        viewImage(hsv, 'HSV') 

        # (hMin = 111 , sMin = 101, vMin = 16), (hMax = 138 , sMax = 255, vMax = 255) PURPLE
        # (hMin = 97 , sMin = 188, vMin = 0), (hMax = 179 , sMax = 255, vMax = 178) Blue
        # (hMin = 38 , sMin = 154, vMin = 188), (hMax = 51 , sMax = 255, vMax = 255) Green
        # (hMin = 25 , sMin = 195, vMin = 0), (hMax = 51 , sMax = 255, vMax = 255) Yellow
        # (hMin = 55 , sMin = 58, vMin = 0), (hMax = 86 , sMax = 255, vMax = 255) Green dark  
        # (hMin = 0 , sMin = 152, vMin = 240), (hMax = 9 , sMax = 255, vMax = 255) Red
        # (hMin = 0 , sMin = 74, vMin = 0), (hMax = 13 , sMax = 170, vMax = 158) # Brown
        # (hMin = 112 , sMin = 38, vMin = 154), (hMax = 140 , sMax = 155, vMax = 255) # Purple
        # (hMin = 123 , sMin = 99, vMin = 206), (hMax = 139 , sMax = 255, vMax = 255)

        colid = 8
        low_color = np.array(hsv_color_ranges[colid][0],np.uint8)
        high_color = np.array(hsv_color_ranges[colid][1],np.uint8)

        layers = {
            'layer_1' : {
                'HCFA02': 0, 
                'HCFA05_ADDRESS': 1, 
                'HCFA05_CITY': 2,
                'HCFA05_STATE': 3,
                'HCFA05_ZIP': 4,
                'HCFA05_PHONE': 5,
                'HCFA21': 6,
                'HCFA24': 7,
                'HCFA33_BILLING': 8,
            },
        }

        fragments = []
        for lkey in layers.keys():
            print('Processing layer : {}'.format(lkey))
            layer = layers[lkey]
            for key in layer.keys():
                val = layer[key]
                box, non_scored_box = self.__fragment(segmask, hsv, lkey, val)     
                frag = {
                    'layer':lkey,
                    'key':key,
                    'id':val,
                    'box':    np.array(non_scored_box,np.uint8)
                }
                fragments.append(frag)

                print('****************')
                print(non_scored_box)

                print(box)
                print('****************')
                try:
                    cv2.drawContours(segmask, [box], -1, (255, 0, 0), 2, 1)
                    org = [box[3][0]+5, box[3][1]-5]
                    label = '{label} ({id})'.format(id=colid, label=key)
                    _=cv2.putText(segmask, label, org, font, .4, (0, 0, 0), 1, cv2.LINE_AA)
                except Exception as e:
                    print(e)

                break

        viewImage(segmask, 'final')
        print(fragments)

        dir_out  ='/tmp/fragments'
        # process extracted fragments
        for frag in fragments:
            print(frag)
            key = frag['key']
            _id = frag['id']
            box = frag['box'] # xywh
            print(box)

            # snip = img[box[1]:box[3], box[0]: box[2]]
            # P1 = Top LEft, P2 = Bottom Right

            print(img.shape)
            snip = img[box[0]:box[1], :]

            output_filename='%s-%d.png' % (key, _id)
            print('File written : %s' % (output_filename))
            imwrite(os.path.join(dir_out, output_filename), snip)

        return 

        mask = cv2.inRange(hsv, low_color, high_color)
        viewImage(mask, 'mask')

        # Extract the area of interest
        # result_white = cv2.bitwise_and(img, img, mask=mask)
        # viewImage(result_white, 'result_white') 

        # Grayscale
        #  find Canny Edges
        edged = cv2.Canny(mask, 30, 200)

        # Use Blur
        blur = cv2.GaussianBlur(edged, (3, 3), 0)
        viewImage(blur,'blur')

        _, threshold = cv2.threshold(blur, 70, 255, 0)
        viewImage(threshold, 'Threashold') 
        
        contours, hierarchy =  cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours, -1, (255, 255, 0), 2)
        # viewImage(img,'Contours')

        print("Contours len = " + str(len(contours)))

        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        # Blue color in BGR
        for cnt in contours:
            # (center(x, y), (width, height), angle of rotation)
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
            # 90.0 deg https://theailearner.com/tag/cv2-minarearect/
            if angle < 90 -10 or angle > 90 + 10:
                continue
            # convert
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Calculate the moments and get area
            # Trying to filter out small pieces
            M = cv2.moments(cnt)
            area = M['m00']
            # print(area)
            if area > 50:
                try:
                    cv2.drawContours(img, [box], -1, (255, 0, 0), 2, 1)
                    org = [box[3][0]+5, box[3][1]-5]
                    label = 'id : {id}'.format(id=colid)
                    _=cv2.putText(img, label, org, font, .4, (0, 0, 0), 1, cv2.LINE_AA)
                except Exception as e:
                    print(e)

        viewImage(img,'final')

def processLEAF(self, img_path):
    "Process form"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = apply_filter(img)
    viewImage(img, "Source Image") 

    print("Processing segmentation")
    h = img.shape[0]
    w = img.shape[1]

    print("{} : {}".format(h, w))
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    viewImage(hsv_img, 'HSV') 

    if False:
        pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
        norm = colors.Normalize(vmin=-1.,vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()

        hsv_seg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_seg)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")

        axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")
        plt.show()
        
    ## getting green HSV color representation
    green = np.uint8([[[26, 84, 63]]])
    green_hsv = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    print(green_hsv)

    low_color = np.array([45 , 100, 50] )
    high_color = np.array([75, 255, 255])

    curr_mask = cv2.inRange(hsv_img, low_color, high_color)
    viewImage(curr_mask, 'curr_mask')

    hsv_img[curr_mask > 0] = ([75,255,200])
    viewImage(hsv_img, 'hsv_img')

    result_white = cv2.bitwise_and(img, img, mask=curr_mask)
    viewImage(result_white, 'result_white') ## 2

    ## converting the HSV image to Gray inorder to be able to apply 
    ## contouring
    RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
    viewImage(gray, 'Gray') 

    ret, threshold = cv2.threshold(gray, 90, 255, 0)
    viewImage(threshold, 'Threashold') 

    contours, hierarchy =  cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    viewImage(img,'Contours') 


if __name__ == '__main__':
    img_path ='./assets/forms-seg/001_fake.png'
    # img_path ='./assets/forms-seg/baseline.jpg'
    # img_path ='./assets/forms-seg/001_fake_green.jpg'
    
    segmenter = FormSegmeneter(network="")
    segmenter.process(img_path)  