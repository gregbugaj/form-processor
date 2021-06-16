import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
import numpy as np

def viewImage(image, name='Display'):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
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
class FormSegmeneter:
    def __init__(self, network):
        self.network = network
        print("Initialzed")

    def process(self, img_path):
        """
            Form segmentation 
        """
        img = cv2.imread(img_path)
        # converting from BGR to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        viewImage(hsv, 'HSV') 

        if True:
            mask = cv2.inRange(hsv,(10, 100, 20), (25, 255, 255))
            cv2.imshow("orange", mask)
            cv2.waitKey()
            cv2.destroyAllWindows()

        # return
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = apply_filter(img)
        viewImage(img, "Source Image") 

        print("Processing segmentation")
        h = img.shape[0]
        w = img.shape[1]

        print("{} : {}".format(h, w))
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        viewImage(hsv, 'HSV') 

        # Orange (10, 100, 20), (25, 255, 255)
        #  PURPLE (hMin = 111 , sMin = 101, vMin = 16), (hMax = 138 , sMax = 255, vMax = 255)
        low_color = np.array([10, 100, 20],np.uint8)
        high_color = np.array([25, 255, 255],np.uint8)
        
        mask = cv2.inRange(hsv, low_color, high_color)
        # mask = cv2.inRange(hsv, (10, 100, 20), (25, 255, 255))
        # mask = cv2.inRange(hsv, (135, 205, 198), (25, 255, 255))
        # mask = cv2.inRange(hsv,(10, 100, 20), (25, 255, 255))
        viewImage(mask, 'mask')

        # hsv[curr_mask > 0] = ([75,255,200])
        # viewImage(hsv, 'hsv_img')
        result_white = cv2.bitwise_and(img, img, mask=mask)
        viewImage(result_white, 'result_white') ## 2

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

def processXX(self, img):
    "Process form"
    print("Processing segmentation")
    h = img.shape[0]
    w = img.shape[1]

    print("{} : {}".format(h, w))
    seg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_seg = cv2.cvtColor(seg, cv2.COLOR_RGB2HSV)
    hsv_img = hsv_seg

    # plt.imshow(seg)
    # plt.show()

    ## getting green HSV color representation
    green = np.uint8([[[26, 84, 63]]])
    green_hsv = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
    print( green_hsv)

    green_low = np.array([81 , 68, 84] )
    green_high = np.array([75, 255, 255])
    curr_mask = cv2.inRange(hsv_img, green_low, green_high)
    hsv_img[curr_mask > 0] = ([75,255,200])
    viewImage(hsv_img) ## 2


    if False:
        pixel_colors = seg.reshape((np.shape(seg)[0]*np.shape(seg)[1], 3))
        norm = colors.Normalize(vmin=-1.,vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()

        r, g, b = cv2.split(seg)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Red")
        axis.set_ylabel("Green")
        axis.set_zlabel("Blue")
        plt.show()

    if False:
        hsv_seg = cv2.cvtColor(seg, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_seg)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")

        axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")
        plt.show()


    light_orange = (41, 176,  84)
    dark_orange = (24, 255, 255)

    lo_square = np.full((10, 10, 3), light_orange, dtype=np.uint8) / 255.0
    do_square = np.full((10, 10, 3), dark_orange, dtype=np.uint8) / 255.0

    plt.subplot(1, 2, 1)
    plt.imshow(hsv_to_rgb(do_square))
    plt.subplot(1, 2, 2)
    plt.imshow(hsv_to_rgb(lo_square))
    plt.show()

    mask = cv2.inRange(hsv_seg, light_orange, dark_orange)
    result = cv2.bitwise_and(seg, seg, mask=mask)

    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()

    # final_mask = mask + mask_white

    # final_result = cv2.bitwise_and(nemo, nemo, mask=final_mask)
    # plt.subplot(1, 2, 1)
    # plt.imshow(final_mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(final_result)
    # plt.show()