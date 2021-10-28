
import os
import numpy as np
import cv2
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim

from boxes.box_processor import BoxProcessor
from form.icr_processor import IcrProcessor


def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def apply_filter(src_img):
    img = src_img.copy()
    # blur = cv2.blur(img,(5,5))
    # blur0= cv2.medianBlur(blur,5)
    # blur1= cv2.GaussianBlur(blur0,(5,5),0)
    # blur2= cv2.bilateralFilter(blur1,9,75,75)

    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # im = cv2.filter2D(blur2, -1, kernel)
    # im2 = cv2.filter2D(src_img, -1, kernel)

    # sharpened_image = unsharp_mask(blur2)
    # sharpened_image2 = unsharp_mask(src_img)
    # blurX= cv2.GaussianBlur(sharpened_image2,(5,5),0)

    return unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=0, threshold=0)


def blend_with_mask_matrix(src1, src2, mask):
    res_channels = []
    for c in range(0, src1.shape[2]):
        a = src1[:, :, c]
        b = src2[:, :, c]
        m = mask[:, :, c]
        res = cv2.add(
            cv2.multiply(b, cv2.divide(np.full_like(m, 255) - m,
                         255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
            cv2.multiply(a, cv2.divide(
                m, 255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
            dtype=cv2.CV_8U)
        res_channels += [res]
    res = cv2.merge(res_channels)
    return res


def main(real_img, fake_img):
    print(f'real_img : {real_img}')
    print(f'fake_img : {fake_img}')

    real = cv2.imread(real_img, cv2.IMREAD_GRAYSCALE)
    fake = cv2.imread(fake_img, cv2.IMREAD_GRAYSCALE)

    print(real.shape)
    print(fake.shape)

    blended_img = 255 - cv2.bitwise_xor(real, fake)
    diff_img =  255 -  cv2.bitwise_xor(fake, blended_img)

    # blended = cv2.addWeighted(diff_img, .5,  blended_img, 1, 0)
    # blended[blended >= 125] = [255]

    imwrite('/tmp/segmentation-mask/blended_img.png', blended_img)
    imwrite('/tmp/segmentation-mask/diff_img.png', diff_img)

def process(img_path):
    work_dir = '/tmp/form-segmentation'

    snippet = cv2.imread(img_path)
    id = '272943_0031516168746_001'
    key = 'HCFA24'
    boxer = BoxProcessor(work_dir, cuda=False)
    boxes, img_fragments, lines, _ = boxer.extract_bounding_boxes(
        id, 'field', snippet)

    icr = IcrProcessor(work_dir)
    icr.recognize(id, key, snippet, boxes, img_fragments, lines)


if __name__ == '__main__':

    # process(os.path.expanduser('~/tmp/hicfa_mask/blended.png'))
    # process(os.path.expanduser('~/tmp/hicfa_mask/final.png'))

    if True:
        main(os.path.expanduser('/tmp/segmentation-mask/0_src.png'),
             os.path.expanduser(
                 '/tmp/segmentation-mask/0_mask.png'),
             # ,
             )
