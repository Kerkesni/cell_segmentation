from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.signal import convolve
import cv2


def get_proj_img(image, radius):
    image = image.astype('int')
    workingDims = tuple((e + 2*radius) for e in image.shape)

    h,w = image.shape

    ori_img = np.zeros(workingDims)
    mag_img = np.zeros(workingDims) # Magnitutde Projection Image


    a1 = np.matrix([1, 2, 1])
    a2 = np.matrix([-1, 0, 1])
    Kx = a1.T * a2
    Ky = a2.T * a1

    sobel_x = convolve(image, Kx)
    sobel_y = convolve(image, Ky)
    sobel_norms = np.hypot(sobel_x, sobel_y)

    # Distances to afpx, afpy (affected pixels)
    dist_afpx = np.multiply(np.divide(sobel_x, sobel_norms, out = np.zeros(sobel_x.shape), where = sobel_norms!=0), radius)
    dist_afpx = np.round(dist_afpx).astype(int)

    dist_afpy = np.multiply(np.divide(sobel_y, sobel_norms, out = np.zeros(sobel_y.shape), where = sobel_norms!=0), radius)
    dist_afpy = np.round(dist_afpy).astype(int)

    for cords, sobel_norm in np.ndenumerate(sobel_norms):
        i, j = cords

        pos_aff_pix = (i+dist_afpx[i,j], j+dist_afpy[i,j])
        neg_aff_pix = (i-dist_afpx[i,j], j-dist_afpy[i,j])

        ori_img[pos_aff_pix] += 1
        ori_img[neg_aff_pix] -= 1
        mag_img[pos_aff_pix] += sobel_norm
        mag_img[neg_aff_pix] -= sobel_norm

    ori_img = ori_img[:h, :w]
    mag_img = mag_img[:h, :w]

    print ("Did it go back to the original image size? ")
    print (ori_img.shape == image.shape)

    # try normalizing ori and mag img
    return ori_img, mag_img

def get_sn(ori_img, mag_img, radius, kn, alpha):

    ori_img_limited = np.minimum(ori_img, kn)
    fn = np.multiply(np.divide(mag_img,kn), np.power((np.absolute(ori_img_limited)/kn), alpha))

    # convolute fn with gaussian filter.
    sn = gaussian_filter(fn, 0.25*radius)

    return sn

def do_frst(image, radius, kn, alpha, ksize = 3):
    ori_img, mag_img = get_proj_img(image, radius)
    sn = get_sn(ori_img, mag_img, radius, kn, alpha)

    return sn

img = cv2.imread('datasets/CUSTOM/BM_GRAZ_HE_0001_01.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

result = do_frst(gray, 10, 10, 2)
print(result)

cv2.imshow("Output:", result)
cv2.waitKey(0)