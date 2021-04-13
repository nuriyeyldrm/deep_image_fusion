from sporco.util import tikhonov_filter
import pywt
import numpy as np
from imageio import imread

"""
# Performs Image fusion using Discrete wavelet transform (DWT) with Daubechies filter
# input: two Images dataset (matrix) to be fused
# output: one Fused Image dataset (matrix)"""


def Fusion_DWT_db2(image1, image2):
    # decomposing each image using Discrete wavelet transform(DWT) with Daubechies filter (db2)
    coefficients_1 = pywt.wavedec2(image1, 'db2', level=2)
    coefficients_2 = pywt.wavedec2(image2, 'db2', level=2)
    # creating variables to be used
    coefficients_h = list(coefficients_1)
    # fusing the decomposed image data
    coefficients_h[0] = (coefficients_1[0] + coefficients_2[0]) * 0.5
    # creating variables to be used
    temp1 = list(coefficients_1[1])
    temp2 = list(coefficients_2[1])
    temp3 = list(coefficients_h[1])
    # fusing the decomposed image data
    temp3[0] = (temp1[0] + temp2[0]) * 0.5
    temp3[1] = (temp1[1] + temp2[1]) * 0.5
    temp3[2] = (temp1[2] + temp2[2]) * 0.5
    coefficients_h[1] = tuple(temp3)
    # Creating fused image by reconstructing the fused decomposed image
    result = pywt.waverec2(coefficients_h, 'db2')
    return result


def lowpass(s, lda, npad):  # In this function, low pass filtering is done by using Tikhonov filter.
    return tikhonov_filter(s, lda, npad)


def signaltonoise(a, axis, ddof):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


# idx for selecting image, you can change manually 1 to 21 (21 different infrared and 21 different visible image)
idx = 1

vis = imread('IV_images/VIS%d.png' % idx)
ir = imread('IV_images/IR%d.png' % idx)

npad = 16
lda = 5
vis_low, vis_high = lowpass(vis.astype(np.float32) / 255, lda, npad)
ir_low, ir_high = lowpass(ir.astype(np.float32) / 255, lda, npad)

img = Fusion_DWT_db2(vis.astype(np.float32) / 255, ir_high)


print("\nsignal to noise ratio for image : ", np.max(signaltonoise(img, axis=0, ddof=0)))
