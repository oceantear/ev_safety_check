#!/usr/bin/env python

import sys
import cv2
import numpy as np

def resizeKeepAspectRatio(srcImage, dstSize, bgColor):

    # print dstSize   # (height, width)
    # print dstSize[0] #224
    # print dstSize[1] #224
    # print srcImage.shape[0] #480
    # print srcImage.shape[1] #640

    h1 = dstSize[0] * srcImage.shape[0] / srcImage.shape[1]
    w2 = dstSize[1] * srcImage.shape[1] / srcImage.shape[0]

    # print h1
    # print w2

    if h1 <= dstSize[0]: 
        dstImage = cv2.resize(srcImage, (int(dstSize[0]), int(h1)))
    else:
        dstImage = cv2.resize(srcImage, (int(w2), int(dstSize[1])))

    # print dstImage.shape[0]  #168
    # print dstImage.shape[1]  #224

    top = int((dstSize[1] - dstImage.shape[0]) / 2)
    down = int((dstSize[1] - dstImage.shape[0] + 1) / 2)
    left = int((dstSize[0] - dstImage.shape[1]) / 2)
    right = int((dstSize[0] - dstImage.shape[1] + 1) / 2)

    # print top, down, left, right

    output = cv2.copyMakeBorder(dstImage, top, down, left, right, cv2.BORDER_CONSTANT, value=bgColor)

    return output

def horizontalShift(img, tx=0.0):
    rows, cols = img.shape[:2]
    M = np.float32([[1,0,cols*tx],[0,1,0]])
    shiftedImage = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return shiftedImage



def zoom(img, z=1.0):
    zoomedImage = cv2.resize(img, None, fx=z, fy=z) 

    return zoomedImage

def cropCenter(img, size):
    rows,cols = img.shape[:2]
    croppedImage = img[rows/2-size[0]/2:rows/2+size[0]/2, cols/2-size[1]/2:cols/2+size[1]/2]

    return croppedImage


def preprocessImage(img, tx=0.0, z=1.0):
    resizedImage = resizeKeepAspectRatio(img, (224, 224), 0)
    shiftedImage = horizontalShift(resizedImage, tx)
    zoomedImage = zoom(shiftedImage, z)
    croppedImage = cropCenter(zoomedImage, (224, 224))

    return croppedImage
    
def main(args):
    srcImage = cv2.imread('ladybug.jpg')
    cv2.imshow('srcImage',srcImage)
    cv2.waitKey(0)

    outputImage = preprocessImage(srcImage, 0.0, 1.0)
 
    cv2.imshow('outputImage', outputImage)
    cv2.waitKey(0)


if __name__ == '__main__':
    main(sys.argv)
    
