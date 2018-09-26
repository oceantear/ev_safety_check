#!/usr/bin/env python
import os
import sys
import cv2
from pp0 import preprocessImage
from pp0 import resizeKeepAspectRatio
#import pylab as pl

ladybugPath = "/home/jimmy/catkin_ws/src/ev_safety_check/src/"
writeladybugPath = "/home/jimmy/catkin_ws/src/ev_safety_check/src/"

InputTrainSafeDir = "/home/jimmy/catkin_ws/src/ev_safety_check/Img/train/safe/"
InputTrainUnSafeDir = "/home/jimmy/catkin_ws/src/ev_safety_check/Img/train/unsafe/"

writeTrainSafeDir = "/home/jimmy/catkin_ws/src/ev_safety_check/Img/preprossedImg/safe/"
writeTrainUnSafeDir = "/home/jimmy/catkin_ws/src/ev_safety_check/Img/preprossedImg/unsafe/"

InputTestSafeDir = "/home/jimmy/catkin_ws/src/ev_safety_check/Img/test/safe/"
InputTestUnSafeDir = "/home/jimmy/catkin_ws/src/ev_safety_check/Img/test/unsafe/"

writeTestSafeDir ="/home/jimmy/catkin_ws/src/ev_safety_check/Img/preprossedImg/test/safe/"
writeTestUnSafeDir ="/home/jimmy/catkin_ws/src/ev_safety_check/Img/preprossedImg/test/unsafe/"

InputTrainCarDir = "/home/jimmy/catkin_ws/src/ev_safety_check/Img/train/car/"
InputTestCarDir  = "/home/jimmy/catkin_ws/src/ev_safety_check/Img/test/car/"

OutputTrainCarDir = "/home/jimmy/catkin_ws/src/ev_safety_check/Img/preprossedImg/train/car/"
OutputTestCarDir = "/home/jimmy/catkin_ws/src/ev_safety_check/Img/preprossedImg/test/car/"

InputGrayDir = "/home/jimmy/ev_safety_check/original_640_480/elevator/alldata/alldata/nopeople/"
OutputGrayDir = "/home/jimmy/ev_safety_check/image/preprossedImg/20180925/nopeople/"

InputDir = InputGrayDir
OutputDir = OutputGrayDir


FileExtentionNameJPG = "jpg"
FileExtentionNameJPEG = "jpeg"

shiftTx = [-0.15,
           -0.1 ,
           -0.05 ,
            0 ,
            0.05 ,
            0.1 ,
            0.15]  #-0.05 , -0,1 , -0.15 ,0, 0.05 , 0,1 , 0.15
zoomIn  = [ 1 , 
            1.1 ,
            1.2 ,
            1.3 ]   #1 , 1.1 , 1.2 , 1.3

def genExtentionFileName(shiftTx,zoomIn):

    extentionFname = "default"
    print "shiftTx = ",shiftTx ,"zoomIn = ",zoomIn
    if shiftTx == -0.15 and zoomIn == 1:
        extentionFname = "_ntx015_zoom1"
    elif shiftTx == -0.15 and zoomIn == 1.1:
        extentionFname = "_ntx015_zoom11"
    elif shiftTx == -0.15 and zoomIn == 1.2:
        extentionFname = "_ntx015_zoom12"
    elif shiftTx == -0.15 and zoomIn == 1.3:
        extentionFname = "_ntx015_zoom13"

    elif shiftTx == -0.1 and zoomIn == 1:
        extentionFname = "_ntx01_zoom1"
    elif shiftTx == -0.1 and zoomIn == 1.1:
        extentionFname = "_ntx01_zoom11"
    elif shiftTx == -0.1 and zoomIn == 1.2:
        extentionFname = "_ntx01_zoom12"
    elif shiftTx == -0.1 and zoomIn == 1.3:
        extentionFname = "_ntx01_zoom13"

    elif shiftTx == -0.05 and zoomIn == 1:
        extentionFname = "_ntx005_zoom1"
    elif shiftTx == -0.05 and zoomIn == 1.1:
        extentionFname = "_ntx005_zoom11"
    elif shiftTx == -0.05 and zoomIn == 1.2:
        extentionFname = "_ntx005_zoom12"
    elif shiftTx == -0.05 and zoomIn == 1.3:
        extentionFname = "_ntx005_zoom13"

    elif shiftTx == 0 and zoomIn == 1:
        extentionFname = "_tx0_zoom1"
    elif shiftTx == 0 and zoomIn == 1.1:
        extentionFname = "_tx0_zoom11"
    elif shiftTx == 0 and zoomIn == 1.2:
        extentionFname = "_tx0_zoom12"
    elif shiftTx == 0 and zoomIn == 1.3:
        extentionFname = "_tx0_zoom13"

    elif shiftTx == 0.05 and zoomIn == 1:
        extentionFname = "_tx005_zoom1"
    elif shiftTx == 0.05 and zoomIn == 1.1:
        extentionFname = "_tx005_zoom11"
    elif shiftTx == 0.05 and zoomIn == 1.2:
        extentionFname = "_tx005_zoom12"
    elif shiftTx == 0.05 and zoomIn == 1.3:
        extentionFname = "_tx005_zoom13"

    elif shiftTx == 0.1 and zoomIn == 1:
        extentionFname = "_tx01_zoom1"
    elif shiftTx == 0.1 and zoomIn == 1.1:
        extentionFname = "_tx01_zoom11"
    elif shiftTx == 0.1 and zoomIn == 1.2:
        extentionFname = "_tx01_zoom12"
    elif shiftTx == 0.1 and zoomIn == 1.3:
        extentionFname = "_tx01_zoom13"

    elif shiftTx == 0.15 and zoomIn == 1:
        extentionFname = "_tx015_zoom1"
    elif shiftTx == 0.15 and zoomIn == 1.1:
        extentionFname = "_tx015_zoom11"
    elif shiftTx == 0.15 and zoomIn == 1.2:
        extentionFname = "_tx015_zoom12"
    elif shiftTx == 0.15 and zoomIn == 1.3:
        extentionFname = "_tx015_zoom13"
    
    return extentionFname


def main(args):


    for fname in os.listdir( InputDir ):
        if fname.endswith( FileExtentionNameJPEG ):
            srcImage = cv2.imread( InputDir + fname )
            #cv2.imshow('srcImage',srcImage)
            #cv2.waitKey(0)

            for tx in shiftTx:
                for zoomin in zoomIn:
                    #if tx != 0:
                    outputImage = preprocessImage(srcImage, tx, zoomin)
                    #cv2.imshow('outputImage', outputImage)
                        
                    extensionName = genExtentionFileName(tx,zoomin)
                    print ("extensionName = ",extensionName)
                    #cv2.waitKey(0)
                    str = fname.split("."+ FileExtentionNameJPEG )
                    cv2.imwrite( OutputDir + str[0] + extensionName+'.jpeg', outputImage )



def ladybug(args):
            srcImage = cv2.imread('ladybug.jpg')
            cv2.imshow('srcImage',srcImage)
            cv2.waitKey(0)

            outputImage = preprocessImage(srcImage, 0.15,1.1)
 
            cv2.imshow('outputImage', outputImage)
            cv2.waitKey(0)
            #cv2.imwrite(ladybugPath+"resize"+'ladybug.jpeg',outputImage)

            outputImage1 = preprocessImage(srcImage, 0.15,1.2)
 
            cv2.imshow('outputImage', outputImage)
            cv2.waitKey(0)
            #cv2.imwrite(ladybugPath+"resize"+'ladybug.jpeg',outputImage)

            outputImage = preprocessImage(srcImage, 0.15,1.3)
 
            cv2.imshow('outputImage', outputImage)
            cv2.waitKey(0)
            #cv2.imwrite(ladybugPath+"resize"+'ladybug.jpeg',outputImage)

def resize224():
    for fname in os.listdir( InputDir ):
        if fname.endswith( FileExtentionNameJPEG ):
            srcImage = cv2.imread( InputDir + fname )
            print fname
            #cv2.imshow('srcImage',srcImage)
            #cv2.waitKey(0)

            #for tx in shiftTx:
            #    for zoomin in zoomIn:
                    #if tx != 0:
            #outputImage = resizeKeepAspectRatio(srcImage, (224, 224), 0)
            #cv2.imshow('srcImage',outputImage)
            #cv2.waitKey(0)

            outputImage1 = preprocessImage(srcImage, 0, 1.0)
            #cv2.imshow('outputImage',outputImage1)
            #cv2.waitKey(0)
                    #cv2.imshow('outputImage', outputImage)
                        
                    #extensionName = genExtentionFileName(tx,zoomin)
                    #print ("extensionName = ",extensionName)
                    #cv2.waitKey(0)
                    #str = fname.split("."+ FileExtentionNameJPEG )
            cv2.imwrite( OutputDir + fname, outputImage1 )



def printshiftTx(args):
    for x in shiftTx:
        print x


    



if __name__ == '__main__':
    
    #main(sys.argv)
    resize224()
    
