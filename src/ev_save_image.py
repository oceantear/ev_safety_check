#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
from std_msgs.msg import *
from threading import Timer
from datetime import datetime
import os
import rospkg

bridge = CvBridge()
count = 0
startSaveImageFlag = False
clearStartSaveImageTimer = 3
totalTime = 0
folderName = None

rospack = rospkg.RosPack()
packPath = rospack.get_path('ev_safety_check')
print(packPath)

imagePath = packPath + "/Image/"


def saveImage(data):
    global count, startSaveImageFlag, folderName

    if startSaveImageFlag == True:
        count = count + 1
        #frame rate : 10, 1ps/sec
        if count != 10:
            return
        count = 0

        try:
            cv2_image = bridge.imgmsg_to_cv2(data, "bgr8")
            
        except CvBridgeError as e:
            print(e)
        else:
            
            now = rospy.Time.now()
            fname = str(now) + ".jpeg"
            cv2.imwrite(os.path.join( imagePath + folderName , fname) , cv2_image)

def startSaveImage(data):
    global startSaveImageFlag,startTime, imagePath ,folderName

    startTime = time.time()
    folderName = time.strftime("%Y-%m-%d", time.localtime())
    #print("start training time = ",folderName)

    if not os.path.exists(imagePath + folderName):
        print("floder not exist :",imagePath + folderName)
        os.makedirs(imagePath + folderName)
    

    startSaveImageFlag = True
    t1 = Timer(clearStartSaveImageTimer ,clearSaveImageFlag)
    t1.daemon = True
    t1.start()
    
    return

def clearSaveImageFlag():
    global startSaveImageFlag, startTime ,endTime

    startSaveImageFlag = False
    endTime = time.time()
    totalTime = endTime - startTime
    print("toralTime = ",totalTime)

    return
        

def main(args):
    rospy.init_node('ev_save_image', anonymous=True)
    image_sub = rospy.Subscriber("/camera_rear/image_rect_color", Image, saveImage, queue_size=1 ,buff_size=5000000)
    #rospy.Subscriber("/usb_cam/image_rect_color", Image, saveImage, queue_size=1 ,buff_size=5000000)
    rospy.Subscriber('/checkEV',Bool,startSaveImage)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
