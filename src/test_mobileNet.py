#!/usr/bin/env python
from __future__ import print_function

from threading import Timer
import sys
import rospy
import rospkg
import numpy as np
import cv2
from std_msgs.msg import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import load_model
from keras_applications.mobilenet import relu6
from keras.layers import convolutional
from keras.applications.imagenet_utils import decode_predictions
import time
from threading import Lock


rospack = rospkg.RosPack()
packPath = rospack.get_path('ev_safety_check')
print(packPath)

bridge = CvBridge()
lock = Lock()

checkCBPub = rospy.Publisher('/checkEVcb',Bool,queue_size=1)
continuousSafetyCheckResultUnsafePub = rospy.Publisher('/continuousSafetyCheckResultUnsafe',Bool,queue_size=1)

# load model
modelFileName = packPath + "/models/mobileNet_ex1.h5"
model = load_model(modelFileName,custom_objects={
                   'relu6': relu6,
                   'DepthwiseConv2D': convolutional.DepthwiseConv2D})

# this line makes code not crashed
result = model.predict(np.zeros((1,224,224,3)))  

startCheck = False
continuousSafetyCheckStartFlag = False


totalTime = 0
imagePredictionCount = 0
safeCount = 0
unSafeCount = 0
chekc_elevator_time_start = 0
chekc_elevator_time_end = 0
continuousCheckImageDetectionCountThreshold = 2
continuousCheckUnsafeThreshold = 1
chekc_elevatorTimer = 3
continuousSafetyCheckTimer1 = 0.5
continuousSafetyCheckTimer2 = 3


def imagePrediction(data):
    global safeCount, unSafeCount, imagePredictionCount, startCheck, totalTime, continuousSafetyCheckStartFlag, continuousCheckUnsafeThreshold
    
    if not startCheck:
        return
        
    start1  = time.time()
    imagePredictionCount = imagePredictionCount + 1
    
    try:
        cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
        cv2_img = cv2.resize(cv2_img, (224, 224))
        
    except CvBridgeError as e:
        print(e)
    else:
    
        np_img = np.reshape(cv2_img, (1,224,224,3)).astype('float32')
        np_img_normalized = np_img/255
        
        prediction = model.predict(np_img_normalized, verbose=0)
        label = prediction.argmax(axis=-1)
        #print ("prediction = ",prediction)
        #print ('result = ', label[0])
        end2 = time.time()

        #0: safe, 1: unsafe
        if label[0] == 0:
            safeCount = safeCount + 1
        elif label[0] == 1:
            unSafeCount = unSafeCount + 1
        
        print("[Prediction]",'time:',totalTime ,"safeCount =",safeCount ,"unsafeCount =",unSafeCount, "Duration =", end2 - start1)
        totalTime = totalTime + (end2 - start1)

        if continuousSafetyCheckStartFlag == True:
            
            lock.acquire()
            if unSafeCount >= continuousCheckUnsafeThreshold:
                print("continuousSafetyCheckT1CB: unsafey!!!!")
                continuousSafetyCheckResultUnsafePub.publish(True)
                continuousSafetyCheckStartFlag = False
                startCheck = False
            lock.release()
   
    return

def resetCounter():
    global safeCount, unSafeCount, imagePredictionCount, totalTime, chekc_elevator_time_start , chekc_elevator_time_end

    safeCount = 0
    unSafeCount = 0
    imagePredictionCount = 0
    totalTime = 0
    chekc_elevator_time_start = 0
    chekc_elevator_time_end = 0

    return

def chekc_elevator(msg):
    global startCheck, chekc_elevator_time_start
    print("[chekc_elevator]")
    resetCounter()
    chekc_elevator_time_start = time.time()
    
    
    startCheck = True   
    t = Timer(chekc_elevatorTimer ,chekc_elevatorCB)
    t.daemon = True
    t.start()
    
    return

def chekc_elevatorCB():
    global safeCount, unSafeCount, startCheck, checkCBPub
    global imagePredictionCount, chekc_elevator_time_start, chekc_elevator_time_end, totalTime

    startCheck = False
    print ('[chekc_elevatorCB] result(safe, unSafe): ' + str(safeCount) + ', ' + str(unSafeCount))
    print ('[chekc_elevatorCB] count: ' + str(imagePredictionCount) )
    
    chekc_elevator_time_end = time.time()
    
    if totalTime == 0:
        totalTime = chekc_elevator_time_end - chekc_elevator_time_start

    print ('[chekc_elevatorCB] time: ' + str(totalTime) )
    if imagePredictionCount == 0:
        rospy.logerr("Count of Image is ZERO, Please check camera!!")
        rospy.loginfo("T0 report unsafe")
        checkCBPub.publish(False)
        return

    checkResult = float(safeCount) / float(imagePredictionCount - 1)
    if checkResult >= 0.9:
        checkCBPub.publish(True)
    else:
        checkCBPub.publish(False)
    
    
    return

def continuousSafetyCheckStart(msg):
    global startCheck, continuousSafetyCheckStartFlag, chekc_elevator_time_start

    resetCounter()
    chekc_elevator_time_start = time.time()
    continuousSafetyCheckStartFlag = True
    startCheck = True
    
    t1 = Timer(continuousSafetyCheckTimer1 ,continuousSafetyCheckT1CB)
    t1.daemon = True
    t1.start()

    t2 = Timer(continuousSafetyCheckTimer2 ,continuousSafetyCheckT2CB)
    t2.daemon = True
    t2.start()

    return


def continuousSafetyCheckT1CB():
    global imagePredictionCount 
    global continuousSafetyCheckStartFlag
    global continuousCheckImageDetectionCountThreshold
    global chekc_elevator_time_start, chekc_elevator_time_end

    
    if imagePredictionCount < continuousCheckImageDetectionCountThreshold:
        lock.acquire()
        if continuousSafetyCheckStartFlag:
            chekc_elevator_time_end = time.time()
            totalTime = chekc_elevator_time_end - chekc_elevator_time_start
            rospy.logerr("Count of image is ZERO, Please check camera!!")
            rospy.loginfo("continuous T1 report unsafe")
            rospy.loginfo("[TotalTime] : " + str(totalTime))
            continuousSafetyCheckResultUnsafePub.publish(True)
        else:
            rospy.loginfo("T1: report unsafe, but ignore publishing it, ImageDetectionCount = " + str(imagePredictionCount))
        lock.release()
    
    return

def continuousSafetyCheckT2CB():
    global startCheck , continuousSafetyCheckStartFlag
    
    startCheck = False
    continuousSafetyCheckStartFlag = False
    
    return
         

def main(args):
    rospy.init_node('ev_safty_check_test', anonymous=True)
    #image_sub = rospy.Subscriber("/camera_rear/image_rect_color", Image, imagePrediction, queue_size=1 ,buff_size=5000000)
    image_sub = rospy.Subscriber("/usb_cam/image_rect_color", Image, imagePrediction, queue_size=1 ,buff_size=5000000)
    rospy.Subscriber('/checkEV',Bool,chekc_elevator)
    rospy.Subscriber('/continuousSafetyCheckStart', Bool, continuousSafetyCheckStart)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
