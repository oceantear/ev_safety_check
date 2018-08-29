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
modelFileName = packPath + "/models/4labels_AndewNDataClassfied_1280dense.h5"
model = load_model(modelFileName,custom_objects={
                   'relu6': relu6,
                   'DepthwiseConv2D': convolutional.DepthwiseConv2D})

# this line makes code not crashed
result = model.predict(np.zeros((1,224,224,3)))  

startCheck = False
continuousSafetyCheckStartFlag = False
chekc_elevator_function_flag = False
continuousSafetyCheck_function_flag = False



totalTime = 0
imagePredictionCount = 0
carCount = 0
fewpeopleCount = 0
manypeopleCount = 0
nopeopleCount = 0
someoneCount = 0

carScore = 1.0
fewPeopleScore = 1.0
manyPeopleScore = 1.0
noPeopleScore = 1.0

continuousSafetyCheckLPFGain = 0.8 #0.9
continuousSafetyCheckScore = 1.0

chekc_elevator_time_start = 0
chekc_elevator_time_end = 0
continuousCheckImageDetectionCountThreshold = 1
continuousCheckUnsafeThreshold = 1
chekc_elevatorTimer = 3
continuousSafetyCheckTimer1 = 1
continuousSafetyCheckTimer2 = 3

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

def imagePrediction(data):
    global carCount, fewpeopleCount, manypeopleCount, nopeopleCount, imagePredictionCount, startCheck, totalTime, continuousSafetyCheckStartFlag
    global continuousCheckUnsafeThreshold, continuousSafetyCheckScore, continuousSafetyCheckLPFGain, continuousSafetyCheckScore
    global carScore, fewPeopleScore, manyPeopleScore, noPeopleScore
    global someoneCount
    if not startCheck:
        return
        
    start1  = time.time()
    ImageTime = data.header.stamp.secs + (data.header.stamp.nsecs / math.pow(10,9))
    print("systemTime :", start1 ,"Imagetime sec :", ImageTime)
    SystemTime = start1
    print("time lag: ",start1 - ImageTime)

    formatImageTime = time.asctime( time.localtime(data.header.stamp.secs) )
    formatSystemTime = time.asctime( time.localtime(start1) )
    
    #print("systemTime :", systemTime ,"Imagetime sec :", imageTime)
    
    try:
        
        cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
        #cv2.imshow('original image',cv2_img)
        #cv3_img = cv2.resize(cv2_img, (224, 224))
        #cv2.imshow('directly resize',cv3_img)
        #cv2.waitKey(0)
        cv2_img = resizeKeepAspectRatio(cv2_img,(224,224),0)
        #cv2.imshow('my resize',cv4_img)
        #cv2.waitKey(0)
        
    except CvBridgeError as e:
        print(e)
    else:
    
        np_img = np.reshape(cv2_img, (1,224,224,3)).astype('float32')
        np_img_normalized = np_img/255
        
        prediction = model.predict(np_img_normalized, verbose=0)
        print("prediction = ",prediction)
        label = prediction.argmax(axis=-1)
        #safe : prediction[0][0] unsafe : prediction[0][1]
        #print ("prediction = ",prediction[0][0] ,", ",prediction[0][1])
        #print ('result = ', label[0])
        end2 = time.time()

        #0:car, 1:fewpeople, 2:manypeople, 3:nopeople
        #if label[0] == 0:
        #    carCount = carCount + 1
        #elif label[0] == 1:
        #    fewpeopleCount = fewpeopleCount + 1
        #elif label[0] == 2:
        #    manypeopleCount = manypeopleCount + 1
        #elif label[0] == 3:
        #    nopeopleCount = nopeopleCount + 1
	if label[0] == 0 or label[0] == 1 or label[0] == 2:
	   someoneCount = someoneCount + 1
	elif label[0] == 3:
	   nopeopleCount = nopeopleCount + 1
        
	
        totalTime = totalTime + (end2 - start1)
        print("[Pred]",'time:',"%.4f" % totalTime ,"carCount =",carCount,"fewCount =",fewpeopleCount ,"manyCount =",manypeopleCount, "noCount =",nopeopleCount)
        
        #continuousSafetyCheckScore = continuousSafetyCheckLPFGain * continuousSafetyCheckScore + (1.0 - continuousSafetyCheckLPFGain) * prediction[0][0]
          
	

        carScore = continuousSafetyCheckLPFGain * carScore + (1.0 - continuousSafetyCheckLPFGain) * prediction[0][0]
        fewPeopleScore = continuousSafetyCheckLPFGain * fewPeopleScore + (1.0 - continuousSafetyCheckLPFGain) * prediction[0][1]
        manyPeopleScore = continuousSafetyCheckLPFGain * manyPeopleScore + (1.0 - continuousSafetyCheckLPFGain) * prediction[0][2]
        noPeopleScore = continuousSafetyCheckLPFGain * noPeopleScore + (1.0 - continuousSafetyCheckLPFGain) * prediction[0][3]

        print("carScore =","%.4f" % carScore,"fewScore =","%.4f" % fewPeopleScore,"manyScore =","%.4f" % manyPeopleScore ,"noScore =","%.4f" % noPeopleScore)

        imagePredictionCount = imagePredictionCount + 1
        continuousSafetyCheckScore = someoneCount / imagePredictionCount
    
        print("someoneCount =",someoneCount,"imagePredictionCount =",imagePredictionCount ,"continuousSafetyCheckScore = ",continuousSafetyCheckScore)

        if continuousSafetyCheckStartFlag == True:
                        
            lock.acquire()
            if continuousSafetyCheckScore > continuousSafetyCheckLPFGain:
                print("continuousSafetyCheckT1CB: unsafey!!!!")
                continuousSafetyCheckResultUnsafePub.publish(True)
                continuousSafetyCheckStartFlag = False
                startCheck = False
            lock.release()
   
    return

def resetCounter():
    global carCount, fewpeopleCount, manypeopleCount, nopeopleCount, imagePredictionCount, totalTime, chekc_elevator_time_start , chekc_elevator_time_end, continuousSafetyCheckScore
    global carScore, fewPeopleScore, manyPeopleScore, noPeopleScore
    global someoneCount
    carCount = 0
    fewpeopleCount = 0
    manypeopleCount = 0
    nopeopleCount = 0
    imagePredictionCount = 0
    someoneCount = 0
    totalTime = 0
    chekc_elevator_time_start = 0
    chekc_elevator_time_end = 0
    continuousSafetyCheckScore = 1.0

    carScore = 1.0
    fewPeopleScore = 1.0
    manyPeopleScore = 1.0
    noPeopleScore = 1.0

    return

def chekc_elevator(msg):
    global startCheck, chekc_elevator_time_start, chekc_elevator_function_flag
    
    if chekc_elevator_function_flag == False:
        print("[chekc_elevator]")
        chekc_elevator_function_flag = True
        resetCounter()
        chekc_elevator_time_start = time.time()
        
        startCheck = True   
        t = Timer(chekc_elevatorTimer ,chekc_elevatorCB)
        t.daemon = True
        t.start()
    
    return

def chekc_elevatorCB():
    global startCheck, checkCBPub, chekc_elevator_function_flag
    global imagePredictionCount, chekc_elevator_time_start, chekc_elevator_time_end, totalTime
    global continuousSafetyCheckScore

    startCheck = False
    #print ('[chekc_elevatorCB] result(safe, unSafe): ' + str(safeCount) + ', ' + str(unSafeCount))
    #print("carScore =",carScore,"fewPeopleScore =",fewPeopleScore,"manyPeopleScore =",manyPeopleScore ,"noPeopleScore =",noPeopleScore)
    print ('[chekc_elevatorCB] count: ' + str(imagePredictionCount) )
    
    chekc_elevator_time_end = time.time()
    
    if totalTime == 0:
        totalTime = chekc_elevator_time_end - chekc_elevator_time_start

    print ('[chekc_elevatorCB] time: ' + str(totalTime) )
    if imagePredictionCount <= 1:
        rospy.logerr("Count of Image is ZERO, Please check camera!!")
        rospy.loginfo("T0 report unsafe")
        checkCBPub.publish(False)
        return

    
    '''checkResult = float(safeCount) / float(imagePredictionCount)
    if checkResult >= 0.9:
        checkCBPub.publish(True)
    else:
        checkCBPub.publish(False)
    '''
    if continuousSafetyCheckScore > continuousSafetyCheckLPFGain:
        checkCBPub.publish(False)
    else:
        checkCBPub.publish(True)

    
    chekc_elevator_function_flag = False

    return

def continuousSafetyCheckStart(msg):
    global startCheck, continuousSafetyCheckStartFlag, chekc_elevator_time_start, continuousSafetyCheck_function_flag

    if continuousSafetyCheck_function_flag == False:
        print("[continuousSafetyCheckStart]")
        continuousSafetyCheck_function_flag = True
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
            continuousSafetyCheckStartFlag = False
            rospy.logerr("Count of image is ZERO, Please check camera!!")
            rospy.loginfo("continuous T1 report unsafe")
            rospy.loginfo("[TotalTime] : " + str(totalTime))
            continuousSafetyCheckResultUnsafePub.publish(True)
        else:
            rospy.loginfo("T1: report unsafe, but ignore publishing it, ImageDetectionCount = " + str(imagePredictionCount))
        lock.release()
    
    return

def continuousSafetyCheckT2CB():
    global startCheck , continuousSafetyCheckStartFlag, continuousSafetyCheck_function_flag
    
    startCheck = False
    continuousSafetyCheckStartFlag = False
    continuousSafetyCheck_function_flag = False

    return
         

def main(args):
    rospy.init_node('ev_safty_check_test', anonymous=True)
    image_sub = rospy.Subscriber("/camera_rear/image_rect_color", Image, imagePrediction, queue_size=1 ,buff_size=5000000)
    #rospy.Subscriber("/usb_cam/image_rect_color", Image, imagePrediction, queue_size=1 ,buff_size=5000000)
    rospy.Subscriber('/checkEV',Bool,chekc_elevator)
    rospy.Subscriber('/continuousSafetyCheckStart', Bool, continuousSafetyCheckStart)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
