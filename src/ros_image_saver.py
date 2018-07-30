#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
count = 0

def callback(data):
    global count
    count = count + 1
    if count != 10:
        return
    count = 0

    try:
        cv2_image = bridge.imgmsg_to_cv2(data, "bgr8")
        #cv2_image_64x48 = cv2.resize(cv2_image, (64, 48))
        # resize, grey out ...etc
    except CvBridgeError as e:
        print(e)
    else:
        #cv2.imwrite('camera_image.jpeg', cv2_image)
        now = rospy.Time.now()
        fname = str(now) + ".jpeg"
        #print(str(now) + "_img.jpeg")
        cv2.imwrite(fname, cv2_image)
        

def main(args):
    rospy.init_node('image_saver', anonymous=True)
    image_sub = rospy.Subscriber("/camera_rear/image_rect_color", Image, callback)

    now = rospy.Time.now()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
