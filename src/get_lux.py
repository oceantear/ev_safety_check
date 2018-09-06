#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from PIL import Image ,ImageStat

def main(args):
    #imag = Image.open("")
    #imag = imag.convert('RGB')
    
    '''for x in 336:
        for y in 252:
            pixelRGB = imag.getpixel((x,y))
            R = R + pixelRGB
            G = G + pixelRGB
            B = B + pixelRGB

    brightness = sum([R,G,B])/3
    '''
    '''
    /media/jimmy/DATA/ROS_bag_files/CITIGO_bagfile/EVWB1_image_rect_color_screenshot_30.08.2018.png             111             
    /media/jimmy/DATA/ROS_bag_files/CITIGO_bagfile/EVWB1_image_rect_color_screenshot_30.08.2018_2.png           113              
    /media/jimmy/DATA/ROS_bag_files/CITIGO_bagfile/EVWB3_image_rect_color_screenshot_30.08.2018.png             117
    /media/jimmy/DATA/ROS_bag_files/CITIGO_bagfile/EVWB3_image_rect_color_screenshot_30.08.2018_2.png           116
    /home/jimmy/Documents/image_rect_color_screenshot_04.09.2018.png                                            130
    /home/jimmy/Documents/image_rect_color_screenshot_06.09.2018.png                                            139
    /home/jimmy/Documents/image_rect_color_screenshot_06.09.2018_6.png                                          129
    '''

    imag = Image.open("/home/jimmy/Documents/black.png")                       
    stat = ImageStat.Stat(imag)
    print("average pixel brightness",stat.mean[0])


    return


if __name__ == '__main__':
    main(sys.argv)