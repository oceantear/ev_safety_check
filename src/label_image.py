#!/usr/bin/env python
import os
import sys
import cv2
from pynput.keyboard import Key, Listener
import sys, termios, tty, os, time
import shutil

ImagePath = "/home/jimmy/ev_safety_check/image/preprossedImg/alldata/gray/"
MoveImgTo = "/home/jimmy/catkin_ws/src/ev_safety_check/Img/preprossedImg/test/safe/"
ControversialImgPath = "/home/jimmy/ev_safety_check/image/preprossedImg/controversial/"
button_delay = 0.2

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
 
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def moveFile(fname):
    shutil.move(ImagePath + fname ,MoveImgTo + fname)

def moveControversialFile(fname):
    shutil.move(ImagePath + fname ,ControversialImgPath + fname)

i = 0
def main(args):
    for fname in os.listdir( ImagePath ):
        global i
        if fname.endswith( "jpeg" ):
            print '====== ',i,' ======'
            print 'fname :',fname
            
            srcImage = cv2.imread( ImagePath + fname )
            cv2.imshow('srcImage',srcImage)
            cv2.waitKey(10)
            keyboardInput = getch()
            print  keyboardInput
            if keyboardInput == "q":
                time.sleep(button_delay)
                break
            elif keyboardInput == "m" :
                print 'move Image from :',ImagePath, ' to: ',MoveImgTo
                moveFile(fname)
            elif keyboardInput == "c" :
                print 'move Image from :',ImagePath, ' to: ',ControversialImgPath
                moveControversialFile(fname)
            else:
                print  keyboardInput
                time.sleep(button_delay)
                
            i += 1

if __name__ == '__main__':
    #main(sys.argv)
    #main(sys.argv)
    main(sys.argv)

