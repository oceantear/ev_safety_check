import os, random
import shutil

ImagePath = ""
MoveImgTo = ""

def moveFile(fname):
    shutil.move(ImagePath + fname ,MoveImgTo + fname)

for i in range(10):
    fname = random.choice(os.listdir("/home/advrobot/ev_safety_check/train/car"))
    print("fname =",fname)
    moveFile(fname)