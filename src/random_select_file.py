import os, random
import shutil

ImagePath = "/home/jimmy/ev_safety_check/image/preprossedImg/alldata/gray/fewpeople/"
MoveImgTo = "/home/jimmy/ev_safety_check/image/preprossedImg/train/fewpeople/"

def moveFile(fname):
    shutil.move(ImagePath + fname ,MoveImgTo + fname)

for i in range(12320):
    fname = random.choice(os.listdir(ImagePath))
    print("i =",i,"fname =",fname)
    moveFile(fname)