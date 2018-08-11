import os, random
import shutil

ImagePath = "/home/jimmy/ev_safety_check/image/preprossedImg/alldata/nopeople/"
MoveImgTo = "/home/jimmy/ev_safety_check/image/preprossedImg/test/nopeople/"

def moveFile(fname):
    shutil.move(ImagePath + fname ,MoveImgTo + fname)

for i in range(2800):
    fname = random.choice(os.listdir(ImagePath))
    print("i =",i,"fname =",fname)
    moveFile(fname)