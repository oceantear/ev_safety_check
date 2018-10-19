import os, random
import shutil

ImagePath = "/home/jimmy/ev_safety_check/image/preprossedImg/20180928/original_color_gray/nopeople/color/"
MoveImgTo = "/home/jimmy/ev_safety_check/image/preprossedImg/20180928/forTrainging/test/color_gray/color/nopeople/"

def moveFile(fname):
    shutil.move(ImagePath + fname ,MoveImgTo + fname)

for i in range(18):
    fname = random.choice(os.listdir(ImagePath))
    print("i =",i,"fname =",fname)
    moveFile(fname)