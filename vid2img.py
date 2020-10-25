import cv2
import os
from os import listdir

def vid_to_image(file_name):
    vidcap = cv2.VideoCapture(file_name)

    success, image = vidcap.read()
    count = 0
    new_name = file_name[0:-4]
    print(new_name)

    while success:

        cv2.imwrite(new_name + ("_%d.jpg" % count), image)
        success, image = vidcap.read()
        count += 1
        if (count % 100 == 0):
            print(count)


def dir_to_vid(directory_path):
    vid_files = [f for f in listdir(directory_path)]

    count = 0
    for f in vid_files:
        if f.endswith(".jpg"):
            continue
        if (count > 0):
            print("what: " + str(count) + "\n\n\n")
            vid_to_image(directory_path + "/" + f)
        count += 1

video_names = os.path.abspath("/home/conradli/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Clips")
print(video_names)
dir_to_vid(video_names + "/Deceptive")
#dir_to_vid(video_names + "/Truthful")