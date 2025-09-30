import cv2
import mediapipe as mp
import os
import time
import Handtracking as ht

camera=cv2.VideoCapture(0)
overlaylist=[]
folderpath="Fingers"
mylist=os.listdir(folderpath)
print(mylist)

for impath in mylist:
    image=cv2.imread(f"{folderpath}/{impath}")
    resized=cv2.resize(image,(200,200))
    overlaylist.append(resized)
previous_time=0

detector=ht.hand_detector()

Tipids=[4,8,12,16,20]
while True:
    success,img=camera.read()
    img=detector.findhands(img)
    location=detector.find_location(img, draw=False)
    

    if location:

        fingers=[]

        if location[Tipids[0]][1] > location[Tipids[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for i in range(1,5):
            if location[Tipids[i]][2] < location[Tipids[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalfingers=fingers.count(1)
        print(totalfingers)
        h,w,c=overlaylist[totalfingers-1].shape
        img[0:h,0:w]=overlaylist[totalfingers-1]
        cv2.rectangle(img,(20,225),(170,425), (0,255,0),cv2.FILLED)
        cv2.putText(img, str(totalfingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)
    current_time=time.time()
    fps=1/(current_time-previous_time)
    cv2.putText(img,f'FPS: {int(fps)}',(400,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0), 3)
    previous_time=current_time
    cv2.imshow("image",img)
    cv2.waitKey(1)
