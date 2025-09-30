import cv2
import mediapipe as mp
import time


class hand_detector():
    def __init__(self,mode=False,maxhand=4,detection_con=0.5,tracking_con=0.5):    
        self.mode=mode
        self.detection_con=detection_con
        self.maxhand=maxhand
        self.tracking_con=tracking_con
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands()
        self.lines=mp.solutions.drawing_utils

    def findhands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #shows the dots in the hand
        self.performance=self.hands.process(imgRGB)
        if self.performance.multi_hand_landmarks:
            #shows this for each hand
            for number in self.performance.multi_hand_landmarks:
                if draw:
                    # we display img not RGB
                    self.lines.draw_landmarks(img,number, self.mpHands.HAND_CONNECTIONS)
        return(img)


    def find_location(self,img, handnum=0, draw=True):

        location=[]
        if self.performance.multi_hand_landmarks:
            myhand=self.performance.multi_hand_landmarks[handnum]
            for id,position in enumerate(myhand.landmark):
                # The position of points of our hand ratio of the image
                #finding the pixel values of our hands
                h,w,c=img.shape
                cx,cy=int(position.x*w),int(position.y*h)
                #print(id,cx,cy)

                location.append([id,cx,cy])
                if id==12:
                    cv2.circle(img, (cx,cy), 10, (160,0,160),cv2.FILLED)
        return location



def main():
    detector=hand_detector()
    #Choosing our webcam
    camera=cv2.VideoCapture(0)
    #this module connects the lines
    #mpHands=mp.solutions.hands
    # Track the lines between the points of your hand
    #lines=mp.solutions.drawing_utils
    #this function only uses RGB so convert your image to RGB
    #hands=mpHands.Hands(False,4)
    # running our webcam
    while True:
        success,img=camera.read()
        img=detector.findhands(img)
        list=detector.find_location(img)
        if len(list)!=0:
            print(list)
        #shows the dots in the hand
        # to know how many hands we have
        cv2.imshow("image",img)
        # Turns a frame into a video
        cv2.waitKey(1)

if __name__ =="__main__":
    main()

