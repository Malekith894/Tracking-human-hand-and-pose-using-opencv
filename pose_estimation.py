import mediapipe as mp
import cv2
import time

class posedetector():
    def __init__(self,mode=False, upbody=False, smooth=True, detection_con=0.5, 
    tracking_con=0.5):
        self.mode=mode
        self.upbody=upbody
        self.smooth=smooth
        self.detection_con=detection_con
        self.tracking_con=tracking_con
        #creating object
        self.mppose=mp.solutions.pose
        self.pose=self.mppose.Pose()
        #draws the dots
        self.mpDraw=mp.solutions.drawing_utils

    def findpose(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # process only takes RGB images
        self.performance=self.pose.process(imgRGB)
        # Drawing the dots
        if draw:
            if self.performance.pose_landmarks:
                #print(performance.pose_landmarks)
                self.mpDraw.draw_landmarks(img, self.performance.pose_landmarks,self.mppose.POSE_CONNECTIONS)
        return img
        
    def findposition(self,img, draw=True):
        self.location=[]
        #enumerate gives the numbrt of the loop as well
        if self.performance.pose_landmarks:
            for id, location in enumerate(self.performance.pose_landmarks.landmark):
                h,w,c=img.shape
                # to get the pixel values use int 
                cx,cy=int(w*location.x),int(h*location.y)
                self.location.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy),5,(255,0,0),cv2.FILLED )
                #print(id,location)
        return self.location
    

def main():
    camera=cv2.VideoCapture("pose_videos/1.mp4")
    previous_time=0
    detector=posedetector()

    while True:
        success,img=camera.read()
        detector.findpose(img)
        location=detector.findposition(img,draw=False)
        cv2.circle(img, (location[14][1],location[14][2]),15,(160,0,0),cv2.FILLED )
        print(location[0])
        # Resizing the frame
        img=cv2.resize(img, (860,860))
        #Calculation of fps
        current_time=time.time()
        fps=1/(current_time-previous_time)
        previous_time=current_time
        cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)
        cv2.imshow("image",img)
        cv2.waitKey(1)
if __name__=="__main__":
    main()
