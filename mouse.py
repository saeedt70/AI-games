import cv2
import mediapipe as mp
from pynput import mouse
mose=mouse.Controller()
from math import hypot
cap = cv2.VideoCapture(0)
import time

mpHand = mp.solutions.hands
Hand=mpHand.Hands( max_num_hands=1,min_detection_confidence=0.80)
mpdraw = mp.solutions.drawing_utils

while True:
      success,img = cap.read()
      if not success:break
      img=cv2.flip(img,1)
      img=cv2.resize(img, (1280, 720))
      h,w,c=img.shape
      imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      res=Hand.process(imgRGB)

      lmList = []
      if res.multi_hand_landmarks:
         for handlandmark in res.multi_hand_landmarks:
            for id,lm in enumerate(handlandmark.landmark):
                h,w,_ = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy]) 
            mpdraw.draw_landmarks(img,handlandmark,mpHand.HAND_CONNECTIONS)
    
      if lmList != []:
        x1,y1 = lmList[12][1],lmList[12][2]
        x2,y2 = lmList[10][1],lmList[10][2]
        #
        xx1,yy1 = lmList[4][1],lmList[4][2]
        xx2,yy2 = lmList[5][1],lmList[5][2]
        
        #cv2.circle(img,(x1,y1),4,(255,0,0),cv2.FILLED)
        #cv2.circle(img,(x2,y2),4,(255,0,0),cv2.FILLED)
        #cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)

        length = hypot(x2-x1,y2-y1)
        length1 = hypot(xx2-xx1,yy2-yy1)
        if(length<15 or length1<25):
          mose.click(mouse.Button.left,1)
          time.sleep(1)
         
      
      #print(res.multi_handedness)
      if res.multi_handedness:
            for hand in res.multi_hand_landmarks:
                  for id,ld in enumerate(hand.landmark):
                         if  id == 0:
                           x=int(ld.x*w)
                           y=int(ld.y*h)
                           
                           #cv2.circle(img,(x,y),10,(0,255,0),3)
                           #print(x)
                           mose.position=(x+15,600)

 
            #mpdraw.draw_landmarks(img,hand,mpHand.HAND_CONNECTIONS)
           
            
      #cv2.imshow('Image',img)
      if cv2.waitKey(1) & 0xff==ord('q'):
            break
