import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHand = mp.solutions.hands
Hand=mpHand.Hands(min_detection_confidence=0.80)
mpdraw = mp.solutions.drawing_utils

while True:
      success,img = cap.read()
      if not success:break
      cv2.flip(img,1)
      h,w,c=img.shape
      imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      res=Hand.process(imgRGB)
      #print(res.multi_handedness)
      if res.multi_handedness:
            for hand in res.multi_hand_landmarks:
                  for id,ld in enumerate(hand.landmark):
                         if id == 8:
                           x=int(ld.x*w)
                           y=int(ld.y*h)
                           
                           cv2.circle(img,(x,y),10,(0,255,0),3)
            mpdraw.draw_landmarks(img,hand,mpHand.HAND_CONNECTIONS)
           
            
      cv2.imshow('Image',img)
      if cv2.waitKey(1) & 0xff==ord('q'):
            break
