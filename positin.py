import cv2
import mediapipe
import math
def getpose(data,h: int,w: int)->list:
      out=list()
      for id,lm in enumerate(data.landmark):
          out.append((int(lm.x*w),int(lm.y*h)))
      return out
def getangle(data,pt1: int,pt2: int,pt3: int)->int:
      x1,y1=data[pt1]
      x2,y2=data[pt2]
      x3,y3=data[pt3]
      angle=math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
      return(angle)



cap = cv2.VideoCapture(0)
pose = mediapipe.solutions.pose.Pose()



while True:
      success,img = cap.read()
      if not success:break
      imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      res=pose.process(imgRGB)
      h,w,c=img.shape
      #print(res.multi_handedness)
      if res.pose_landmarks:
           #print(res.pose_landmarks)
           data=getpose(res.pose_landmarks,h,w)
           #print(data)
           angle=getangle(data,12,14,16)
           cv2.circle(img,data[12],5,(255,0,255),cv2.FILLED)
           cv2.circle(img,data[14],5,(255,0,255),cv2.FILLED)
           cv2.circle(img,data[16],5,(255,0,255),cv2.FILLED)
           if angle <= 15:
                 print('hi')

           
           
      cv2.imshow('Image',img)
      if cv2.waitKey(1) & 0xff==ord('q'):
            break
