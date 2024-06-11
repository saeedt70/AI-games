import cv2
import mediapipe as mp
create = None
opname = "output.avi"
cap = cv2.VideoCapture(0)
mpMesh = mp.solutions.face_mesh 
Mesh=mpMesh.FaceMesh()
mpdraw = mp.solutions.drawing_utils

while True:
      success,img = cap.read()
      if not success:break
      cv2.flip(img,1)
      imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      res=Mesh.process(imgRGB)
      
      if res.multi_face_landmarks:
            for face_landmark in res.multi_face_landmarks:
                  mpdraw.draw_landmarks(img,face_landmark)
      cv2.imshow('Image',img)
      if create is None:
         fourcc = cv2.VideoWriter_fourcc(*'XVID')
         create = cv2.VideoWriter(opname, fourcc, 30, (img.shape[1], img.shape[0]), True)
      create.write(img)
      if cv2.waitKey(1) & 0xff==ord('q'):
            break
