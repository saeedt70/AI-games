import cv2
import mediapipe as mp
import numpy as np
mp_selfie_segmentation=mp.solutions.selfie_segmentation
cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:break
    image=cv2.flip(image,1)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    selfie_segmentation=mp_selfie_segmentation.SelfieSegmentation()
    results = selfie_segmentation.process(image)
    condition=np.stack((results.segmentation_mask,)*3,axis=-1)>0.8
    bg_image=cv2.imread("1.jpg")
    bg_image=cv2.resize(bg_image,(image.shape[1],image.shape[0]))
    output_image=np.where(condition,image,bg_image)
    cv2.imshow('Image',output_image)
    if cv2.waitKey(1) & 0xff==ord('q'):
               break
