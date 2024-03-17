import cv2
import numpy as np 
import os 
from matplotlib import pyplot as plt 
import time 
import mediapipe as mp 

cap = cv2.VideoCapture(0) # opens webcam & reads feed 
 
while cap.isOpened(): 
    ret, frame = cap.read() # obtain still frame 
    cv2.imshow('OpenCV Feed', frame) 
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release() # release webcam 
cv2.destroyAllWindows()