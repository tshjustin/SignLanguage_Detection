import cv2
import mediapipe as mp
from detection import mediapipe_detection, draw_landmarks, draw_styled_landmarks

mp_holistic = mp.solutions.holistic # detection 
mp_drawing = mp.solutions.drawing_utils # drawing model 

cap = cv2.VideoCapture(0) # opens webcam & reads feed 

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: # set mediapipe model  - set initial detection conf then subsequent tracking conf 
    while cap.isOpened(): 
        ret, frame = cap.read() # obtain still frame 
        
        image, results = mediapipe_detection(frame, holistic)
        # print(results)
        
        draw_styled_landmarks(image, results)
        cv2.imshow('OpenCV Feed', image) 
        
        if cv2.waitKey(10) & 0xFF == ord('q'): # break key 
            break

    cap.release() # release webcam 
    cv2.destroyAllWindows()

results.count
print(results.count)