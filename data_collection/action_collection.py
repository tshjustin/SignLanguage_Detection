import os 
import cv2
import numpy as np 
import mediapipe as mp
from settings import DATA_PATH
from detection import mediapipe_detection, draw_styled_landmarks
from extract_feature import extract_coordinates
from settings import FRAMES, SEQUENCE

actions = np.array(['hello', 'thanks', 'iloveyou']) # actions to detect 

mp_holistic = mp.solutions.holistic # detection 
mp_drawing = mp.solutions.drawing_utils # drawing model 

cap = cv2.VideoCapture(0) # opens webcam & reads feed 

def create_folder():
    for action in actions: 
        for sequence in range(SEQUENCE):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

def collection(actions):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: # set mediapipe model  - set initial detection conf then subsequent tracking conf 
        for action in actions:
        # Loop through videos
            for sequence in range(SEQUENCE):
                # Loop through each frame of the video 
                for frame_num in range(FRAMES):

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # frame break - each action would be outputed followed by break to allow for data collection 
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200),  # tracking the frames 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500) # break 
                        
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                    
                    # export the coordianates that are captured 
                    keypoints = extract_coordinates(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break with 'q'
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
    cap.release()
    cv2.destroyAllWindows()
                        
if __name__ == '__main__':
    collection(actions, FRAMES, SEQUENCE)