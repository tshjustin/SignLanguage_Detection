import os 
import numpy as np 
import cv2
import mediapipe as mp
from detection import mediapipe_detection, draw_styled_landmarks
from extract_feature import extract_coordinates

DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['hello', 'thanks', 'iloveyou']) # actions to detect 

no_sequences = 30 # number of frames each sequence is represented with 

sequence_length = 30 # Videos are going to be 30 frames in length

mp_holistic = mp.solutions.holistic # detection 
mp_drawing = mp.solutions.drawing_utils # drawing model 

cap = cv2.VideoCapture(0) # opens webcam & reads feed 

def create_folder():
    for action in actions: 
        for sequence in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

def train(actions, sequence_length, no_sequences):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: # set mediapipe model  - set initial detection conf then subsequent tracking conf 
        for action in actions:
        # Loop through videos
            for sequence in range(no_sequences):
                # Loop through each frame of the video 
                for frame_num in range(sequence_length):

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
                        cv2.waitKey(2000) # break 
                        
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                    
                    # export the coordianates that are captured 
                    keypoints = extract_coordinates(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

if __name__ == '__main__':
    create_folder()