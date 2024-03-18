import cv2 
import mediapipe as mp 
import numpy as np 
from data_collection.detection import mediapipe_detection, draw_styled_landmarks
from data_collection.extract_feature import extract_coordinates
from tensorflow.keras.models import load_model
from settings import COLORS

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def prob_viz(res, actions, input_frame, COLORS):
    '''
    Rendering probability of each action occuring for model evaluation
    '''
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), COLORS[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

def inference(actions):
    model = load_model('model/model.h5')
    
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # extract sequences from current recorded actions 
            keypoints = extract_coordinates(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]   
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0] # predict action 
                print(actions[np.argmax(res)]) # obtain the action that is has the highest probability 
                predictions.append(np.argmax(res)) # obtain index of action 
                
                if np.unique(predictions[-10:])[0]==np.argmax(res): # check that the last 10 predictions on the frame are the same 
                    if res[np.argmax(res)] > threshold: # if prediction above confidence 
                        
                        # appends the current action to the sentence for tracking purposes 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # add probability distribution 
                image = prob_viz(res, actions, image, COLORS)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    actions = np.array(['hello', 'thanks', 'iloveyou'])
    inference(actions)