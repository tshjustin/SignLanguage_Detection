from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np 

actions = np.array(['hello', 'thanks', 'iloveyou'])

# create label map to map actions to its labels 
def create_labels(actions):
    label_map = {label:num for num, label in enumerate(actions)}
    return label_map

if __name__ == '__main__':
    a = create_labels(actions)
    print(a)