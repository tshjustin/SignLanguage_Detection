import os 
from tensorflow.keras.callbacks import TensorBoard

# +------------+
# | Constants  |
# +------------+
DATA_PATH = os.path.join('MP_Data')
EPCOHS = 2000
COLORS = [(245,117,16), (117,245,16), (16,117,245)]

# +------------+
# | Logging    |
# +------------+
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir) # trace training of Neural Network