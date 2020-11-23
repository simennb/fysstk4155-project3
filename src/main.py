import numpy as np
import tensorflow as tf
import librosa

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
