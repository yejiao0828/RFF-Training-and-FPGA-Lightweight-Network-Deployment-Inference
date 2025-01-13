'''
 Evaluate the quantized model
 Author: chao.zhang
'''

import os
 
# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_DIR = '../output'
QAUNT_MODEL = 'quantized_model.h5'
image_dir = '../code/test_spectrograms.npy'
image_size = (102,63)
image_shape = (102,63,1)

# Load the quantized model
print('\nLoad quantized model..')
path = os.path.join(MODEL_DIR, QAUNT_MODEL)
with vitis_quantize.quantize_scope():
    model = models.load_model(path)

# Compile the model
print('\nCompile model..')
model.compile(optimizer="rmsprop", 
        loss="categorical_crossentropy",
        metrics=['accuracy']
        )

# get dataset
data = np.load('test_spectrograms.npy', allow_pickle=True).item()
print(type(data))  # 确认 data 是字典类型
print(data.keys())  # 确认字典中包含的键
test_x = data['spectrograms']
test_y = data['labels']
test_labels = keras.utils.to_categorical(test_y, num_classes=25)
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_labels)).batch(64)

# Evaluate model with test data
print("\nEvaluate model on test Dataset")
loss, acc = model.evaluate(test_dataset)  # returns loss and metrics
print("loss: %.3f" % loss)
print("acc: %.3f" % acc)
