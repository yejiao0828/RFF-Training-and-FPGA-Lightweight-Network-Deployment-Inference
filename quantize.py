'''
 Quantize the float point model
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
FLOAT_MODEL = 'float_model.h5'
QAUNT_MODEL = 'quantized_model.h5'

# Load the floating point trained model
print('Load float model..')
path = os.path.join(MODEL_DIR, FLOAT_MODEL)
try:
    float_model = models.load_model(path)
    float_model.summary()
except:
    print('\nError:load float model failed!')

# get input dimensions of the floating-point model
height = float_model.input_shape[1]
width = float_model.input_shape[2]
print("\nmodel input size:", height, width)

# get the validation dataset for quantization
print("\nLoad validation dataset for quantization..")
data = np.load('test_spectrograms.npy', allow_pickle=True).item()
print(type(data))  # 确认 data 是字典类型
print(data.keys())  # 确认字典中包含的键
test_x = data['spectrograms']

# Run quantization
print('\nRun quantization..')
quantizer = vitis_quantize.VitisQuantizer(float_model)
quantized_model = quantizer.quantize_model(
        calib_dataset=test_x
        )

# Save quantized model
path = os.path.join(MODEL_DIR, QAUNT_MODEL)
quantized_model.save(path)
print('\nSaved quantized model as',path)