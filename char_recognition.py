import numpy as np
import tensorflow as tf
import cv2

img_size = 128
batch_size = 32
model = None
labels = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def initializeModel(_model):
    global model
    model = tf.keras.models.load_model(_model)

def recognize(img):
    global model
    if(model==None): raise Exception('model not loaded')
    img_resized = cv2.resize(img,(img_size,img_size),interpolation = cv2.INTER_CUBIC)
    input_arr_batch = np.array([img_resized])
    result = model.predict(input_arr_batch)
    return labels[np.argmax(result)],np.argmax(result)