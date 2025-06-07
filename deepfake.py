import tensorflow as tf
import numpy as np
from keras.models import load_model
import cv2


model = load_model("./deepfake_3conv_16-32-64_dense_512.keras")

img = cv2.imread("./real-vs-fake/test/fake/0ABCBZ0CWN.jpg")
resize = tf.image.resize(img, (224,224))

y_pred = 1 - model.predict(np.expand_dims(resize/255, 0))

print(y_pred) # Xác suất mặt người là có phải fake ko (fake -> 1)

if y_pred > 0.5:
    print("Fake") # Xác suất trên 0,5 là hàng fake
else:
    print("Real") # Xác suất dưới 0.5 là hàng real


