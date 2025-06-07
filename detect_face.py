import cv2
import tensorflow as tf
# from tensorflow.keras.models import load_model
import numpy as np
from keras._tf_keras.keras.models import load_model

model_predict = load_model('deepfake_3conv_16-32-64_dense_512.keras')

# Đường dẫn tới ảnh cần nhận diện khuôn mặt
image_path = r".\chup-anh-nhom-tuyet-dep.jpg"

# Load ảnh
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade cho nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Phát hiện các khuôn mặt
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

max_risk = 0

# Kiểm tra xem có phát hiện được khuôn mặt nào không
if len(faces) == 0:
    print("Không phát hiện được khuôn mặt nào.")
    max_risk = -1
else:
    # Duyệt qua từng khuôn mặt được phát hiện
    for i, (x, y, w, h) in enumerate(faces):
        # Cắt khuôn mặt từ ảnh gốc
        face = image[y:y+h, x:x+w]
        
        # Nhận diện real/deepfake bằng model
        resize = tf.image.resize(face, (224,224))
        y_pred = 1 - model_predict.predict(np.expand_dims(resize/255, 0))
        print(y_pred)
        if y_pred > 0.5:
            print("Fake")
        else:
            print("Real")
        max_risk = max(y_pred,max_risk)

print("Xac suat lon nhat: ", max_risk) #xác suất tổng quát

        # # Đường dẫn để lưu ảnh khuôn mặt
        # face_image_path = f"face_{i+1}.jpg"

        # # Lưu ảnh khuôn mặt
        # cv2.imwrite(face_image_path, face)
        # print(f"Khuôn mặt {i+1} đã được lưu tại: {face_image_path}")
