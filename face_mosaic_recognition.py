import numpy as np
from PIL import Image
import face_recognition
import cv2

def mosaic_face(pil_image):
    image = np.array(pil_image)
    rgb_image = image[:, :, ::-1]  # BGR → RGB

    # 顔位置の検出
    face_locations = face_recognition.face_locations(rgb_image, model='hog')  # 'cnn' はGPU推奨

    for top, right, bottom, left in face_locations:
        # モザイク処理
        face = image[top:bottom, left:right]
        if face.size == 0:
            continue  # 念のためスキップ

        face = cv2.resize(face, (10, 10), interpolation=cv2.INTER_LINEAR)
        face = cv2.resize(face, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
        image[top:bottom, left:right] = face

    return Image.fromarray(image)
