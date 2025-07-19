import numpy as np
from PIL import Image
import mediapipe as mp
import cv2

def mosaic_face(pil_image):
    image = np.array(pil_image)
    h, w, _ = image.shape

    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            x = max(0, x)
            y = max(0, y)

            face = image[y:y+height, x:x+width]
            face = cv2.resize(face, (10, 10), interpolation=cv2.INTER_LINEAR)
            face = cv2.resize(face, (width, height), interpolation=cv2.INTER_NEAREST)
            image[y:y+height, x:x+width] = face
            break  # 最初の1人だけでOKなら break

    return Image.fromarray(image)
