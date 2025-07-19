import cv2
import numpy as np
from PIL import Image

def mosaic_face(pil_image):
    image = np.array(pil_image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3
    )

    if len(faces) > 0:
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        for (x, y, w, h) in faces:
            print(f"Detected face: {x}, {y}, {w}, {h}")
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (10, 10), interpolation=cv2.INTER_LINEAR)
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
            image[y:y+h, x:x+w] = face
            break

    return Image.fromarray(image)
