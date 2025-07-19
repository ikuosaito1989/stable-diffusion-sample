from diffusers import StableDiffusionPipeline
import torch
import os
import cv2
import numpy as np
from PIL import Image

# ==== モザイク処理関数 ====
def mosaic_face(pil_image):
    image = np.array(pil_image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (10, 10), interpolation=cv2.INTER_LINEAR)
        face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = face

    return Image.fromarray(image)

# ==== 設定 ====
prompt = (
    "Full-body photo of a 20-year-old Japanese man wearing Air Jordan 1 sneakers, "
    "white oversized t-shirt, dark slim jeans, standing against a wall, street fashion, "
    "realistic photo, face obscured or not visible, looking away, low angle"
)

negative_prompt = (
    "closeup, blurry, cropped, watermark, distorted legs, ugly face, looking at camera, face visible"
)

output_path = "images/aj1_japanese_man4.png"

# ==== モデル読み込み ====
print("🔁 Loading model...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cpu")

pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

# ==== 画像生成 ====
print("🎨 Generating image...")
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=768,
    width=512,
    guidance_scale=7.5,
    num_inference_steps=40
).images[0]

# ==== モザイク処理 ====
image = mosaic_face(image)

# ==== 保存 ====
image.save(output_path)
print(f"✅ Image saved to: {output_path}")

# ==== Macで画像を開く（オプション） ====
if os.name == "posix":
    os.system(f"open {output_path}")
