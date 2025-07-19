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

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3
    )

    # 面積が大きい顔だけを対象にする（最も大きなもの1つ）
    if len(faces) > 0:
        # 面積順に並べ替え（大きい順）
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)

        for (x, y, w, h) in faces:
            print(f"Detected face: {x}, {y}, {w}, {h}")
            # 最小サイズ制限（小さすぎる領域はスキップ）
            # if w < 50 or h < 50:
            #     continue

            # 顔部分を縮小→拡大でモザイク化
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (10, 10), interpolation=cv2.INTER_LINEAR)
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
            image[y:y+h, x:x+w] = face
            break  # 最も大きい顔だけモザイク

    return Image.fromarray(image)

# ==== 設定 ====
prompt = (
    "Full-body photo of a 20-year-old Japanese man, standing in front of a wall, "
    "wearing white and black t-shirt, slim pants, and Air Jordan 1 sneakers. "
    "His face is blurred or turned away. Realistic street fashion, visible shoes, full legs, "
    "lower body and feet included, detailed footwear, full body visible, high detail, low angle"
)

negative_prompt = (
    "cropped, closeup, missing legs, missing feet, blurry, distorted hands, distorted feet, watermark"
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
    height=904,
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
