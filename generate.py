from diffusers import StableDiffusionPipeline
import os
from face_mosaic_mediapipe import mosaic_face

# ==== 設定 ====
prompt = (
    "Full body photo of a handsome 20 year old Japanese male, 175cm 66kg, "
    "clean skin, symmetrical face, sharp jawline, styled hair, wearing a black T-shirt, wide pants, Nike sneakers, "
    "K-pop idol style, model-like proportions, street fashion, highly detailed, full body in frame, low angle"
)

negative_prompt = (
    "cropped, closeup, missing legs, missing feet, blurry, distorted hands, distorted feet, watermark"
)

output_path = "images/aj1_japanese_man15.png"

# ==== モデル読み込み ====
print("🔁 Loading model...")
pipe = pipe = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V5.1_noVAE")

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
