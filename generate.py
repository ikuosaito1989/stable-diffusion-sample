from diffusers import StableDiffusionPipeline
import os
from face_mosaic_mediapipe import mosaic_face

# ==== 設定 ====
prompt = (
    "Full body photo, handsome, 25 year old, Japanese male, golden blonde hair, 170cm, 70kg, "
    "symmetrical face, styled hair, strong facial features, medium length hair, black t-shirt, blue wide pants, wearing Nike high cut sneakers,"
    "urban background, model like proportions, Detailed photo, full body in frame, low angle"
)

negative_prompt = (
    "cropped, closeup, missing legs, missing feet, blurry, distorted hands, distorted feet, watermark"
)

output_path = "images/aj1_japanese_man19.png"

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
