from diffusers import StableDiffusionPipeline
import torch
import os

# ==== 設定 ====
prompt = (
    "A full-body photo of a 20-year-old Japanese man standing against an urban wall, "
    "wearing Air Jordan 1 sneakers, white t-shirt, slim dark jeans, street fashion, "
    "175cm tall, pixelated or blurred face, brown hair, realistic photo"
)

negative_prompt = (
    "blurry, lowres, ugly face, distorted body, bad anatomy, watermark, text, cropped, closeup"
)

output_path = "images/aj1_japanese_man4.png"

# ==== モデル読み込み ====
print("🔁 Loading model...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cpu")

pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=768,
    width=512,
    guidance_scale=7.5,
    num_inference_steps=40
).images[0]

# ==== 保存 ====
image.save(output_path)
print(f"✅ Image saved to: {output_path}")

# ==== Macで画像を開く（オプション） ====
if os.name == "posix":
    os.system(f"open {output_path}")
