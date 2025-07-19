from diffusers import StableDiffusionPipeline
import torch
import os

# ==== 設定 ====
prompt = "20 year old 175cm Japanese male wearing Air Jordan 1, white t-shirt, full body, standing, realistic photo, street fashion, urban background, mosaic face, brown hair"
negative_prompt = "blurry, lowres, ugly face, distorted body, bad anatomy, watermark, text"
output_path = "images/aj1_japanese_man2.png"

# ==== モデル読み込み ====
print("🔁 Loading model...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cpu")

# ==== 画像生成 ====
print("🎨 Generating image... (this may take several minutes)")
image = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=7.5).images[0]

# ==== 保存 ====
image.save(output_path)
print(f"✅ Image saved to: {output_path}")

# ==== Macで画像を開く（オプション） ====
if os.name == "posix":
    os.system(f"open {output_path}")
