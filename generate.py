from diffusers import StableDiffusionPipeline
import torch
import os

# ==== 設定 ====
prompt = "a 20-year-old Japanese man wearing Air Jordan 1 sneakers, white t-shirt, dark blue jeans, full body, standing, realistic photo, street fashion, urban background"
negative_prompt = "blurry, lowres, ugly face, distorted body, bad anatomy, watermark, text"
output_path = "aj1_japanese_man.png"

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
