!pip install gradio torch transformers diffusers accelerate lpips
import torch
from PIL import Image
import numpy as np
import gradio as gr
import lpips
from transformers import pipeline
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from skimage.metrics import structural_similarity as ssim
def load_models():
  depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
  controlnet = ControlNetModel.from_pretrained(
  "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16 )
  pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
  ).to("cuda")
  pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
  pipe.enable_attention_slicing(1)
  return depth_estimator, pipe
depth_estimator, pipe = load_models()
prompt_dict = {
  "Master Bedroom":"Enhance the bedroom interior while keeping all existing furniture and layout unchanged. Maintain the current wardrobe on the wall behind the bed, the window with metal grills on the right side, and the wall-mounted AC unit above the window. Keep the carved wooden bed in the same position. Improve the room by changing the bedsheet to a neatly spread pastel-colored or earthy-tone fabric with matching pillows. Add a soft rug under the bed. Replace the curtains with modern, semi-sheer ones that allow natural light. Slightly brighten the wall color while keeping it neutral (off-white or light beige). Add minimal wall art or a small painting beside the AC without removing anything. Enhance lighting subtly but realistically, maintaining natural window light and ceiling lighting. Keep everything else in the same place for realism.",
  #"Master Bedroom": "natural modern bedroom, luxury furniture, photorealistic materials, accurate shadows, 4K detail, architectural visualization, interior design magazine quality, professional lighting, nice walls","Children's Bedroom": "colorful children's bedroom with twin beds, playful decor, educational toys, soft pastel colors, cartoon-themed elements, study area, storage solutions, safety features,fun wallpaper, cozy atmosphere",
  "Hall": "modern spacious living room with large sofa, stylish center table, ambient lighting with floor and ceiling lamps, elegant curtains on window, neutral wall tones, indoor plants for decor, textured rug, decorative wall art, ample floor space, cozy and functional layout",
  "Kitchen": "modern kitchen with L-shaped cabinets along two adjacent walls, matte finish cabinetry, built-in stove, sink, clean backsplash tiles, efficient lighting, compact high-end appliances, mandatory spacious central floor for movement, clutter-free and ergonomic layout"}
  negative_prompt = """
    blurry, empty, deformed, bad anatomy, unrealistic lighting, unrealistic furniture,
    poor texture, low quality, cluttered, unsafe elements
    """
def calculate_metrics(orig_img, gen_img):
  orig_gray = np.array(orig_img.convert('L'))
  gen_gray = np.array(gen_img.convert('L'))
  ssim_value = ssim(orig_gray, gen_gray,
    data_range=255,
    win_size=11,
    gaussian_weights=True)
  loss_fn = lpips.LPIPS(net='vgg').to('cuda')
  tensor_orig = lpips.im2tensor(np.array(orig_img)/255.).to('cuda')
  tensor_gen = lpips.im2tensor(np.array(gen_img)/255.).to('cuda')
  lpips_value = loss_fn(tensor_orig, tensor_gen).item()
  return ssim_value, lpips_value
def generate_interior(input_image, room_type):
  if input_image is None or room_type not in prompt_dict:
    return None, "Missing input image or room type"
input_image = input_image.convert("RGB").resize((512, 512))
  depth_map = depth_estimator(input_image)["depth"].resize((512, 512))
  prompt = prompt_dict[room_type]
  generated_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=input_image,
    control_image=depth_map,
    num_inference_steps=35,
    guidance_scale=8.0,
    controlnet_conditioning_scale=0.85,
    generator=torch.Generator("cuda").manual_seed(42)
  ).images[0]
  ssim_score, lpips_score = calculate_metrics(input_image, generated_image)
  metrics = f"SSIM: {ssim_score:.3f} (target > 0.65), LPIPS: {lpips_score:.3f} (target < 0.35)"
  return generated_image, metrics
room_choices = ["Master Bedroom", "Children's Bedroom", "Hall", "Kitchen"]
  gr.Interface(
  fn=generate_interior,
  inputs=[
    gr.Image(type="pil", label="Upload Empty Room Image"),
    gr.Dropdown(choices=room_choices, label="Select Room Type") ],
  outputs=[
    gr.Image(label="Furnished Room Output"),
    gr.Text(label="Quality Metrics (SSIM & LPIPS)")
  ],
  title="🛋️ Interior Design Generator",
  description="Upload an empty room and choose its type to generate a professionally furnished version using Stable Diffusion."
  ).launch()
