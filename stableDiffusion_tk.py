import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from diffusers import AutoencoderKL
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

def stableDiffusion_function(prompt):
    #輸入噪聲及文字
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prompt = [str(prompt)]

    height = 512
    width = 512
    batch_size = 1 #要生成幾張圖片
    generator = torch.manual_seed(32) #設定隨機種子

    latents = torch.randn((batch_size, 4, height//8, width//8), generator=generator, )
    latents = latents.to(device)

    #文字處理
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float32)
    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float32)
    text_encoder = text_encoder.to(device)
    text_input = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length,truncation=True, return_tensors = 'pt') #文字轉token

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0] #token轉embedding
        

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        ['']*batch_size, padding='max_length', max_length=max_length, return_tensors='pt'
    )

    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    #UNet model for generating the latents
    unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32,subfolder='unet')

    #使用LMSDiscreteScheduler
    scheduler = LMSDiscreteScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32,
                                                    subfolder='scheduler')

    #把text_embedding和噪聲當輸入，放進U-Net模型中
    # 超參數設定
    num_inference_steps = 50
    guidance_scale = 7.5

    unet = unet.to(device)
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma

    scheduler.timesteps = scheduler.timesteps.type(torch.float32)
    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents]*2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    #使用vae把latent representation 還原成圖片
    vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32, subfolder='vae')
    import matplotlib.pyplot as plt

    vae = vae.to(device)

    latents = 1/0.18215 * latents #scale and decode the image latents with vae

    with torch.no_grad():
        image = vae.decode(latents).sample
        
    image = (image/2 + 0.5).clamp(0,1)
    image = image.detach().cpu().permute(0,2,3,1).numpy()
    images = (image*255).round().astype('uint8')
    pil_image = Image.fromarray(images[0])
    pil_image = ImageTk.PhotoImage(pil_image)
    
    return pil_image


def generate_image():
    prompt = entry.get()
    generated_image = stableDiffusion_function(prompt)
    result_label.config(image=generated_image)  
    result_label.image = generated_image 

#**
#*界面
#**
font = ("Arial", 20, "bold")

win = tk.Tk()
win.title("==")
win.geometry("800x900")

fm = tk.Frame(win, bg='gray', width=800, height=300)
fm.place(x=0, y=0)

lb1 = tk.Label(text="Please enter prompt:", font=font)
lb1.place(x=270, y=30)

entry = tk.Entry(font=font)
entry.place(x=270, y=100)

btn = tk.Button(text="Generate ==", command=generate_image, width=15, height=3, font=font)
btn.place(x=290, y=200)

result_label = tk.Label(font=font)
result_label.place(x=150,y=350)

win.mainloop()