import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    UNet2DConditionModel, AutoencoderKL,
    StableDiffusionPipeline,
    EulerDiscreteScheduler, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, PNDMScheduler 
)
from PIL import Image

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_scheduler(name, model_name_or_path): 
    scheduler_config = {
        "ddpm": DDPMScheduler,
        "ddim": DDIMScheduler,
        "euler": EulerDiscreteScheduler,
        "dpm": DPMSolverMultistepScheduler,
        "pndm": PNDMScheduler,
    }
    scheduler_class = scheduler_config.get(name.lower())
    if scheduler_class:
        return scheduler_class.from_pretrained(model_name_or_path, subfolder="scheduler")
    else:
        raise ValueError(f"Unknown scheduler: {name}")


@torch.no_grad() 
def run_inference(
    pipeline, 
    prompt, num_images, save_dir,
    num_inference_steps, guidance_scale, current_seed, #
    image_prefix="infer"
):
    os.makedirs(save_dir, exist_ok=True)
    
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=pipeline.device).manual_seed(current_seed)
    
    for i in tqdm(range(num_images), desc=f"Generating for '{prompt[:30]}...'"):
        image = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        sane_prompt_prefix = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in prompt[:25]).rstrip()
        sane_prompt_prefix = sane_prompt_prefix.replace(' ', '_')
        filename = f"{image_prefix}_{sane_prompt_prefix}_seed{current_seed}_img{i}.png"
        image.save(os.path.join(save_dir, filename))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_infer", type=str, required=True, help="Path to the inference YAML configuration file.")
    args = parser.parse_args()

    infer_cfg = load_yaml(args.config_infer)
    
    experiment_name = infer_cfg["experiment_name"] 
    model_dir_from_cfg = infer_cfg["model_dir"] 
    sd_base_model_path = infer_cfg.get("sd_model", "runwayml/stable-diffusion-v1-5")
    
    output_parent_dir = infer_cfg.get("output_dir", os.path.join(model_dir_from_cfg, "inference_results"))
    
    checkpoint_step = infer_cfg.get("checkpoint_step")
    num_inference_images_per_prompt = int(infer_cfg.get("num_inference_images", 4))
    scheduler_type = infer_cfg.get("scheduler", "pndm")
    
    num_inference_steps_cfg = infer_cfg.get("inference_timestep", infer_cfg.get("num_inference_steps", 50))
    num_inference_steps = int(num_inference_steps_cfg)

    guidance_scale = float(infer_cfg.get("guidance_scale", 7.5))
    default_seed = int(infer_cfg.get("seed", 42)) 
    
    bf16 = bool(infer_cfg.get("bf16", False))
    device_str = infer_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    dtype = torch.bfloat16 if bf16 and device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    if bf16 and dtype == torch.float32:
        print("Warning: BF16 was requested but is not available on this device. Using FP32 instead.")

    # --- Find Checkpoint ---
    if not os.path.isdir(model_dir_from_cfg):
        raise RuntimeError(f"Model directory not found: {model_dir_from_cfg}")
    
    ckpts = [f for f in os.listdir(model_dir_from_cfg) if f.startswith("checkpoint_step") and f.endswith(".pt")]
    if not ckpts:
        raise RuntimeError(f"No checkpoint files (e.g., checkpoint_stepXXX.pt) found in: {model_dir_from_cfg}")


    if infer_cfg.get("ckpt_step") > 0:
        ckpt_step = infer_cfg.get("ckpt_step")
        ckpt_filename = f"checkpoint_step{ckpt_step}.pt"
        ckpt_path = os.path.join(model_dir_from_cfg, ckpt_filename)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Specified checkpoint not found: {ckpt_path}")
    else: # Find latest
        latest_step = -1
        latest_ckpt_file = ""
        for ckpt_file in ckpts:
            try:
                step_num = int(ckpt_file.replace("checkpoint_step", "").replace(".pt", ""))
                if step_num > latest_step:
                    latest_step = step_num
                    latest_ckpt_file = ckpt_file
            except ValueError:
                continue 
        if not latest_ckpt_file:
            raise RuntimeError(f"Could not determine the latest checkpoint in {model_dir_from_cfg}")
        ckpt_path = os.path.join(model_dir_from_cfg, latest_ckpt_file)
    
    print(f"Loading inference models using checkpoint: {ckpt_path}")

    # --- Load Models from Base SD Path ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(sd_base_model_path, subfolder="tokenizer", use_fast=False)
        text_encoder = CLIPTextModel.from_pretrained(sd_base_model_path, subfolder="text_encoder").to(device, dtype=dtype)
        vae = AutoencoderKL.from_pretrained(sd_base_model_path, subfolder="vae").to(device, dtype=dtype)
        unet = UNet2DConditionModel.from_pretrained(sd_base_model_path, subfolder="unet").to(device, dtype=dtype)
    except Exception as e:
        print(f"Error loading base models from {sd_base_model_path}: {e}")
        sys.exit(1)

    # --- Load Fine-tuned Weights from Checkpoint ---
    checkpoint_data = torch.load(ckpt_path, map_location=device)
    
    unet.load_state_dict(checkpoint_data["unet"])
    print("Successfully loaded UNet weights from checkpoint.")
    
    if "text_encoder" in checkpoint_data:
        text_encoder.load_state_dict(checkpoint_data["text_encoder"])
        print("Successfully loaded TextEncoder weights from checkpoint.")
    else:
        print("TextEncoder weights not found in checkpoint. Using base model's TextEncoder.")
    
    unet.eval()
    text_encoder.eval()
    vae.eval()
    torch.cuda.empty_cache()

    # --- Create Inference Pipeline ---
    scheduler_for_inference = get_scheduler(scheduler_type, sd_base_model_path)
    
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler_for_inference,
        safety_checker=None, 
        feature_extractor=None,
    )

    # --- Inference Loop for Each Subject/Prompt ---
    subjects_to_infer = infer_cfg.get("subjects", [])
    if not subjects_to_infer and "prompt" in infer_cfg: 
        subjects_to_infer.append({"name": "default", "prompt": infer_cfg["prompt"]})
    
    if not subjects_to_infer:
        print("No subjects or global prompt found in inference config. Nothing to generate.")
        sys.exit(0)

    for subj_config in subjects_to_infer:
        subj_name = subj_config.get("name", "unnamed_subject")
        prompt_text = subj_config.get("prompt")
        
        if not prompt_text:
            print(f"Skipping subject '{subj_name}' due to missing prompt.")
            continue
            
        subject_output_dir = os.path.join(output_parent_dir, experiment_name, subj_name)
        os.makedirs(subject_output_dir, exist_ok=True)
        d
        current_prompt_seed = int(subj_config.get("seed", default_seed))
        set_seed(current_prompt_seed) 

        print(f"\nGenerating images for Subject: '{subj_name}'")
        print(f"  Prompt: '{prompt_text}'")
        print(f"  Seed: {current_prompt_seed}, Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        print(f"  Saving to: {subject_output_dir}")

        run_inference(
            pipeline,
            prompt=prompt_text,
            num_images=num_inference_images_per_prompt,
            save_dir=subject_output_dir,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            current_seed=current_prompt_seed,
            image_prefix=f"{subj_name}"
        )
    
    del pipeline 
    torch.cuda.empty_cache()
    print("\nAll inference tasks complete.")

if __name__ == "__main__":
    main()