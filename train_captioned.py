import os
import sys
import yaml
import torch
import json
import random
import argparse 
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from diffusers import (
    UNet2DConditionModel, AutoencoderKL,
    DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, PNDMScheduler,
    StableDiffusionPipeline
)
from transformers import AutoTokenizer, CLIPTextModel
import torch.nn.functional as F 

# --- Helper Functions ---
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

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Dataset Class  ---
class DreamBoothMultiSubjectDataset(Dataset):
    def __init__(self, subjects_config, tokenizer, size, center_crop, tokenizer_max_length, with_prior_preservation, use_augmentation=False):
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.tokenizer_max_length = tokenizer_max_length
        self.with_prior_preservation = with_prior_preservation
        self.use_augmentation = use_augmentation
        self.subject_names = [s["name"] for s in subjects_config]
        
        self.instance_data = []
        self.class_images_paths = []
        self.class_prompts = []

        for subject in subjects_config:
            instance_data_dir = Path(subject["instance_data_dir"])
            metadata_path = instance_data_dir / "metadata.jsonl"
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"'{metadata_path}' not found. "
                    f"Please run generate_captions.py for the directory '{instance_data_dir}' first."
                )

            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    image_path = instance_data_dir / entry['file_name']
                    if image_path.exists():
                        self.instance_data.append({"path": str(image_path), "caption": entry['text']})

            if self.with_prior_preservation and "class_data_dir" in subject:
                class_data_dir = Path(subject["class_data_dir"])
                if class_data_dir.is_dir():
                    subject_class_paths = sorted([p for p in class_data_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
                    self.class_images_paths.extend(subject_class_paths)
                    self.class_prompts.extend([subject["class_prompt"]] * len(subject_class_paths))

        if not self.instance_data:
            raise ValueError("No instance images with valid metadata found.")

        self.num_instance_images = len(self.instance_data)
        self.num_class_images = len(self.class_images_paths)
        self._length = self.num_instance_images

        transforms_list = [transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)]
        if center_crop:
            transforms_list.append(transforms.CenterCrop(size))
        else:
            transforms_list.append(transforms.RandomCrop(size))
        if self.use_augmentation:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transforms_list.extend([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.image_transforms = transforms.Compose(transforms_list)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        instance_entry = self.instance_data[index % self.num_instance_images]
        instance_image_path = instance_entry["path"]
        instance_caption = instance_entry["caption"]
        
        instance_image = Image.open(instance_image_path).convert("RGB")
        instance_image = ImageOps.exif_transpose(instance_image)
        
        example["instance_pixel_values"] = self.image_transforms(instance_image)
        
        example["instance_prompt_ids"] = self.tokenizer(
            instance_caption, truncation=True, padding="max_length",
            max_length=self.tokenizer_max_length, return_tensors="pt"
        ).input_ids[0]

        # Class 이미지 로직은 기존과 동일
        if self.with_prior_preservation and self.num_class_images > 0:
            class_idx = index % self.num_class_images
            class_image_path = self.class_images_paths[class_idx]
            class_prompt = self.class_prompts[class_idx]
            class_image = Image.open(class_image_path).convert("RGB")
            class_image = ImageOps.exif_transpose(class_image)
            example["class_pixel_values"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt, truncation=True, padding="max_length",
                max_length=self.tokenizer_max_length, return_tensors="pt"
            ).input_ids[0]
        
        return example

# --- Collate Function  ---
def collate_fn_dreambooth(examples, with_prior_preservation):
    instance_pixel_values = torch.stack([e["instance_pixel_values"] for e in examples])
    instance_prompt_ids = torch.stack([e["instance_prompt_ids"] for e in examples])

    batch = {
        "instance_pixel_values": instance_pixel_values,
        "instance_prompt_ids": instance_prompt_ids,
    }

    if with_prior_preservation and "class_pixel_values" in examples[0]: 
        class_pixel_values = torch.stack([e["class_pixel_values"] for e in examples])
        class_prompt_ids = torch.stack([e["class_prompt_ids"] for e in examples])

        batch["pixel_values"] = torch.cat([instance_pixel_values, class_pixel_values], dim=0)
        batch["prompt_ids"] = torch.cat([instance_prompt_ids, class_prompt_ids], dim=0)
    else:

        batch["pixel_values"] = instance_pixel_values
        batch["prompt_ids"] = instance_prompt_ids

    return batch

# --- Class Image Generation ---
@torch.no_grad()
def ensure_class_images(subject_config, sd_model_name, scheduler_type, device, dtype, seed, inference_steps):
    cls_dir = subject_config.get("class_data_dir")
    if not cls_dir:
        print(f"Skipping class image generation for {subject_config['name']}: no class_data_dir defined.")
        return

    os.makedirs(cls_dir, exist_ok=True)
    
    n_needed = subject_config.get("num_class_images", 0)
    if n_needed == 0:
        return

    cls_imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))]
    
    if len(cls_imgs) >= n_needed:
        print(f"{subject_config['name']}: Enough class images exist ({len(cls_imgs)}/{n_needed})")
        return
    
    num_to_generate = n_needed - len(cls_imgs)
    print(f"{subject_config['name']}: Generating {num_to_generate} class images for prompt: '{subject_config['class_prompt']}'...")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_name,
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)
        pipe.scheduler = get_scheduler(scheduler_type, sd_model_name)
    except Exception as e:
        print(f"Error loading StableDiffusionPipeline for class image generation: {e}")
        return

    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    for i in tqdm(range(num_to_generate), desc=f"Generating class images for {subject_config['name']}"):
        img = pipe(
            subject_config["class_prompt"], 
            num_inference_steps=inference_steps, 
            generator=generator
        ).images[0]
        img.save(os.path.join(cls_dir, f"class_gen_{len(cls_imgs) + i:04d}.jpg"))
    
    del pipe
    torch.cuda.empty_cache()

# --- Validation Function ---
@torch.no_grad()
def run_validation(unet, text_encoder, vae, tokenizer, prompt, device, dtype,
                  num_images, save_dir, step, scheduler_type, model_name_or_path,
                  num_inference_steps, guidance_scale, val_seed):
    os.makedirs(save_dir, exist_ok=True)
    
    unet.eval()
    text_encoder.eval()
    vae.eval()

    pipeline = StableDiffusionPipeline(
        vae=vae, 
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet, 
        scheduler=get_scheduler(scheduler_type, model_name_or_path),
        safety_checker=None,
        feature_extractor=None,
    )
    pipeline = pipeline.to(device, dtype=dtype) 

    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(val_seed)
    
    print(f"Running validation with prompt: '{prompt}'")
    for i in range(num_images):
        image = pipeline(
            prompt, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale, 
            generator=generator
        ).images[0]
        image.save(os.path.join(save_dir, f"val_step{step}_img{i}_seed{val_seed}.png"))
    
    del pipeline
    torch.cuda.empty_cache()


# --- Main Training Function ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_base", type=str, required=True, help="configs/ti_base.yaml")
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base Stable Diffusion model name or path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu).")
    args = parser.parse_args()

    with open(args.config_base, "r") as f:
        base_cfg = yaml.safe_load(f)

    cfg_seed = int(base_cfg.get("seed", 42))
    seed_everything(cfg_seed)

    use_augmentation_cfg = bool(base_cfg.get("use_augmentation", False))
    experiment_name = base_cfg["experiment_name"]
    output_root = base_cfg["output_root"]
    output_dir = os.path.join(output_root, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    bf16_enabled_cfg = base_cfg.get("bf16", False)
    dtype = torch.bfloat16 if bf16_enabled_cfg and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    if bf16_enabled_cfg and dtype == torch.float32:
        print("Warning: BF16 was requested but is not available or supported. Using FP32 instead.")

    with_prior_preservation_cfg = base_cfg.get("with_prior_preservation", False) 
    prior_loss_weight = float(base_cfg.get("prior_loss_weight", 1.0))

    subjects_cfg_list = base_cfg["subjects"]
    for subject_conf_item in subjects_cfg_list:
        if with_prior_preservation_cfg and subject_conf_item.get("num_class_images", 0) > 0:
            ensure_class_images(
                subject_conf_item, 
                sd_model_name=args.sd_model,
                scheduler_type=base_cfg.get("class_image_scheduler", "pndm"),
                device=args.device,
                dtype=dtype, 
                seed=cfg_seed,
                inference_steps=int(base_cfg.get("class_image_steps", 50))
            )

    print(f"Loading models from {args.sd_model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer", use_fast=False)
        text_encoder = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder").to(args.device, dtype=dtype)
        vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae").to(args.device, dtype=dtype)
        unet = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet").to(args.device, dtype=dtype)
    except Exception as e:
        print(f"Error loading base models: {e}")
        sys.exit(1)

    train_text_encoder_flag_cfg = base_cfg.get("train_text_encoder", False)
    if not train_text_encoder_flag_cfg:
        text_encoder.requires_grad_(False)
        print("Text encoder weights frozen.")
    
    vae.requires_grad_(False)
    print("VAE weights frozen.")
    torch.cuda.empty_cache()

    print("Initializing dataset...")
    train_dataset = DreamBoothMultiSubjectDataset(
        subjects_cfg_list,
        tokenizer=tokenizer,
        size=int(base_cfg["resolution"]),
        center_crop=bool(base_cfg.get("center_crop", False)),
        tokenizer_max_length=tokenizer.model_max_length,
        with_prior_preservation=with_prior_preservation_cfg,
        use_augmentation=use_augmentation_cfg 
    )
    if len(train_dataset) == 0: 
        print("Error: No instance images found in dataset. Please check your data directories and configuration.")
        sys.exit(1)
    
    # Use the new collate function
    collate_function_to_use = lambda examples: collate_fn_dreambooth(examples, with_prior_preservation_cfg)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=int(base_cfg["train_batch_size"]),                                                
        shuffle=True, 
        collate_fn=collate_function_to_use, 
        drop_last=True, 
        num_workers=base_cfg.get("num_dataloader_workers", 0)
    )

    params_to_optimize = list(unet.parameters())
    if train_text_encoder_flag_cfg:
        print("Including text encoder parameters in optimizer.")
        params_to_optimize.extend(text_encoder.parameters())
    
    optimizer = torch.optim.AdamW(
        params_to_optimize, 
        lr=float(base_cfg["learning_rate"]),
        betas=(base_cfg.get("adam_beta1", 0.9), base_cfg.get("adam_beta2", 0.999)),
        weight_decay=base_cfg.get("adam_weight_decay", 1e-2),
        eps=base_cfg.get("adam_epsilon", 1e-8)
    )
    
    autocast_enabled_runtime = (dtype == torch.bfloat16) 

    noise_scheduler_train = DDPMScheduler.from_pretrained(args.sd_model, subfolder="scheduler")

    max_train_steps_cfg = int(base_cfg["max_train_steps"])
    save_ckpt_steps_cfg = int(base_cfg.get("save_ckpt_steps", 500)) 
    
    step0 = 0
    resume_ckpt_path_cfg = base_cfg.get("resume_from_checkpoint", "")

    if resume_ckpt_path_cfg and os.path.isfile(resume_ckpt_path_cfg):
        print(f"Resuming training from checkpoint: {resume_ckpt_path_cfg}")
        ckpt = torch.load(resume_ckpt_path_cfg, map_location=args.device)
        unet.load_state_dict(ckpt.get("unet", ckpt)) 
        if "text_encoder" in ckpt and train_text_encoder_flag_cfg:
             try:
                text_encoder.load_state_dict(ckpt["text_encoder"])
                print("Loaded text_encoder state from checkpoint.")
             except RuntimeError as e: 
                print(f"Could not load text_encoder state from checkpoint: {e}. Using base TE weights.")

        if "optimizer" in ckpt and base_cfg.get("resume_optimizer", False):
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("Loaded optimizer state from checkpoint.")
            except:
                print("Could not load optimizer state from checkpoint. Initializing new optimizer state.")
        step0 = ckpt.get("step", 0)
        print(f"Resumed from step {step0}. Will continue up to {max_train_steps_cfg}.")


    print("Starting training...")
    unet.train()
    if train_text_encoder_flag_cfg:
        text_encoder.train()

    progress_bar = tqdm(range(step0, max_train_steps_cfg + 1), initial=step0, total=max_train_steps_cfg, desc="Steps")
    data_iter = iter(train_dataloader)

    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        optimizer.zero_grad(set_to_none=True)
        
        pixel_values = batch["pixel_values"].to(args.device, dtype=dtype)
        prompt_ids = batch["prompt_ids"].to(args.device)
        
        current_batch_size = pixel_values.shape[0]

        with torch.autocast(device_type=args.device.split(":")[0], dtype=dtype, enabled=autocast_enabled_runtime):
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler_train.config.num_train_timesteps, (current_batch_size,), device=latents.device).long()
            noisy_latents = noise_scheduler_train.add_noise(latents, noise, timesteps)

            if train_text_encoder_flag_cfg:
                encoder_hidden_states = text_encoder(prompt_ids)[0]
            else:
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(prompt_ids)[0]
            
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

            if noise_scheduler_train.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler_train.config.prediction_type == "v_prediction":
                target = noise_scheduler_train.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler_train.config.prediction_type}")

            if with_prior_preservation_cfg and current_batch_size > base_cfg["train_batch_size"]:

                chunk_size = base_cfg["train_batch_size"] 
                
                model_pred_instance, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target_instance, target_prior = torch.chunk(target, 2, dim=0)

                if model_pred_instance.shape[0] != chunk_size or model_pred_prior.shape[0] != chunk_size :
                     print(f"Warning: Chunk size mismatch. Batch size: {current_batch_size}, Expected chunk: {chunk_size}")

                     instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                     prior_loss_val = 0.0
                     loss = instance_loss
                else:
                    instance_loss = F.mse_loss(model_pred_instance.float(), target_instance.float(), reduction="mean")
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                    loss = instance_loss + prior_loss_weight * prior_loss
                    prior_loss_val = prior_loss.item()
                
                instance_loss_val = instance_loss.item()

            else: 
                instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = instance_loss
                instance_loss_val = instance_loss.item()
                prior_loss_val = 0.0
        
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item(), inst_loss=instance_loss_val, prior_loss=prior_loss_val)

        if step != 0 and (step % save_ckpt_steps_cfg == 0 or step == max_train_steps_cfg):
            ckpt_payload = {
                "unet": unet.state_dict(),
                "step": step 
            }
            if train_text_encoder_flag_cfg:
                ckpt_payload["text_encoder"] = text_encoder.state_dict()
            ckpt_payload["optimizer"] = optimizer.state_dict() # Optional
            
            save_path = os.path.join(output_dir, f"checkpoint_step{step}.pt")
            torch.save(ckpt_payload, save_path)
            print(f"\nCheckpoint saved: {save_path}")

        val_period_cfg = int(base_cfg.get("validation_steps", 500))
        if step != 0 and (step % val_period_cfg == 0 or step == max_train_steps_cfg):
            print(f"\nRunning validation at step {step}...")
            
            validation_prompt_cfg = base_cfg.get("validation_prompt", f"A photo of {train_dataset.subject_names[0] if train_dataset.subject_names else 'sks'}")

            run_validation(
                unet=unet, text_encoder=text_encoder, vae=vae, tokenizer=tokenizer,
                prompt=validation_prompt_cfg, 
                device=args.device, dtype=dtype,
                num_images=int(base_cfg.get("num_validation_images", 4)),
                save_dir=os.path.join(output_dir, "validation_images"),
                step=step,
                scheduler_type=base_cfg.get("validation_scheduler", "pndm"),
                model_name_or_path=args.sd_model,
                num_inference_steps=int(base_cfg.get("validation_inference_steps", 30)),
                guidance_scale=float(base_cfg.get("validation_guidance_scale", 7.5)),
                val_seed = cfg_seed 
            )
            unet.train()
            if train_text_encoder_flag_cfg:
                text_encoder.train()

    print("\nTraining complete.")

if __name__ == "__main__":
    main()