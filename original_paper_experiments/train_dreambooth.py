import os
import yaml
import torch
import shutil
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datetime import datetime
from tqdm import tqdm

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(d, path):
    with open(path, "w") as f:
        yaml.dump(d, f, allow_unicode=True)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 샘플러 선택 함수
def get_scheduler(sched_type, model_name, subfolder="scheduler"):
    sched_type = str(sched_type).lower()
    if sched_type in ["ddim"]:
        from diffusers import DDIMScheduler
        return DDIMScheduler.from_pretrained(model_name, subfolder=subfolder)
    elif sched_type in ["dpm", "dpmsolver", "dpmsolvermultistep"]:
        from diffusers import DPMSolverMultistepScheduler
        return DPMSolverMultistepScheduler.from_pretrained(model_name, subfolder=subfolder)
    elif sched_type in ["euler"]:
        from diffusers import EulerDiscreteScheduler
        return EulerDiscreteScheduler.from_pretrained(model_name, subfolder=subfolder)
    elif sched_type in ["euler_a", "euler-ancestral"]:
        from diffusers import EulerAncestralDiscreteScheduler
        return EulerAncestralDiscreteScheduler.from_pretrained(model_name, subfolder=subfolder)
    elif sched_type in ["pndm"]:
        from diffusers import PNDMScheduler
        return PNDMScheduler.from_pretrained(model_name, subfolder=subfolder)
    else:
        # 기본값: DPM-Solver
        from diffusers import DPMSolverMultistepScheduler
        return DPMSolverMultistepScheduler.from_pretrained(model_name, subfolder=subfolder)

def save_checkpoint(save_dir, unet, text_encoder, optimizer, global_step):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"checkpoint_{global_step}.pt")
    torch.save({
        "unet": unet.state_dict(),
        "text_encoder": text_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
    }, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

def load_checkpoint(ckpt_path, unet, text_encoder, optimizer, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(checkpoint["unet"])
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    print(f"Resumed from checkpoint: {ckpt_path} (step {global_step})")
    return global_step

def image_transform(resolution, center_crop=False):
    from torchvision import transforms
    tfs = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    return transforms.Compose(tfs)

def ensure_class_images(class_data_dir, class_prompt, num_class_images, device="cuda"):
    os.makedirs(class_data_dir, exist_ok=True)
    images = [f for f in os.listdir(class_data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n_exist = len(images)
    n_needed = num_class_images - n_exist
    if n_needed > 0:
        print(f"Generating {n_needed} class images for {class_data_dir}...")
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
        for i in range(n_needed):
            image = pipe(class_prompt, num_inference_steps=30).images[0]
            image.save(os.path.join(class_data_dir, f"generated_{i+n_exist}.jpg"))
        del pipe
        torch.cuda.empty_cache()
    else:
        print(f"Enough class images exist ({n_exist}/{num_class_images}).")

class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, instance_prompt, tokenizer, resolution=512, center_crop=False,
                 class_data_root=None, class_prompt=None, num_class_images=0):
        self.instance_images = list(sorted([os.path.join(instance_data_root, fname)
                                           for fname in os.listdir(instance_data_root)
                                           if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]))
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.center_crop = center_crop
        self.image_transforms = image_transform(resolution, center_crop)
        self.class_images = []
        self.class_prompt = class_prompt
        if class_data_root and os.path.exists(class_data_root):
            self.class_images = list(sorted([os.path.join(class_data_root, fname)
                                             for fname in os.listdir(class_data_root)
                                             if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]))
            if num_class_images > 0:
                self.class_images = self.class_images[:num_class_images]
        self.has_prior = len(self.class_images) > 0 and class_prompt is not None

    def __len__(self):
        if self.has_prior:
            return max(len(self.instance_images), len(self.class_images))
        return len(self.instance_images)

    def __getitem__(self, idx):
        inst_img_path = self.instance_images[idx % len(self.instance_images)]
        inst_image = Image.open(inst_img_path).convert("RGB")
        inst_tensor = self.image_transforms(inst_image)
        inst_prompt_ids = self.tokenizer(self.instance_prompt, truncation=True, padding="max_length", max_length=77, return_tensors="pt").input_ids[0]

        example = {"instance_images": inst_tensor, "instance_prompt_ids": inst_prompt_ids}

        if self.has_prior:
            class_img_path = self.class_images[idx % len(self.class_images)]
            class_image = Image.open(class_img_path).convert("RGB")
            class_tensor = self.image_transforms(class_image)
            class_prompt_ids = self.tokenizer(self.class_prompt, truncation=True, padding="max_length", max_length=77, return_tensors="pt").input_ids[0]
            example["class_images"] = class_tensor
            example["class_prompt_ids"] = class_prompt_ids

        return example

def run_validation(
    unet, text_encoder, vae, tokenizer, prompt, device, dtype,
    num_images, save_dir, step, scheduler_type="dpm", model_name="runwayml/stable-diffusion-v1-5",
    num_inference_steps=30, guidance_scale=7.5
):
    from diffusers import StableDiffusionPipeline
    os.makedirs(save_dir, exist_ok=True)
    scheduler = get_scheduler(scheduler_type, model_name)
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=None, feature_extractor=None
    ).to(device, dtype)
    pipe.set_progress_bar_config(disable=True)
    for i in range(num_images):
        with torch.no_grad():
            image = pipe(prompt, num_inference_steps=num_inference_steps,guidance_scale=guidance_scale).images[0]
        image.save(os.path.join(save_dir, f"val_{step}_{i}.png"))
    del pipe
    torch.cuda.empty_cache()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_base", type=str, default="configs/base.yaml", help="Base config YAML path")
    parser.add_argument("--experiment_name", type=str, default=None, help="실험명/버전명/태그")
    parser.add_argument("--experiment_desc", type=str, default="", help="실험 상세설명")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config_base)
    nowstr = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or base_cfg.get("experiment_name") or f"exp_{nowstr}"
    experiment_desc = args.experiment_desc or base_cfg.get("experiment_desc", "")
    output_dir = os.path.join(base_cfg.get("output_root", "outputs"), experiment_name)

    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.config_base, os.path.join(output_dir, "base.yaml"))
    meta = {
        "experiment_name": experiment_name,
        "experiment_desc": experiment_desc,
        "datetime": nowstr,
        "config_base": args.config_base,
    }
    save_yaml(meta, os.path.join(output_dir, "meta.yaml"))

    # config
    guidance_scale = float(base_cfg.get("guidance_scale", 7.5))
    resolution = int(base_cfg.get("resolution", 512))
    train_batch_size = int(base_cfg.get("train_batch_size", 2))
    max_train_steps = int(base_cfg.get("max_train_steps", 1000))
    learning_rate = float(base_cfg.get("learning_rate", 5e-6))
    seed = int(base_cfg.get("seed", 42))
    bf16 = bool(base_cfg.get("bf16", False))
    save_ckpt_steps = int(base_cfg.get("save_ckpt_steps", 200))
    resume_from_checkpoint = base_cfg.get("resume_from_checkpoint", "")
    instance_data_dir = base_cfg.get("instance_data_dir")
    class_data_dir = base_cfg.get("class_data_dir")
    instance_prompt = base_cfg.get("instance_prompt")
    class_prompt = base_cfg.get("class_prompt")
    num_class_images = int(base_cfg.get("num_class_images", 0))
    with_prior_preservation = bool(base_cfg.get("with_prior_preservation", False))
    center_crop = bool(base_cfg.get("center_crop", False))
    prior_loss_weight = float(base_cfg.get("prior_loss_weight", 1.0))
    validation_steps = int(base_cfg.get("validation_steps", 500))
    num_validation_images = int(base_cfg.get("num_validation_images", 4))
    validation_prompt = base_cfg.get("validation_prompt", instance_prompt)
    validation_dir = os.path.join(output_dir, "validation")
    validation_scheduler = base_cfg.get("validation_scheduler", "dpm")
    train_text_encoder = bool(base_cfg.get("train_text_encoder", False))
    validation_num_inference_steps = int(base_cfg.get("validation_num_inference_steps", 30))

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if bf16 else torch.float32

    if with_prior_preservation and class_data_dir and class_prompt and num_class_images > 0:
        ensure_class_images(class_data_dir, class_prompt, num_class_images, device=device)

    print(f"실험명: {experiment_name}")
    print("결과 저장경로:", output_dir)
    print("Loading model...")

    model_name = "runwayml/stable-diffusion-v1-5"
    from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
    from transformers import CLIPTextModel, CLIPTokenizer

    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device, dtype)
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device, dtype)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device, dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    if train_text_encoder:
        text_encoder.train()
        params = list(unet.parameters()) + list(text_encoder.parameters())
    else:
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad_(False)
        params = list(unet.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        betas=(0.9, 0.999)
    )

    global_step = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        global_step = load_checkpoint(resume_from_checkpoint, unet, text_encoder, optimizer, device)
    else:
        print("Training from scratch (not resuming checkpoint).")

    train_dataset = DreamBoothDataset(
        instance_data_root=instance_data_dir,
        instance_prompt=instance_prompt,
        tokenizer=tokenizer,
        resolution=resolution,
        center_crop=center_crop,
        class_data_root=class_data_dir if with_prior_preservation else None,
        class_prompt=class_prompt,
        num_class_images=num_class_images
    )
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

    use_bf16 = bf16
    if use_bf16:
        scaler = None
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        scaler = torch.cuda.amp.GradScaler()
        autocast_ctx = torch.cuda.amp.autocast()

    print("Start training...")
    step = global_step
    data_iter = iter(train_dataloader)
    pbar = tqdm(total=max_train_steps, initial=step, desc="Steps")
    while step < max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        optimizer.zero_grad()
        with autocast_ctx:
            pixel_values = batch["instance_images"].to(device, dtype)
            prompt_ids = batch["instance_prompt_ids"].to(device)
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(prompt_ids)[0]
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            target = noise
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

            if with_prior_preservation and "class_images" in batch:
                class_pixel_values = batch["class_images"].to(device, dtype)
                class_prompt_ids = batch["class_prompt_ids"].to(device)
                class_latents = vae.encode(class_pixel_values).latent_dist.sample() * vae.config.scaling_factor
                class_noise = torch.randn_like(class_latents)
                class_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (class_latents.shape[0],), device=device).long()
                noisy_class_latents = noise_scheduler.add_noise(class_latents, class_noise, class_timesteps)
                class_encoder_hidden_states = text_encoder(class_prompt_ids)[0]
                class_pred = unet(noisy_class_latents, class_timesteps, class_encoder_hidden_states).sample
                prior_loss = torch.nn.functional.mse_loss(class_pred.float(), class_noise.float(), reduction="mean")
                loss = loss + prior_loss_weight * prior_loss

        if use_bf16:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": loss.item()})

        if step % save_ckpt_steps == 0:
            save_checkpoint(output_dir, unet, text_encoder, optimizer, step)
            print(f"[{step}] Loss: {loss.item():.4f}")

        if validation_steps > 0 and step % validation_steps == 0:
            print(f"Running validation at step {step} ...")
            run_validation(
                unet, text_encoder, vae, tokenizer,
                validation_prompt, device, dtype,
                num_validation_images, validation_dir, step,
                scheduler_type=validation_scheduler,
                num_inference_steps=validation_num_inference_steps,
                guidance_scale=guidance_scale
            )
            print("Validation images saved.")

    save_checkpoint(output_dir, unet, text_encoder, optimizer, step)
    print("Training completed.")

if __name__ == "__main__":
    main()
