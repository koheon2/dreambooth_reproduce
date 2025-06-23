import os
import yaml
import torch
import timm
import lpips
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL
)
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPTextModel,
    CLIPTokenizer
)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_scheduler(sched_type, model_name, subfolder="scheduler"):
    sched_type = str(sched_type).lower()
    if sched_type == "ddim":
        from diffusers import DDIMScheduler
        return DDIMScheduler.from_pretrained(model_name, subfolder=subfolder)
    elif sched_type in ["dpm","dpmsolver","dpmsolvermultistep"]:
        from diffusers import DPMSolverMultistepScheduler
        return DPMSolverMultistepScheduler.from_pretrained(model_name, subfolder=subfolder)
    elif sched_type == "euler":
        from diffusers import EulerDiscreteScheduler
        return EulerDiscreteScheduler.from_pretrained(model_name, subfolder=subfolder)
    elif sched_type in ["euler_a","euler-ancestral"]:
        from diffusers import EulerAncestralDiscreteScheduler
        return EulerAncestralDiscreteScheduler.from_pretrained(model_name, subfolder=subfolder)
    elif sched_type == "pndm":
        from diffusers import PNDMScheduler
        return PNDMScheduler.from_pretrained(model_name, subfolder=subfolder)
    else:
        from diffusers import DPMSolverMultistepScheduler
        return DPMSolverMultistepScheduler.from_pretrained(model_name, subfolder=subfolder)

def preprocess_dino():
    return transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406),
                             std=(0.229,0.224,0.225)),
    ])

def preprocess_lpips():
    return transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),
                             std=(0.5,0.5,0.5)),
    ])

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Inference + fidelity & ablation metrics (DINO, CLIP, PRES, DIV)"
    )
    parser.add_argument("--config_infer",    type=str, required=True,
                        help="inference 설정 yaml 파일 경로")
    parser.add_argument("--real_images_dir", type=str, required=True,
                        help="subject의 실제(레퍼런스) 이미지 폴더")
    parser.add_argument("--class_prompts",   nargs="+", default=[],
                        help="PRES 계산용 class-only 프롬프트들 (예: 'a dog5')")
    parser.add_argument("--output_dir",      type=str, default="",
                        help="결과 저장 폴더 (기본: 모델 폴더 하위 inference/)")
    args = parser.parse_args()

    infer_cfg = load_yaml(args.config_infer)
    ckpt_step             = int(infer_cfg.get("ckpt_step", -1))
    experiment_name       = infer_cfg["experiment_name"]
    output_root           = infer_cfg.get("output_root", "outputs")
    prompts               = infer_cfg["prompts"]
    num_images_per_prompt = int(infer_cfg.get("num_images_per_prompt", 4))
    seed                  = int(infer_cfg.get("seed", 42))
    dtype                 = infer_cfg.get("dtype", "fp32")
    guidance_scale        = float(infer_cfg.get("guidance_scale", 7.5))
    num_inference_steps   = int(infer_cfg.get("num_inference_steps", 30))
    scheduler_type        = infer_cfg.get("scheduler", "dpm")

    # 2) 출력 디렉토리
    model_dir = os.path.join(output_root, experiment_name)
    out_dir = args.output_dir or os.path.join(model_dir, "inference_backpack_rc")
    os.makedirs(out_dir, exist_ok=True)

    # 3) 디바이스 & dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    dtype_torch = dtype_map.get(dtype, torch.float32)

    # 4) 체크포인트 선택
    ckpts = [f for f in os.listdir(model_dir)
             if f.startswith("checkpoint_") and f.endswith(".pt")]
    if not ckpts:
        raise RuntimeError(f"모델 폴더에 checkpoint 파일이 없습니다: {model_dir}")
    if ckpt_step > 0:
        ckpt_path = os.path.join(model_dir, f"checkpoint_{ckpt_step}.pt")
    else:
        ckpt_path = os.path.join(
            model_dir,
            max(ckpts, key=lambda x: int(x.split("_")[1].split(".")[0]))
        )
    print(f" 체크포인트 로드: {ckpt_path}")

    # 5) Stable Diffusion Pipeline 초기화
    model_name  = "runwayml/stable-diffusion-v1-5"
    unet        = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")\
                    .to(device, dtype_torch)
    text_encoder= CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")\
                    .to(device, dtype_torch)
    tokenizer   = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    vae         = AutoencoderKL.from_pretrained(model_name, subfolder="vae")\
                    .to(device, dtype_torch)
    scheduler   = get_scheduler(scheduler_type, model_name)

    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt["unet"])
    text_encoder.load_state_dict(ckpt["text_encoder"])

    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, unet=unet,
        scheduler=scheduler,
        safety_checker=None, feature_extractor=None
    ).to(device, dtype_torch)
    pipe.set_progress_bar_config(disable=True)

    # 6) DINO, CLIP, LPIPS 모델 준비
    print(" DINO, CLIP, LPIPS 로드 중...")
    dino_model   = timm.create_model("vit_base_patch16_224_dino", pretrained=True)\
                    .to(device).eval()
    dino_tf      = preprocess_dino()

    clip_model   = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")\
                    .to(device).eval()
    clip_proc    = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    lpips_fn     = lpips.LPIPS(net='alex').to(device).eval()
    lpips_tf     = preprocess_lpips()

    # 7) Reference real 이미지 임베딩 계산
    real_dino_embeds = []
    real_clip_embeds = []
    for fn in sorted(os.listdir(args.real_images_dir)):
        if not fn.lower().endswith((".jpg","jpeg","png")):
            continue
        img = Image.open(os.path.join(args.real_images_dir, fn))\
                   .convert("RGB")
        # DINO
        x = dino_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            real_dino_embeds.append(dino_model(x).squeeze(0).cpu())

        # CLIP
        inputs = clip_proc(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb_i = clip_model.get_image_features(**inputs)
        emb_i = emb_i / emb_i.norm(dim=-1, keepdim=True)
        real_clip_embeds.append(emb_i.squeeze(0).cpu())

    # 8) 생성 & metric 계산
    metrics = []
    generator = torch.Generator(device=device).manual_seed(seed)

    # (A) subject prompts → DINO, CLIP-I, CLIP-T, 나중에 DIV
    subject_lpips = {}
    subject_dino  = {}

    for prompt in prompts:
        subject_lpips[prompt] = []
        subject_dino[prompt]  = []

        desc = f"[Subject] {prompt[:30]}..."
        pbar = tqdm(range(num_images_per_prompt), desc=desc)
        for i in pbar:
            # --- 생성
            with torch.no_grad():
                img = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images[0]

            fname = f"subj_{prompt.replace(' ','_')}_{i+1:02d}.png"
            img.save(os.path.join(out_dir, fname))

            # DINO metric
            x = dino_tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                gen_d = dino_model(x).squeeze(0).cpu()
            sims = [F.cosine_similarity(gen_d, r, dim=0).item()
                    for r in real_dino_embeds]
            dino_score = sum(sims)/len(sims)
            subject_dino[prompt].append(dino_score)

            # CLIP-I
            ci_in = clip_proc(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                ci_feat = clip_model.get_image_features(**ci_in)
            ci_feat = (ci_feat/ci_feat.norm(dim=-1,keepdim=True)).squeeze(0).cpu()
            clip_i_sims = [
                F.cosine_similarity(ci_feat, ref, dim=0).item()
                for ref in real_clip_embeds
            ]
            clip_i_score = sum(clip_i_sims) / len(clip_i_sims)

            # CLIP-T
            ct_in = clip_proc(text=[prompt], return_tensors="pt").to(device)
            with torch.no_grad():
                txt_feat = clip_model.get_text_features(**ct_in)
            txt_feat = (txt_feat/txt_feat.norm(dim=-1,keepdim=True)).squeeze(0).cpu()
            clip_t_score = F.cosine_similarity(ci_feat, txt_feat, dim=0).item()

            # LPIPS tensor 저장
            lp = lpips_tf(img).unsqueeze(0).to(device)
            subject_lpips[prompt].append(lp)

            # 저장 (div는 나중에 채움)
            metrics.append({
                "filename": fname,
                "prompt":   prompt,
                "type":     "subject",
                "dino":     dino_score,
                "clip_i":   clip_i_score,
                "clip_t":   clip_t_score,
                "pres":     None,
                "div":      None
            })
            pbar.set_postfix(dino=f"{dino_score:.3f}",
                             clip_i=f"{clip_i_score:.3f}",
                             clip_t=f"{clip_t_score:.3f}")

    # (B) class prompts → PRES (=DINO on class images)
    for prompt in args.class_prompts:
        desc = f"[Class] {prompt[:30]}..."
        pbar = tqdm(range(num_images_per_prompt), desc=desc)
        for i in pbar:
            with torch.no_grad():
                img = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images[0]

            fname = f"class_{prompt.replace(' ','_')}_{i+1:02d}.png"
            img.save(os.path.join(out_dir, fname))

            # PRES 계산 (DINO)
            x = dino_tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                gen_d = dino_model(x).squeeze(0).cpu()
            pres_score = sum(
                F.cosine_similarity(gen_d, r, dim=0).item()
                for r in real_dino_embeds
            ) / len(real_dino_embeds)

            metrics.append({
                "filename": fname,
                "prompt":   prompt,
                "type":     "class",
                "dino":     None,
                "clip_i":   None,
                "clip_t":   None,
                "pres":     pres_score,
                "div":      None
            })
            pbar.set_postfix(pres=f"{pres_score:.3f}")

    # 9) DIV 계산 (subject group만)
    for prompt, lp_tensors in subject_lpips.items():
        # lp_tensors: prompt별로 저장해 둔 (1,3,256,256) 텐서 리스트
        distances = []
        N = len(lp_tensors)
        for i in range(N):
            for j in range(i+1, N):
                # (선택) self-self 체크
                if i == j:
                    d0 = lpips_fn(lp_tensors[i], lp_tensors[j]).item()
                    assert d0 < 1e-3, f"Self-LPIPS 불일치: {d0}"

                # LPIPS 거리 계산
                distances.append(
                    lpips_fn(lp_tensors[i], lp_tensors[j]).item()
                )

        # 쌍의 개수 = N*(N-1)/2 로 나누기
        div_score = sum(distances) / len(distances) if distances else 0.0

        # metrics 리스트에 채워넣기
        for row in metrics:
            if row["type"] == "subject" and row["prompt"] == prompt:
                row["div"] = div_score

    # 10) CSV 저장
    df = pd.DataFrame(metrics)
    csv_path = os.path.join(out_dir, "metrics_all_backpack_rc.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n 모든 metric 저장: {csv_path}")

if __name__ == "__main__":
    main()
