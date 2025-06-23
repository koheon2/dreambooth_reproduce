import torch
import clip
import timm
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re
import pandas as pd

def load_models(device):
    print("Loading CLIP and DINO models...")
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    dino_model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, num_classes=0).to(device)
    dino_model.eval()
    dino_preprocess = timm.data.create_transform(**timm.data.resolve_data_config(dino_model.pretrained_cfg))
    print("Models loaded successfully.")
    return {
        "clip": {"model": clip_model, "preprocess": clip_preprocess},
        "dino": {"model": dino_model, "preprocess": dino_preprocess},
    }

@torch.no_grad()
def get_image_embedding(image_path, model_pack, model_name, device):
    model = model_pack[model_name]["model"]
    preprocess = model_pack[model_name]["preprocess"]
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    if model_name == "clip":
        embedding = model.encode_image(image_input)
    elif model_name == "dino":
        embedding = model(image_input)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return embedding.cpu().numpy()


def evaluate_experiments(outputs_dir, instance_data_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device)
    results = []

    experiment_folders = [d for d in Path(outputs_dir).iterdir() if d.is_dir()]
    
    for exp_folder in tqdm(experiment_folders, desc="Evaluating Experiments"):
        exp_name = exp_folder.name
        
        match = re.match(r"([a-zA-Z0-9]+)_(.+)", exp_name)
        if not match:
            continue
        subject_name, category_name = match.groups()
        
        source_images_dir = Path(instance_data_dir) / subject_name
        if not source_images_dir.exists():
            print(f"Warning: Source instance directory not found for {subject_name}. Skipping {exp_name}.")
            continue
        
        source_image_paths = list(source_images_dir.glob("*.*"))
        source_clip_embeds = np.vstack([get_image_embedding(p, models, "clip", device) for p in source_image_paths])
        source_dino_embeds = np.vstack([get_image_embedding(p, models, "dino", device) for p in source_image_paths])
        avg_source_clip_embed = get_image_embedding(source_image_paths[0], models, "clip", device)
        avg_source_dino_embed = get_image_embedding(source_image_paths[0], models, "dino", device) 

        validation_dir = exp_folder / "validation_images"
        if not validation_dir.exists():
            continue
            
        gen_image_paths = list(validation_dir.glob("*.png"))
        clip_similarities, dino_similarities = [], []
        
        for gen_path in gen_image_paths:
            step_match = re.search(r"step(\d+)_", gen_path.name)
            if not step_match:
                continue
            
            step = int(step_match.group(1))
            
            if 1100 <= step <= 1500:
                gen_clip_embed = get_image_embedding(gen_path, models, "clip", device)
                gen_dino_embed = get_image_embedding(gen_path, models, "dino", device)
                clip_sim = cosine_similarity(gen_clip_embed, avg_source_clip_embed)[0][0]
                dino_sim = cosine_similarity(gen_dino_embed, avg_source_dino_embed)[0][0]
                clip_similarities.append(clip_sim)
                dino_similarities.append(dino_sim)

        if not clip_similarities:
            continue
            
        results.append({
            "Experiment": exp_name,
            "Subject": subject_name,
            "Category": category_name, 
            "Avg CLIP Similarity": np.mean(clip_similarities),
            "Avg DINO Similarity": np.mean(dino_similarities),
            "Num Validated Images": len(clip_similarities)
        })

    return results


if __name__ == "__main__":
    OUTPUTS_ROOT_DIR = "outputs"
    INSTANCE_DATA_ROOT_DIR = "data/instance"
    
    if not Path(OUTPUTS_ROOT_DIR).exists() or not Path(INSTANCE_DATA_ROOT_DIR).exists():
        print(f"Error: Make sure '{OUTPUTS_ROOT_DIR}' and '{INSTANCE_DATA_ROOT_DIR}' directories exist.")
    else:
        evaluation_results = evaluate_experiments(OUTPUTS_ROOT_DIR, INSTANCE_DATA_ROOT_DIR)
        
        if not evaluation_results:
            print("\nNo experiments found or no valid validation images were found.")
            print("Please check your 'outputs' folder structure and image filenames.")
        else:
            df_detailed = pd.DataFrame(evaluation_results)
            df_detailed_display = df_detailed.copy()
            df_detailed_display["Avg CLIP Similarity"] = df_detailed_display["Avg CLIP Similarity"].map('{:.4f}'.format)
            df_detailed_display["Avg DINO Similarity"] = df_detailed_display["Avg DINO Similarity"].map('{:.4f}'.format)
            df_detailed_display = df_detailed_display.sort_values(by=["Subject", "Category"]).reset_index(drop=True)
            
            print("\n\n--- Detailed Results (Per Subject) ---")
            print(df_detailed_display.to_string())
            
            df_summary = df_detailed.groupby('Category')[['Avg CLIP Similarity', 'Avg DINO Similarity']].mean()
            df_summary['Num Subjects'] = df_detailed.groupby('Category').size()
            df_summary = df_summary.sort_index()

            df_summary["Avg CLIP Similarity"] = df_summary["Avg CLIP Similarity"].map('{:.4f}'.format)
            df_summary["Avg DINO Similarity"] = df_summary["Avg DINO Similarity"].map('{:.4f}'.format)

            print("\n\n--- Category Summary (Averaged Across All Subjects) ---")
            print(df_summary.to_string())

            df_detailed.to_csv("evaluation_detailed_results.csv", index=False, float_format='%.4f')
            df_summary.to_csv("evaluation_summary_results.csv", float_format='%.4f')
            print(f"\nDetailed results also saved to evaluation_detailed_results.csv")
            print(f"Summary results also saved to evaluation_summary_results.csv")