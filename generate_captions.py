import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
from pathlib import Path
import json
import argparse
from tqdm import tqdm

def generate_captions(image_dir, output_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading BLIP model for captioning...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    print("BLIP model loaded.")

    image_paths = [p for p in Path(image_dir).glob("*.*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for image_path in tqdm(image_paths, desc="Generating captions"):
            try:
                raw_image = Image.open(image_path).convert('RGB')
                
                inputs = processor(raw_image, return_tensors="pt").to(device)
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                metadata_entry = {
                    "file_name": image_path.name,
                    "text": caption
                }
                f.write(json.dumps(metadata_entry) + '\n')

            except Exception as e:
                print(f"Could not process {image_path}: {e}")
                
    print(f"Successfully generated captions for {len(image_paths)} images.")
    print(f"Metadata saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="캡션을 생성할 이미지가 있는 폴더 경로")
    args = parser.parse_args()
    
    output_metadata_file = os.path.join(args.image_dir, "metadata.jsonl")
    
    generate_captions(args.image_dir, output_metadata_file)