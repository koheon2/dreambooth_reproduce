사용법
1. environment settings
   our reproduce is conducted with miniconda
```
conda create -n dreambooth python=3.10
conda activate dreambooth
pip install -r requirements.txt
```
2. file structure
```
dreambooth_project/
├── configs/
│   ├── base.yaml        # train config
│   ├── infer.yaml       # inference config
│   ├── base_cap.yaml       # captioned train config
│   └── infer_cap.yaml       # captioned trained model inference config
├── data/
│   ├── instance/       # instance images
│   │    ├── dog/       
│   │    └── mug/
│   └── class/          # auto generated path for prior preservation loss image  
│        ├── dog/
│        └── mug/
├── original_paper_experiments/    #codes for reproducting original dreambooth experiments
├── our_own_experiments/    #codes for our own experiments
├── outputs/
│   └── (auto generated paths)
├── generate_captions.py #inference with image captioned prompt model
├── inference.py        #inference 
├── train_captioned.py  #train with image captioned prompt
└── train.py            #train 

```
3. train

1.configs setting (configs/base.yaml)
```
python train.py --config_base path/to/config.yaml
ex) python train.py --config_base configs/base.yaml
```
Checkpoints and validation images will be automatically saved under the outputs/ directory.
The output folder name is determined by the experiment_name specified in the YAML config.

4. train with prompting(our improvement code)

1.configs setting (configs/base_cap.yaml)
2.make metadata.jsonl (ex: data/instance/teapot/metadata.jsonl)
```
{"file_name": "00.jpg", "text": "a photo of a sks teapot and a rose on a plate"}
{"file_name": "01.jpg", "text": "a photo of is a sks teapot and a cup on a wooden table"}
{"file_name": "02.jpg", "text": "a photo of is a sks teapot and a cup on a table"}
{"file_name": "03.jpg", "text": "a photo of is a sks teapot and a cup on a table"}
{"file_name": "04.jpg", "text": "a photo of is a sks teapot and a cup on a table"}
```
it can be generated with BLIP captioning
```
python generate_captions.py --image_dir path/to/subject_dir
ex) python generate_captions.py --image_dir data/instance/teapot
```
캡셔닝 generate 후에 metadata.jsonl 파일에 identifier(ex: sks)를 추가하고, CLIP 인코더에 적합하게 a photo of 로 시작하도록 수정
```
{"file_name": "01.jpg", "text": "there is a tea pot and a cup on a wooden table"}
=>
{"file_name": "01.jpg", "text": "a photo of is a sks teapot and a cup on a wooden table"}
```
이후 다음 코드로 train 한다.
```
python train_captioned.py --config_base path/to/configs.yaml
ex) python train_captioned.py --config_base configs/base_cap.yaml
```

6. inference
 
1.configs setting (configs/infer.yaml)
```
python inference.py --config_base path/to/config.yaml
ex) python inference.py --config_base configs/infer.yaml
```

## 📁 Dataset Information

This implementation uses the official dataset provided by the authors of the DreamBooth paper:

**Repository:** https://github.com/google/dreambooth-dataset  
**Paper:** [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation (CVPR 2023)](https://arxiv.org/abs/2208.12242)

The dataset includes 30 subjects across 15 different classes, as used in the paper.

Some images were captured by the authors, and others were sourced from [Unsplash](https://unsplash.com).  
Attribution and license information is provided in:
