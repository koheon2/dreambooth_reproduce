사용법
1. 환경 준비
```
# miniconda 환경 예시
conda create -n dreambooth python=3.10
conda activate dreambooth
pip install -r requirements.txt
```
2. 폴더 구조
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

먼저 학습용 config 작성 (예시: configs/base.yaml)
```
python -m scripts.train_dreambooth --config_base yaml파일경로
ex) python -m scripts.train_dreambooth --config_base configs/base.yaml

outputs/폴더/ 경로에 체크포인트, validation 이미지가 자동 생성. 폴더 이름은 yaml에서 지정되는 experiment_name으로 설정됨
```

4. 이미지 생성 (inference)
 
먼저 inference용 config 작성 (예시: configs/infer.yaml)
```
python -m scripts.inference --config_infer yaml파일경로
ex) python -m scripts.inference --config_infer configs/infer.yaml

outputs/사용한 체크포인트가 있는 폴더/inference/ 경로로 이미지 저장
```
5. 코드 주요 구조

config 파일로 모든 실험 파라미터 관리 (실험 자동화, reproducibility)

data/instance/prompts_and_classes.txt에 연구자들이 사용했던 프롬프트가 저장되어 있음. 
이 프롬프트는 imagen 기준이라서 stable diffusion의 clip에 적합한 프롬프트로 해야 성능이 더 잘 나올것 같아요.

train:

validation 기능은 훈련이 잘 되고 있는지 훈련 중간에 이미지 생성해서 체크하는 기능.

inference:

원하는 checkpoint로, 여러 prompt/seed/batch/샘플러 실험 지원

실험별 폴더/이미지명 자동 관리

기타 등등...


## 📁 Dataset Information

This implementation uses the official dataset provided by the authors of the DreamBooth paper:

**Repository:** https://github.com/google/dreambooth-dataset  
**Paper:** [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation (CVPR 2023)](https://arxiv.org/abs/2208.12242)

The dataset includes 30 subjects across 15 different classes, as used in the paper.

Some images were captured by the authors, and others were sourced from [Unsplash](https://unsplash.com).  
Attribution and license information is provided in:
