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
│   ├── base.yaml        # 학습 config
│   └── infer.yaml       # 추론 config
├── data/
│   ├── instance/       # 훈련할 이미지 폴더들
│   │    ├── dog/       
│   │    └── mug/
│   └── class/          # prior preservation loss 위한 자동 생성 이미지 저장 경로
│        ├── dog/
│        └── mug/
├── scripts/
│   ├── train_dreambooth.py
│   └── inference.py
└── outputs/
    └── (실험별 폴더 자동 생성)
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