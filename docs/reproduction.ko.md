# 재현 및 라이브 데모 실행 안내

[English](./reproduction.md) | [README](../README.ko.md)

## 1. 로컬 인터랙티브 웹 데모 및 CLI 실행

```powershell
# 의존성 설치 (Python 3.10 이상 권장)
pip install -r requirements.txt

# Google Drive 공개 모델 가중치 다운로드
python verify_author.py download --model ml    # 21.63 MB (CPU)
python verify_author.py download --model dl    # 860.27 MB (GPU 권장, CPU 가능)

# 로컬 아티팩트 상태 확인
python verify_author.py status

# MIT Red 인터랙티브 웹 데모 실행 (http://127.0.0.1:8000)
.\launch-nlu.bat demo
# Linux / macOS:
sh ./launch-nlu.sh demo

# 명령줄(CLI) 단일 글 쌍 판별 (ml, dl, both 선택)
python verify_author.py predict --model both --check-symmetry --text-file-1 a.txt --text-file-2 b.txt

# 단위·API·대칭성·Locked Test 정합성 테스트 실행
pytest -v tests/test_inference_and_demo.py
```

## 2. Google Colab 전체 학습 재현

세 트랙 모두 `notebooks/colab/` 및 Google Drive에 단일 실행 통합 노트북으로 공개되어 있습니다.

1. [Google Drive 공개 폴더](https://drive.google.com/drive/folders/1ij9ZwlLtimS26bAE9B_D5zL3raAKJody)에서 [benchmark_1024.zip (61.2 MB)](https://drive.google.com/file/d/1Y9xfMRskEUZPYw9f5jdiM6ws9L-Cwdr8/view)을 준비합니다.
2. 아래 노트북을 Colab에서 열고 `benchmark_1024.zip`을 업로드한 뒤 위에서 아래로 한 번 실행합니다.
   - **Track 01 (CPU 8+ vCPU, 32 GB RAM)**: [train_01_stylometric_ml.ipynb](https://colab.research.google.com/drive/1jAuZKPe7mcuL9jB9SAIQmGr-jQ-CdjtJ) → `model_01_stylometric_ml.joblib`, `results_01_stylometric_ml.zip` 생성
   - **Track 02 (L4 24GB / A100)**: [train_02_deberta_v3_base.ipynb](https://colab.research.google.com/drive/1U6Fv97IcC061XdJlVYWM79D9YSOsD9Bk) → `results_02_deberta_v3_base.zip` 생성
   - **Track 03 (L4 24GB / A100 / H100)**: [train_03_deberta_v3_large.ipynb](https://colab.research.google.com/drive/1ROS0vdfbpZupVoM38Dz41eDB1tCAGjWL) → `model_03_deberta_v3_large.pt`, `results_03_deberta_v3_large.zip` 생성 (`results_01`/`02` ZIP을 함께 올리면 3대 모델 McNemar 통계 비교까지 자동 실행)
