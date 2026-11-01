# 2026-10-25 과제 (#3) — CNN 학습 시각화 · NumPy vs. CuPy GPU 가속 · MNIST/CIFAR-10 GPU 학습 · IoU/NMS

6회차(2026-10-25) 수업 노트북 `03_DL-and-GPU.ipynb`와 7회차(2026-10-31) `04_DL-Object-Detection.ipynb`의 TODO 중
과제로 지정된 부분(강의 README 「과제」 3번)을 독립 실행 가능한 `.py` 파일 6개로 제출합니다.

대상 TODO (노트북 셀 순서):
| 노트북 절 | TODO 내용 | 제출 파일 |
|---|---|---|
| 03 F-16) 모델 성능 확인 | Epoch별 Loss / Accuracy 그래프 | `mnist_cnn_training_plots.py` |
| 03 F-17) 학습된 모델의 Feature Map | Conv1 8개 · Conv2 16개 · 6개 층별 대표 4개 (viridis) | `mnist_cnn_training_plots.py` |
| 03 G-4) 이미지 잔상 효과 (GPU 가속) | `motion_trail_np()` (CPU) · `motion_trail_cp()` (GPU), CPU↔GPU 복사, 시간 비교 | `motion_trail.py` |
| 03 G-5) Convolution 연산 (GPU 가속) | `convolution2d()` CPU 처리 · `convolution2d_cp()` (GPU) 변환-워밍업-처리 파이프라인, CPU/GPU 결과·시간 비교 | `convolution.py` |
| 03 I-3) GPU 기반 MNIST 학습 | `main_gpu()` 작성, `main_cpu()`와 학습 시간 비교 | `mnist_cpu_gpu_training.py` |
| 03 J-12~16) CIFAR-10 GPU 학습 | 5 epoch 학습·평가, epoch별/전체 시간 출력, 모델 `.pth` 저장·로드, 예측 40장 확인 | `cifar10_gpu_training.py` + `src/models/CIFAR10/CIFAR10_CNN.pth` |
| 04 B) IoU | `(x,y,w,h)→(x1,y1,x2,y2)` 변환, IoU 계산, 사과 이미지 3장 비교 | `iou_nms.py` |
| 04 C) NMS | Confidence Thresholding, NMS 직접 구현, `cv2.dnn.NMSBoxes()`와 비교 | `iou_nms.py` |

## 파일 구조

```
Assignments/2026-10-25/
├── mnist_cnn_training_plots.py   # (1/6) MNIST CNN 3 epoch CPU 학습 → Loss/Accuracy 그래프 → 층별 Feature Map
├── motion_trail.py               # (2/6) 잔상 효과 CPU(NumPy) + GPU(CuPy) 구현, CPU vs GPU 시간 비교표
├── convolution.py                # (3/6) 2D convolution CPU(NumPy) + GPU(CuPy) 구현, CPU vs GPU 결과·시간 비교표
├── mnist_cpu_gpu_training.py     # (4/6) MNIST CNN을 CPU·GPU(PyTorch CUDA)에서 각각 학습해 시간 비교
├── cifar10_gpu_training.py       # (5/6) CIFAR-10 CNN GPU 5 epoch 학습 → .pth 저장 → 다시 로드 → 예측 40장
├── iou_nms.py                    # (6/6) IoU 계산(사과 3장), Confidence Thresholding + NMS 직접 구현, OpenCV NMS와 비교
├── output/                       # 실행 시 결과 PNG 저장
└── README.md

src/models/CIFAR10/CIFAR10_CNN.pth  # cifar10_gpu_training.py가 저장하는 학습된 모델 (노트북과 같은 경로)
```

- **파일 간 연결**: `convolution.py`는 `motion_trail.py`의 `load_input_image()`를, `mnist_cpu_gpu_training.py`·`cifar10_gpu_training.py`는 `mnist_cnn_training_plots.py`의 모델·학습/평가 함수(`SimpleCNN`, `train_one_epoch`, `evaluate` 등)를 import합니다. 노트북에서도 "MNIST에서 정의한 함수를 재사용"하므로 같은 구조를 따랐고, 따라서 **6개 파일은 같은 폴더에 있어야** 합니다.
- **CPU·GPU를 한 파일에 둔 이유**(2·3·4번): 한 번 실행으로 CPU → GPU 순서로 처리하고 시간 비교표를 바로 출력하기 위해서입니다. 맥(Apple Silicon)에는 CUDA가 없어 `import cupy`가 실패하므로 2·3번은 `try/except ImportError`로 감싸 CPU 결과만 출력합니다. 4·5번(PyTorch)은 노트북과 같이 CUDA가 없으면 `RuntimeError`로 멈추되, 맥에서 코드를 확인할 수 있도록 `--gpu-device mps` / `--device mps` 옵션을 두었습니다.
- **입력 데이터**는 저장소 루트의 `src/`를 파일 위치 기준 상대 경로(`Path(__file__)` 두 단계 위)로 참조하므로 어느 위치에서 실행해도 됩니다.
  - `src/images/seagull.jpg` — 잔상·convolution 입력 (수업과 동일)
  - `src/images/two_apples1~3.png` — IoU 입력 (수업과 동일)
  - `src/datasets/MNIST/`, `src/datasets/cifar-10-batches-py/` — 없으면 `torchvision`이 자동 다운로드 (CIFAR-10은 약 170MB, `.gitignore`로 제외)

### 각 파일의 함수 흐름

**`mnist_cnn_training_plots.py`** — `main()`: 난수 고정 → `load_mnist_loaders()` → `SimpleCNN` 생성(CPU) → `train_model()`(3 epoch, `train_one_epoch()`/`evaluate()` 반복, 기록 수집) → `plot_loss_history()` → `plot_accuracy_history()` → `visualize_trained_feature_maps()`(테스트 이미지 1장 순전파 → `show_feature_maps()` viridis).
- `SimpleCNN`, `train_one_epoch()`, `evaluate()`, `show_feature_maps()`는 수업 노트북 코드를 그대로 옮긴 것이고, 그래프 2개와 Feature Map 시각화 함수가 과제 부분입니다.

**`motion_trail.py`** — `main()`: `load_input_image()`(흑백 → 1280x720 → float32 0~1) → `make_weights()`(1.0→0.05 선형 감소, 합 1 정규화) → `run_motion_trail_cpu()`(시간 측정) → CuPy 있으면 `run_motion_trail_gpu()`(`cp.asarray()` 업로드 → 64x64 워밍업 → `motion_trail_cp()` → `cp.asnumpy()` 다운로드, 단계별 시간) → `print_comparison()`(시간표·속도 비·최대 오차) → 이미지 출력.
- `motion_trail_np()`: 출력 픽셀 (y, x)마다 shift = 0..31에 대해 `weights[shift] * image[y, x - shift]`를 더합니다. `x - shift < 0`이면 왼쪽 경계 밖이라 더할 값이 없고, shift는 커지기만 하므로 `break`.
- `motion_trail_cp()`: shift 반복은 Python이 하되, 각 shift에서 `output[:, shift:] += image[:, :W - shift] * weights[shift]` 한 줄을 GPU가 병렬 처리. 픽셀 반복문이 없습니다.

**`convolution.py`** — `main()`: 이미지 로드 → `run_edge_convolution_cpu()`(세로/가로 edge 필터 2개를 `convolution2d()`로 적용, 시간 측정, ReLU) → CuPy 있으면 `run_edge_convolution_gpu()`(변환 → 워밍업 → 처리 → 복사, 단계별 시간) → `np.allclose()`로 결과 일치 확인 → `print_comparison()`(시간표·속도 비) → 이미지 출력.
- `convolution2d_cp()`: 커널 원소 9개를 하나씩 고르고, 그 원소와 곱해질 입력 픽셀 전체를 `padded[ky : ky + stride*(H-1)+1 : stride, kx : ...]` slicing으로 한 번에 뽑아 `output += input_region * kernel[ky, kx]`로 누적합니다. 출력 픽셀(921,600개) 반복이 커널 원소(9개) 반복으로 바뀝니다.

**`mnist_cpu_gpu_training.py`** — `main()`: `main_cpu()` → `main_gpu()` → 시간 차·속도 비 출력(노트북 I절 비교 셀과 동일한 형식).
- `main_cpu()`/`main_gpu()`는 모두 `train_on_device(device, epochs)`를 호출합니다. 장치 외 조건(시드·데이터·모델·Adam)을 완전히 같게 두려고 학습 루프를 한 함수로 모았고, `main_gpu()`는 거기에 "CUDA 사용 가능 확인(없으면 `RuntimeError`)"만 더한 얇은 함수입니다.
- `synchronize(device)`: GPU 연산은 비동기로 쌓이기만 하므로, epoch 시간을 재기 전에 `torch.cuda.synchronize()`로 GPU 작업이 끝나길 기다립니다(CuPy의 `cp.cuda.Stream.null.synchronize()`에 대응). CPU면 아무것도 하지 않습니다.

**`cifar10_gpu_training.py`** — `main()`: 난수 고정 → `resolve_device()`(CUDA 확인) → `load_cifar10_loaders()`(학습셋만 RandomCrop·HorizontalFlip 증강, 정규화) → `CIFAR10CNN` 생성·파라미터 수 출력 → `warm_up()`(64장 0 입력으로 순전파 3회) → `train_model()`(AdamW, 5 epoch, epoch별/전체 시간) → `save_checkpoint()`(`.pth`) → `load_checkpoint()`(새 모델에 로드, 메타데이터 출력) → `plot_predictions()`(테스트 40~79번 이미지 4x10 격자, 틀리면 빨간 제목).
- `ConvBlock`(Conv→BN→ReLU→Conv→BN→ReLU→MaxPool)과 `CIFAR10CNN`(블록 3개 → 전역 평균 풀링 → Dropout → Linear)은 수업 노트북 구조 그대로입니다.
- `.pth`에는 가중치 외에 optimizer 상태·마지막 테스트 정확도·epoch·클래스 이름·정규화 평균/표준편차를 함께 담습니다(노트북 J-14 형식). 로드할 때 `weights_only=True`로 pickle 임의 코드 실행을 막고, `map_location="cpu"`로 저장 장치와 달라도 읽히게 했습니다.

**`iou_nms.py`** — `main()`: `run_iou_demo()`(사과 3장) → `run_nms_demo()`(난수 상자 20개) → 창 표시.
- `convert_bounding_box_to_corners()`: `cv2.boundingRect()`의 `(x, y, w, h)` → `(x1, y1, x2, y2)`. 노트북 TODO의 `convert_bbox_coord`.
- `calculate_iou(a, b)`: 두 상자 모두 모서리 좌표. 교집합 상자 = (max(x1), max(y1), min(x2), min(y2)), 겹치지 않으면 너비·높이가 음수가 되므로 `max(0, …)`로 0 처리. 합집합 = A + B − 교집합, 0이면 IoU 0.
- `prune_low_confidence()`: confidence ≥ 0.5인 상자만 남김(노트북 `prune_low_conf`).
- `apply_nms()`: `np.argsort(scores)[::-1]`로 confidence 내림차순 인덱스 → 맨 앞을 keep → 나머지와 IoU 계산 → IoU < 0.2인 것만 다음 후보로 → 반복. `cv2.dnn.NMSBoxes()` 결과와 집합 비교해 일치 여부를 출력합니다.
- 난수 상자는 `random.Random(42)` 전용 생성기로 만들어 실행마다 같은 결과가 나옵니다(노트북은 매번 달라져 비교가 어려움).

## 실행 방식

```bash
cd Assignments/2026-10-25

# 어디서나 (CPU)
python mnist_cnn_training_plots.py   # 3 epoch 학습 후 그래프·Feature Map 창 5개
python iou_nms.py                    # IoU 3장 + NMS 전/후/OpenCV 비교 창 2개

# Jetson (CUDA). 먼저 전력 모드를 최대로:
sudo nvpmodel -m 2 && sudo jetson_clocks        # MAXN SUPER (기본 25W 모드는 약 2배 느림)
python motion_trail.py               # CPU → GPU(CuPy) 순서로 실행 + 시간 비교표
python convolution.py                # CPU → GPU(CuPy) + 결과·시간 비교표
python mnist_cpu_gpu_training.py     # CPU 3 epoch(약 10분) → GPU 3 epoch → 속도 비
python cifar10_gpu_training.py       # GPU 5 epoch → src/models/CIFAR10/CIFAR10_CNN.pth 저장 → 예측 40장

# 맥에서 코드 확인용 (실측 비교 목적 아님)
python mnist_cpu_gpu_training.py --gpu-device mps --epochs 1
python cifar10_gpu_training.py --device mps --epochs 1
```

인자 없이 실행되며, 결과는 matplotlib 창으로 표시되고 `output/` 폴더에 PNG로도 저장됩니다.

## 의존성

| 패키지 | Jetson (JetPack 6.2, 수업 환경) | 맥 검증 환경 |
|---|---|---|
| Python | 3.10 | 3.10.20 |
| NumPy | 1.21.5 | 1.21.5 |
| OpenCV | 4.7 | 4.7.0 |
| Matplotlib | 3.7 | 3.7.5 |
| PyTorch / torchvision | 2.8.0 / 0.23.0 (CUDA 12.6, `pypi.jetson-ai-lab.io/jp6/cu126` 휠, 수업 F절) | 2.8.0 / 0.23.0 (MPS) |
| CuPy | `cupy-cuda12x==12.3.0` + `fastrlock==0.8.3` (수업 G절 설치 방법, `--no-deps`) | 설치 불가 (CUDA 없음) |
| scikit-image (선택) | 수업 환경에 설치됨 (NMS 배경 허블 사진) | 없음 → 검은 배경으로 대체 |

- CuPy 설치(Jetson, 수업 노트북 G절과 동일):
  ```bash
  python -m pip download --no-deps --only-binary=:all: "cupy-cuda12x==12.3.0" -d ~/Downloads
  python -m pip install --no-deps "fastrlock==0.8.3"
  python -m pip install --no-deps --only-binary=:all: ~/Downloads/cupy_cuda12x-12.3.0-*.whl
  ```
  `--no-deps`를 쓰는 이유: CuPy가 요구하는 NumPy 최신 버전이 Jetson의 NumPy 1.21.5·OpenCV 4.7과 충돌하므로 의존성 자동 설치를 막습니다.

## 실행 결과

### 맥 (CPU / MPS)

- `mnist_cnn_training_plots.py` (3 epoch, CPU, 총 24.4초):

  | Epoch | Train loss | Train acc | Test loss | Test acc | 시간 |
  |---|---|---|---|---|---|
  | 1 | 0.3133 | 90.52% | 0.0953 | 96.99% | 9.5s |
  | 2 | 0.0925 | 97.14% | 0.0690 | 97.73% | 7.5s |
  | 3 | 0.0666 | 97.92% | 0.0493 | 98.21% | 7.4s |

  결과 PNG: `output/cnn_loss.png`, `output/cnn_accuracy.png`, `output/feature_maps_conv1.png`, `output/feature_maps_conv2.png`, `output/feature_maps_by_layer.png`.
  층별 Feature Map에서 conv1 Channel 2처럼 값이 전부 음수인 채널은 ReLU를 지나면 완전히 0(relu1/pool1 Channel 2가 빈 화면)이 되는 것을 확인할 수 있습니다.

- `motion_trail.py`: 1280x720, 잔상 길이 32 → CPU **6.44초** (CuPy 없음 → GPU 생략 메시지 출력). 출력 왼쪽 가장자리가 어두운 띠로 나오는 것은 shift보다 왼쪽에 있는 픽셀은 더할 원본이 없어 밝기가 줄기 때문입니다(의도된 결과).
- `convolution.py`: 1280x720, 3x3 필터 2개 → CPU **5.48초** (CuPy 없음 → GPU 생략 메시지 출력).
- `iou_nms.py` (`output/iou_apples.png`, `output/nms_result.png`):

  | 이미지 | IoU | 상황 |
  |---|---|---|
  | two_apples1 | 0.0000 | 두 사과가 떨어져 있어 교집합 0 |
  | two_apples2 | 0.3046 | 일부 겹침 |
  | two_apples3 | 0.4934 | 많이 겹침 |

  난수 상자 20개 → confidence ≥ 0.5: 17개 → NMS(IoU < 0.2): **4개**. `cv2.dnn.NMSBoxes()`도 같은 4개를 남겨 **결과 일치(True)**.
- `mnist_cpu_gpu_training.py --gpu-device mps --epochs 1`: CPU 7.89s vs MPS 7.55s (1.04x). 맥 MPS는 작은 모델에서 이득이 거의 없고, 실측 비교는 Jetson CUDA가 기준입니다.
- `cifar10_gpu_training.py --device mps --epochs 1`: 파라미터 1,148,874개, 1 epoch 46.3초, 테스트 정확도 64.49%, `.pth` 13.2MB 저장·로드 확인, 예측 40장 중 26장 정답(코드 경로 확인용 — 제출 `.pth`는 Jetson 5 epoch 결과).

### Jetson Orin Nano Super (CUDA, MAXN SUPER)

- `mnist_cpu_gpu_training.py`: (실행 후 기입)
- `cifar10_gpu_training.py`:

  - 모델: `CIFAR10CNN` (총 파라미터 1,148,874개)
  - 학습: CUDA, 5 epoch, 총 578.14초

    | Epoch | Train loss | Train accuracy | Test loss | Test accuracy | 시간 |
    |---|---:|---:|---:|---:|---:|
    | 1 | 1.2864 | 53.15% | 0.9697 | 65.61% | 118.04s |
    | 2 | 0.8881 | 68.69% | 0.8497 | 71.38% | 115.55s |
    | 3 | 0.7133 | 75.45% | 0.6729 | 77.64% | 113.67s |
    | 4 | 0.6225 | 78.67% | 0.7103 | 76.18% | 115.58s |
    | 5 | 0.5532 | 81.32% | 0.5861 | 80.65% | 115.30s |

  - 마지막 테스트 정확도: **80.65%**
  - 모델 저장: `src/models/CIFAR10/CIFAR10_CNN.pth` (13.2MB)
  - 예측 결과: `output/cifar10_predictions.png` — 40장 중 **31장 정답**
- `motion_trail.py` / `convolution.py` GPU 비교표: (실행 후 기입)

## 기타 참고 사항

### 수업 중 작성한 초안의 오류 수정

- **`motion_trail_np()` 초안 3가지**(6회차 노트북):
  1. `output = input_np.copy()`로 시작 → 함수 밖 전역 변수를 쓰는 데다, shift=0 항(원본 × weights[0])이 이미 들어 있는 원본 위에 또 더해져 이중 누적. → `np.zeros_like(image)`에서 0부터 누적.
  2. `min(255, max(0, …))` 클램프 → 입력은 이미 0~1 float32이고 가중치 합이 1이라 결과가 0~1을 벗어날 수 없음. 255 기준 클램프는 아무 효과가 없어 삭제.
  3. `range(TRAIL_LENGTH)` 상수 반복 → 노트북 워밍업 셀은 길이 4짜리 가중치를 넘기므로 `weights[4]`에서 IndexError. → `range(len(weights))`.
- **IoU 초안**(7회차 노트북 `Labs/2026-10-31/04_DL-Object-Detection.ipynb`): `calculate_iou()` 안에서 `convert_bbox_coord()`를 다시 호출했는데, 호출하는 셀도 이미 변환한 좌표를 넘겨 **두 번 변환**됐습니다. `(x1, y1, x2, y2)`를 `(x, y, w, h)`로 착각해 상자가 실제보다 커져 IoU가 부풀려졌습니다(수업 중 값 0.4438 / 0.6324 → 수정 후 0.3046 / 0.4934). 제출본은 `calculate_iou()`가 모서리 좌표만 받고 변환은 호출부에서 한 번만 합니다. NMS 안에서도 상자가 `(x1, y1, x2, y2)`이므로 같은 수정이 적용됩니다.
- **NMS 호출 셀**(노트북 제공 코드): `prune_low_conf()` 결과를 `new_boxes`에 받고도 `apply_nms(boxes, scores, …)`에 원본을 넘겨 confidence thresholding이 버려졌고, 결과 그림도 `scores[i]`(원본 점수)를 썼습니다. 제출본은 걸러진 상자를 NMS에 넣고 남은 상자의 점수를 그립니다.

### CuPy 코드 검증 방법 (맥에서는 GPU 미실행)
- 맥에는 CUDA가 없어 `motion_trail.py`, `convolution.py`의 GPU 경로는 맥에서 실행하지 못했습니다. 대신 `cupy` 모듈 자리에 NumPy를 끼워 넣는(`sys.modules["cupy"] = numpy 흉내 모듈`) 검증 스크립트로 **로직만** 확인했습니다. 37x53 난수 이미지로 `motion_trail_cp()` vs `motion_trail_np()`(가중치 길이 32·4), `convolution2d_cp()` vs `convolution2d()`(3x3·5x5 커널, stride 1·2·3, padding 0·1·2)를 비교해 모두 `np.allclose` 통과(최대 오차 2e-6 이하, float32 누적 순서 차이).
- 실측은 위 「Jetson」 절 참고.

### 설계 메모
- `convolution2d_cp()`의 slicing 끝 인덱스를 `ky + stride*(H-1) + 1`로 쓴 이유: 출력 크기 H개를 stride 간격으로 정확히 뽑기 위함. stride=1이면 `ky : ky + H`와 같습니다.
- 워밍업을 작은 입력으로 먼저 하는 이유(CuPy 64x64, CIFAR-10 64장 0 입력 3회): 첫 CUDA 호출에 섞이는 Context 생성·cuDNN 초기화·메모리 할당 시간을 측정에서 빼되, 본 처리와 같은 코드 경로를 한 번 지나가게 하기 위함. MNIST CPU/GPU 비교(`mnist_cpu_gpu_training.py`)는 노트북 I절에 워밍업이 없어 그대로 두었고, 그래서 GPU 1 epoch째 시간에 초기화가 포함됩니다.
- CIFAR-10 학습 batch를 32로 줄인 이유: MNIST(1x28x28)보다 입력이 크고(3x32x32) 모델 채널도 많아(64~256) 같은 batch면 GPU 메모리를 더 쓰기 때문(노트북 J-7).
- `.pth`는 `.gitignore`의 `*.pth` 규칙에 걸리므로 `!src/models/CIFAR10/CIFAR10_CNN.pth` 예외를 추가해 이 파일만 커밋했습니다. CIFAR-10 데이터셋(170MB)은 제외.
- `iou_nms.py`의 함수 이름은 노트북 TODO 이름(`convert_bbox_coord`, `prune_low_conf`)의 줄임말을 풀어 `convert_bounding_box_to_corners`, `prune_low_confidence`로 썼습니다(저장소 네이밍 규칙). 역할은 동일합니다.

### AI 활용 vs 본인 기여
- 본인: 과제 범위(노트북 TODO 선별) 결정, CPU/GPU를 한 파일에 두고 시간 비교표를 한 번에 출력하는 구조 결정, 수업 중 초안 작성(`motion_trail_np()`, `main_gpu()`, IoU/NMS 4개 함수 — `Labs/2026-10-25`, `Labs/2026-10-31` 노트북) 및 오류 수정 방향 확인, Jetson 실행·실측·결과 확인.
- AI 보조: 노트북 셀·초안을 함수·`main()` 구조의 `.py`로 옮기는 작업, CuPy 함수(`motion_trail_cp`, `convolution2d_cp`)의 slicing 식 작성, cupy→numpy 치환 검증 스크립트, `--device` 옵션·체크포인트 저장/로드 정리, docstring·주석·README 정리.
- 수업 중 작성한 노트북 원본은 `Labs/2026-10-25/`, `Labs/2026-10-31/`에 있습니다.
