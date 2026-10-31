# 2026-10-25 과제 — CNN 학습 결과 시각화, NumPy vs. CuPy GPU 가속 (잔상 효과·Convolution)

6회차(2026-10-25) 수업 노트북 `03_DL-and-GPU.ipynb`의 TODO 중 수업 시간에 작성하지 못한 부분을 독립 실행 가능한 `.py` 파일로 제출합니다.

대상 TODO (노트북 셀 순서):
| 노트북 절 | TODO 내용 | 제출 파일 |
|---|---|---|
| F-16) 모델 성능 확인 | Epoch별 Loss / Accuracy 그래프 | `mnist_cnn_training_plots.py` |
| F-17) 학습된 모델의 Feature Map | Conv1 8개 · Conv2 16개 · 6개 층별 대표 4개 (viridis) | `mnist_cnn_training_plots.py` |
| G-4) 이미지 잔상 효과 | `motion_trail_np()` (CPU) | `motion_trail_numpy.py` |
| G-4) 이미지 잔상 효과 | `motion_trail_cp()` (GPU), CPU↔GPU 복사, 시간 비교 | `motion_trail_cupy.py` |
| G-5) Convolution 연산 | `convolution2d()`로 CPU 처리·시간 측정 | `convolution_numpy.py` |
| G-5) Convolution 연산 | `convolution2d_cp()` (GPU), 변환-워밍업-처리 파이프라인, CPU/GPU 결과 비교 | `convolution_cupy.py` |

## 파일 구조

```
Assignments/2026-10-25/
├── mnist_cnn_training_plots.py   # MNIST CNN 3 epoch CPU 학습 → Loss/Accuracy 그래프 → 층별 Feature Map
├── motion_trail_numpy.py         # 잔상 효과 CPU(NumPy) 구현 + 이미지 로드·가중치 (다른 파일이 import)
├── motion_trail_cupy.py          # 잔상 효과 GPU(CuPy) 구현 + CPU vs GPU 시간 비교   [Jetson 전용]
├── convolution_numpy.py          # 2D convolution CPU(NumPy) + edge 필터 (다른 파일이 import)
├── convolution_cupy.py           # 2D convolution GPU(CuPy) + CPU vs GPU 결과·시간 비교 [Jetson 전용]
├── output/                       # 실행 시 결과 PNG 저장 (맥에서 실행한 결과 포함)
└── README.md
```

- **NumPy 파일과 CuPy 파일을 분리**한 이유: 맥(Apple Silicon)에는 CUDA가 없어 `import cupy` 자체가 실패합니다. CPU 버전은 어디서나 돌고, GPU 버전은 Jetson에서만 돌도록 나눴습니다.
- **파일 간 연결**: `*_cupy.py`는 같은 폴더의 `*_numpy.py`에서 CPU 함수·필터·이미지 로드 함수를 `import`해 CPU vs GPU를 비교합니다. `convolution_numpy.py`도 `motion_trail_numpy.py`의 `load_input_image()`를 가져와 같은 1280x720 흑백 이미지를 씁니다. 따라서 **5개 파일은 같은 폴더에 있어야** 합니다.
- **입력 데이터**는 저장소 루트의 `src/`를 파일 위치 기준 상대 경로(`Path(__file__)` 두 단계 위)로 참조하므로 어느 위치에서 실행해도 됩니다.
  - `src/images/seagull.jpg` — 잔상·convolution 입력 (수업과 동일)
  - `src/datasets/MNIST/` — 없으면 `torchvision`이 자동 다운로드

### 각 파일의 함수 흐름

**`mnist_cnn_training_plots.py`** — `main()`: 난수 고정 → `load_mnist_loaders()` → `SimpleCNN` 생성(CPU) → `train_model()`(3 epoch, `train_one_epoch()`/`evaluate()` 반복, 기록 수집) → `plot_loss_history()` → `plot_accuracy_history()` → `visualize_trained_feature_maps()`(테스트 이미지 1장 순전파 → `show_feature_maps()` viridis).
- `SimpleCNN`, `train_one_epoch()`, `evaluate()`, `show_feature_maps()`는 수업 노트북 코드를 그대로 옮긴 것이고, 그래프 2개와 Feature Map 시각화 함수가 과제 부분입니다.

**`motion_trail_numpy.py`** — `main()`: `load_input_image()`(흑백 → 1280x720 → float32 0~1) → `make_weights()`(1.0→0.05 선형 감소, 합 1 정규화) → `motion_trail_np()`(시간 측정) → 출력.
- `motion_trail_np()`: 출력 픽셀 (y, x)마다 shift = 0..31에 대해 `weights[shift] * image[y, x - shift]`를 더합니다. `x - shift < 0`이면 왼쪽 경계 밖이라 더할 값이 없고, shift는 커지기만 하므로 `break`.

**`motion_trail_cupy.py`** — `main()`: CPU 처리 → `cp.asarray()` 업로드(시간) → 64x64로 워밍업 → `motion_trail_cp()`(시간) → `cp.asnumpy()` 다운로드(시간) → 속도 비·최대 오차 출력 → 이미지 3장.
- `motion_trail_cp()`: shift 반복은 Python이 하되, 각 shift에서 `output[:, shift:] += image[:, :W - shift] * weights[shift]` 한 줄을 GPU가 병렬 처리. 픽셀 반복문이 없습니다.

**`convolution_numpy.py`** — `main()`: 이미지 로드 → `run_edge_convolution_cpu()`(세로/가로 edge 필터 2개를 `convolution2d()`로 적용, 시간 측정, ReLU) → 출력.

**`convolution_cupy.py`** — `main()`: 이미지 로드 → CPU 처리 → `run_edge_convolution_gpu()`(변환 → 워밍업 → 처리) → `np.allclose()`로 결과 일치 확인·속도 비 출력 → 이미지 6장.
- `convolution2d_cp()`: 커널 원소 9개를 하나씩 고르고, 그 원소와 곱해질 입력 픽셀 전체를 `padded[ky : ky + stride*(H-1)+1 : stride, kx : ...]` slicing으로 한 번에 뽑아 `output += input_region * kernel[ky, kx]`로 누적합니다. 출력 픽셀(921,600개) 반복이 커널 원소(9개) 반복으로 바뀝니다.

## 실행 방식

```bash
cd Assignments/2026-10-25

# 어디서나 실행 가능 (CPU)
python mnist_cnn_training_plots.py   # 3 epoch 학습 후 그래프·Feature Map 창 5개
python motion_trail_numpy.py         # 잔상 효과 CPU, 약 8초(맥)
python convolution_numpy.py          # edge convolution CPU, 약 7초(맥)

# Jetson(CUDA + CuPy)에서만 실행 가능
python motion_trail_cupy.py          # CPU vs GPU 잔상 효과 시간 비교
python convolution_cupy.py           # CPU vs GPU convolution 결과·시간 비교
```

인자 없이 실행되며, 결과는 matplotlib 창으로 표시되고 `output/` 폴더에 PNG로도 저장됩니다.

## 의존성

| 패키지 | Jetson (JetPack 6.2, 수업 환경) | 맥 검증 환경 |
|---|---|---|
| Python | 3.10 | 3.10.20 |
| NumPy | 1.21.5 | 1.21.5 |
| OpenCV | 4.7 | 4.7.0 |
| Matplotlib | 3.7 | 3.7.5 |
| PyTorch / torchvision | JetPack용 wheel (수업 F절 설치 방법) | 2.8.0 / 0.23.0 (CPU) |
| CuPy | `cupy-cuda12x==12.3.0` + `fastrlock==0.8.3` (수업 G절 설치 방법, `--no-deps`) | 설치 불가 (CUDA 없음) |

- CuPy 설치(Jetson, 수업 노트북 G절과 동일):
  ```bash
  python -m pip download --no-deps --only-binary=:all: "cupy-cuda12x==12.3.0" -d ~/Downloads
  python -m pip install --no-deps "fastrlock==0.8.3"
  python -m pip install --no-deps --only-binary=:all: ~/Downloads/cupy_cuda12x-12.3.0-*.whl
  ```
  `--no-deps`를 쓰는 이유: CuPy가 요구하는 NumPy 최신 버전이 Jetson의 NumPy 1.21.5·OpenCV 4.7과 충돌하므로 의존성 자동 설치를 막습니다.

## 실행 결과 (맥, CPU)

- `mnist_cnn_training_plots.py` (3 epoch, CPU, 총 24.4초):

  | Epoch | Train loss | Train acc | Test loss | Test acc | 시간 |
  |---|---|---|---|---|---|
  | 1 | 0.3133 | 90.52% | 0.0953 | 96.99% | 9.5s |
  | 2 | 0.0925 | 97.14% | 0.0690 | 97.73% | 7.5s |
  | 3 | 0.0666 | 97.92% | 0.0493 | 98.21% | 7.4s |

  결과 PNG: `output/cnn_loss.png`, `output/cnn_accuracy.png`, `output/feature_maps_conv1.png`, `output/feature_maps_conv2.png`, `output/feature_maps_by_layer.png`.
  층별 Feature Map에서 conv1 Channel 2처럼 값이 전부 음수인 채널은 ReLU를 지나면 완전히 0(relu1/pool1 Channel 2가 빈 화면)이 되는 것을 확인할 수 있습니다.

- `motion_trail_numpy.py`: 1280x720, 잔상 길이 32 → **8.15초**. 출력 왼쪽 가장자리가 어두운 띠로 나오는 것은 shift보다 왼쪽에 있는 픽셀은 더할 원본이 없어 밝기가 줄기 때문입니다(의도된 결과).
- `convolution_numpy.py`: 1280x720, 3x3 필터 2개 → **6.75초**.
- CuPy 파일 2개는 맥에서 실행하지 못했습니다(아래 참고 사항).

## 기타 참고 사항

### CuPy 코드 검증 방법 (실제 GPU 미실행)
- 맥에는 CUDA가 없어 `motion_trail_cupy.py`, `convolution_cupy.py`는 **실행하지 못했고 실측 시간이 없습니다.** Jetson에서 돌리면 노트북 수업 설명대로 GPU 처리 시간이 CPU보다 크게 짧고, 복사(업로드·다운로드) 시간을 포함하면 속도 비가 줄어드는 결과가 나와야 합니다.
- 대신 `cupy` 모듈 자리에 NumPy를 끼워 넣는(`sys.modules["cupy"] = numpy 흉내 모듈`) 검증 스크립트로 **로직만** 확인했습니다. 37x53 난수 이미지로 `motion_trail_cp()` vs `motion_trail_np()`(가중치 길이 32·4), `convolution2d_cp()` vs `convolution2d()`(3x3·5x5 커널, stride 1·2·3, padding 0·1·2)를 비교해 모두 `np.allclose` 통과(최대 오차 2e-6 이하, float32 누적 순서 차이).
- 검증하지 못한 부분: `cp.cuda.Stream.null.synchronize()`의 타이밍, 실제 속도 비, Jetson 메모리 부족 여부.

### 수업 중 작성한 `motion_trail_np()` 초안의 오류 3가지 (이번 제출에서 수정)
1. `output = input_np.copy()`로 시작 → 함수 밖 전역 변수를 쓰는 데다, shift=0 항(원본 × weights[0])이 이미 들어 있는 원본 위에 또 더해져 이중 누적. → `np.zeros_like(image)`에서 0부터 누적.
2. `min(255, max(0, …))` 클램프 → 입력은 이미 0~1 float32이고 가중치 합이 1이라 결과가 0~1을 벗어날 수 없음. 255 기준 클램프는 아무 효과가 없어 삭제.
3. `range(TRAIL_LENGTH)` 상수 반복 → 노트북 워밍업 셀은 길이 4짜리 가중치를 넘기므로 `weights[4]`에서 IndexError. → `range(len(weights))`.

### 설계 메모
- `convolution2d_cp()`의 slicing 끝 인덱스를 `ky + stride*(H-1) + 1`로 쓴 이유: 출력 크기 H개를 stride 간격으로 정확히 뽑기 위함. stride=1이면 `ky : ky + H`와 같습니다.
- 워밍업을 64x64 작은 입력으로 하는 이유: 첫 CUDA 호출에 섞이는 커널 준비·메모리 할당 시간을 측정에서 빼되, 본 처리와 같은 코드 경로를 한 번 지나가게 하기 위함.
- MNIST 학습은 수업 F절과 같이 CPU로 고정했습니다(GPU 학습 비교는 H절 범위라 이번 과제에서 제외).

### AI 활용 vs 본인 기여
- 본인: 과제 범위(노트북 TODO 선별) 결정, NumPy/CuPy 파일 분리·import 연결 구조 결정, `motion_trail_np()` 초안 작성(수업 중) 및 오류 수정 방향 확인, 실행·결과 확인.
- AI 보조: 노트북 셀을 함수·`main()` 구조의 `.py`로 옮기는 작업, CuPy 함수(`motion_trail_cp`, `convolution2d_cp`)의 slicing 식 작성, cupy→numpy 치환 검증 스크립트, docstring·주석·README 정리.
- 수업 중 작성한 노트북 원본은 `Labs/2026-10-25/`에 있습니다.
