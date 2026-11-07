---
title: 최종 프로젝트 — 보드 체크리스트
brand: 온디바이스 AI
nav:
  - 후보 비교 | topic-candidates.html
  - 설계 초안 | design-draft.html
  - 보드 체크 | board-check.html
  - Mac 설치 | mac-setup.html
  - 조사 보고서 | research-report.html
  - 선행 사례 | landscape-sweep.html
---

# 보드 체크리스트 — 최종 프로젝트 전 Jetson 상태 확인

작성 2026-11-04. Jetson(`~/vision-llm`, `.venv` 활성화)에서 순서대로 실행하고 **출력 원문**을 각 항목 아래 붙여 주세요. 추측으로 채우지 말 것.

## 이미 아는 것(제출 repo 실습 기록 기준)

| 항목 | 값 | 근거 |
|---|---|---|
| 보드 | Orin Nano 8GB Super, RAM 7620MB, 유휴 약 1.5GB 사용 | NOTES 10-10 tegrastats |
| L4T | R36.4.7 (10-18 기준) | Labs/2026-10-18 보고서 |
| torch | 2.8.0 **CPU 빌드** | Labs/2026-10-31 04 nb 출력 |
| YOLO11n PyTorch 추론 | 285 ms/장 (CPU) | 같은 nb, exec 10 |
| TensorRT | 10.3.0 import 됨. ONNX export 시작, INT8 셀 미실행 | 같은 nb |
| 메모리 경고 | 첫 predict에서 `NvMapMemAllocInternalTagged ... error 12` | 같은 nb, exec 9 |
| 카메라 | USB `/dev/video0`, MJPG 1280x720 30fps, GStreamer | Labs/2026-10-18 보고서 |
| Gemma | 보드 실행 기록 없음 | Labs에 11-01 파일 없음 |

## 1. 시스템

```bash
cat /etc/nv_tegra_release | head -n 1        # R36.5 이상인지 (05 nb §F-3 요구)
sudo nvpmodel -q                              # 현재 전력 모드
free -m                                       # RAM / SWAP(zram)
df -h ~                                       # 모델 파일 둘 공간(GGUF 약 2~3GB 예상, 확인 필요)
```

결과:

## 2. torch CUDA 여부 (YOLO GPU 경로의 전제)

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`False`면 PyTorch 경로는 CPU. TensorRT `.engine` 추론이 CUDA torch 없이 ultralytics에서 되는지는 **미확인** → 3번에서 직접 확인.

결과:

## 3. YOLO TensorRT 엔진

```bash
ls -la src/models/YOLO/            # yolo11n.onnx / .engine 존재 여부
```

없으면 04 nb §H 순서(ONNX → FP16 engine)대로 만든 뒤:

```python
# yolo_engine_check.py — 엔진 추론 1장 시간 + 카메라 루프 FPS
import time, cv2
from ultralytics import YOLO

model = YOLO("src/models/YOLO/yolo11n.engine")
model.predict("src/images/city.png", verbose=True)   # 출력의 inference ms 기록

# 카메라 루프 100프레임 FPS (04 nb §E 파이프라인과 동일 카메라 설정)
pipeline = ("v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720,framerate=30/1 "
            "! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink")
capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
start = time.time()
for _ in range(100):
    is_ok, frame = capture.read()
    if not is_ok:
        break
    model.predict(frame, verbose=False)
print("YOLO 단독 FPS:", 100 / (time.time() - start))
```

결과(inference ms, 단독 FPS, 에러 유무):

### 3-1. 추적용 패키지 `lap`

추가 2026-11-06. 프로젝트의 `detector.py`는 `model.predict`가 아니라 `model.track`(추적 id)을 쓴다. Mac에서는 첫 `model.track` 때 ultralytics가 `lap`이 없다며 스스로 설치했다(`lap==0.5.14`). 보드에서 같은 일이 생기면 첫 실행에 인터넷이 필요하고, 고정해 둔 numpy 1.21.5·cv2 4.7이 바뀌는지는 **미확인**이다. 그래서 미리 `--no-deps`로 깔고 버전을 확인한다.

```bash
pip install --no-deps lap
python -c "import lap, numpy, cv2; print(lap.__version__, numpy.__version__, cv2.__version__)"
```

결과(lap 버전 / numpy·cv2 버전이 그대로인지 / 설치 에러 원문):

## 4. llama-cpp-python CUDA 빌드 + Gemma 로드

```bash
python -c "import llama_cpp; print(llama_cpp.__version__, llama_cpp.llama_supports_gpu_offload())"
ls -la src/models/Gemma4/          # google_gemma-4-E2B-it-Q4_K_M.gguf, mmproj-*.gguf 크기
```

`llama_supports_gpu_offload()`가 `False`면 CPU 빌드 → 05 nb §F-2 명령으로 재설치.

```python
# gemma_check.py — 로드 시간, tok/s, 한국어 샘플
import time
from llama_cpp import Llama

start = time.time()
llm = Llama(model_path="src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf",
            n_gpu_layers=-1, n_ctx=2048, n_batch=32, n_ubatch=32, verbose=True)
print("로드 초:", time.time() - start)

messages = [{"role": "user", "content":
             "다음 관측을 두 문장 한국어 상황 보고로 요약해: 사람 2명, 1번 사람이 knife 보유, 2번 사람 화면 이탈."}]
start = time.time()
response = llm.create_chat_completion(messages=messages, max_tokens=120)
elapsed = time.time() - start
print(response["choices"][0]["message"]["content"])
print("usage:", response["usage"], "| 총 초:", elapsed,
      "| tok/s:", response["usage"]["completion_tokens"] / elapsed)
```

`verbose=True` 출력에 GPU offload된 레이어 수·VRAM 사용량이 찍힙니다. 그 줄도 붙여 주세요.

결과(로드 초, tok/s, 한국어 답변 원문, offload 줄):

## 5. 동시 구동 메모리 (핵심)

터미널 1: `tegrastats --interval 1000 | tee tegrastats.log`
터미널 2: 3번 카메라 루프를 돌리면서, 터미널 3에서 4번 `gemma_check.py` 실행.

기록할 것: tegrastats `RAM x/7620MB` 최대값, `SWAP` 사용량, OOM·NvMap 에러 유무, 동시 구동 중 YOLO FPS.

결과:

## 6. 전력 모드별 (여유 있으면)

`sudo nvpmodel -m <번호>`로 15W와 MAXN 각각에서 3·4번 반복. 모드 번호는 `nvpmodel -q`의 목록 기준.

결과:

## 7. 터미널 한글 표시 (SITREP 출력 위치 결정용)

SITREP과 Evidence는 터미널에 출력하기로 했다(HUD의 `cv2.putText`는 한글을 못 그림). 4번 `gemma_check.py`를 **SSH가 아니라 Jetson에 연결한 모니터의 터미널**에서 실행해, 한국어 답변이 깨지지 않고 보이는지 확인한다.

깨지면(□□□ 또는 빈칸): 글꼴 파일을 프로젝트에 넣고 PIL로 HUD 창 안에 그리는 방식으로 바꾼다.

결과(보임 / 깨짐):

## 8. 음성 입출력 — Whisper.cpp(STT) · Piper(TTS)

추가 2026-11-06. 최종 프로젝트가 "유저 음성 입력을 받아 음성으로 출력"을 요구한다(강의 repo `5765dfd`의 06 노트북 셀 108). 명령은 같은 노트북 C절(셀 56~103) 그대로다. 그 노트북은 다음 커밋에서 삭제됐으니 11-07 수업에서 바뀌면 수업 내용을 따른다.

### 8-1. 오디오 장치

```bash
arecord -l        # 마이크: card 번호, device 번호 기록
aplay -l          # 스피커: card 번호, device 번호 기록
```

결과(마이크 `plughw:?,?`, 스피커 장치명):

### 8-2. Whisper.cpp 빌드 + 한국어 인식 1회

```bash
cd ~/vision-llm
git clone https://github.com/ggml-org/whisper.cpp.git
sudo apt install -y cmake
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
cd whisper.cpp
cmake -B build -DGGML_CUDA=1
cmake --build build -j4 --config Release
ls -lh build/bin/whisper-cli
sh ./models/download-ggml-model.sh base
ls -lh models/ggml-base.bin

mkdir -p ../src/audio
# -D 값은 8-1에서 확인한 번호로 바꾼다. 16kHz·mono·5초
arecord -D plughw:2,0 -f S16_LE -r 16000 -c 1 -d 5 $HOME/vision-llm/src/audio/input.wav
# 녹음된 길이가 정말 5초인지 확인 (Mac에서는 4.4초로 짧게 나온 적이 있다)
python3 -c "import wave; w = wave.open('$HOME/vision-llm/src/audio/input.wav'); print(w.getnframes() / w.getframerate())"
time build/bin/whisper-cli -m models/ggml-base.bin -f $HOME/vision-llm/src/audio/input.wav -l ko -nt
```

기본은 CUDA 빌드(`build/`)다(2026-11-06 결정: Jetson 트랙은 GPU 연산이 기본). 노트북의 Python 코드는 CPU 빌드 경로(`build-cpu/bin/whisper-cli`)를 쓰므로 경로를 `build/bin/whisper-cli`로 바꿔야 한다. `whisper-cli` 출력에 GPU를 썼다는 줄이 있는지 확인해 그 줄도 붙인다.

CUDA 빌드가 실패하거나 8-4에서 메모리가 모자라면 CPU 빌드로 비교한다:

```bash
cmake -B build-cpu
cmake --build build-cpu --target whisper-cli -j4 --config Release
time build-cpu/bin/whisper-cli -m models/ggml-base.bin -f $HOME/vision-llm/src/audio/input.wav -l ko -nt
```

Mac에서는 2026-11-06에 `ffmpeg` 녹음 wav가 5초가 아니라 약 4.4초로 나왔다. 마이크 입력의 타임스탬프가 건너뛰어 생긴 일이었고 Mac 쪽 명령은 고쳤다(`platform_config.py`). `arecord -d 5`에도 같은 현상이 있는지는 **미확인**이라 위 길이 확인 줄을 넣었다. 5.0보다 뚜렷이 짧으면 그 값을 적는다.

결과(녹음 길이 초 / 말한 문장 / 인식된 문장 원문 / CUDA `time` real 초 / GPU 사용 줄 / CPU 빌드도 쟀으면 그 초):

### 8-3. Piper 설치 + 한국어 발화 1회

```bash
deactivate
cd ~/vision-llm
python3 -m venv .piper_venv
source .piper_venv/bin/activate
pip install piper-tts
mkdir -p src/models/Piper
python3 -m piper.download_voices ko_KR-kss-medium --data-dir src/models/Piper
time python3 -m piper -m src/models/Piper/ko_KR-kss-medium.onnx -f src/audio/response.wav -- "표적 한 명이 가까운 구역으로 들어왔습니다."
aplay src/audio/response.wav
deactivate && source .venv/bin/activate
```

결과(설치 에러 유무 / `time`의 real 초 / 소리가 났는지, 발음이 알아들을 만한지):

### 8-4. 넷을 같이 돌릴 때 (핵심)

5번과 같은 방식으로 tegrastats를 켜고, YOLO 카메라 루프 + `gemma_check.py`가 도는 중에 8-2의 `whisper-cli`와 8-3의 `piper`를 한 번씩 실행한다.

기록할 것: RAM 최대값, SWAP, OOM·NvMap 에러 유무, 그동안 YOLO FPS, STT·TTS 각각 걸린 초(단독 때와 비교).

결과:

## 9. CuPy

추가 2026-11-06. Jetson 트랙은 배열 연산에 CuPy를 쓴다. 03 노트북 G절에서 `cupy-cuda12x==12.3.0`을 설치했는지, 지금 `.venv`에서 불러와지는지 확인한다.

```bash
python -c "import cupy as cp, numpy as np; print(cp.__version__, np.__version__); x = cp.arange(10); print(x.sum().item(), cp.cuda.runtime.getDeviceCount())"
```

결과(버전 두 개, 합 45, 장치 수 1이 나오는지 / 에러 원문):

## 결과표 (채워서 돌려주기)

| 조건 | YOLO FPS | LLM tok/s | 이벤트 1건 응답 초 | RAM 피크 MB |
|---|---|---|---|---|
| FP16 · 현재 모드 · YOLO 단독 | | – | – | |
| FP16 · 현재 모드 · 동시 | | | | |
| INT8 · 현재 모드 · 동시 | | | | |
| (옵션) MAXN · 동시 | | | | |

| 음성 구간 | 단독 초 | 동시(YOLO+Gemma) 초 | RAM 피크 MB |
|---|---|---|---|
| STT: 5초 녹음 → 글 (`whisper-cli`, base, CUDA) | | | |
| TTS: 한 문장 → wav (`piper`, ko_KR-kss-medium) | | | |

## 막히면

- `arecord`가 "Device or resource busy" → PulseAudio가 마이크를 잡고 있는 것. 노트북 Python 코드처럼 앞에 `pasuspender --`를 붙인다.
- `aplay`에서 소리가 안 남 → `aplay -l`의 장치로 `aplay -D plughw:<card>,<device> ...`를 지정해 본다. Orin Nano 개발 보드에는 오디오 단자가 없어 USB 스피커나 모니터(HDMI/DP) 소리를 써야 한다.

- R36.4.7에서 CUDA 메모리 오류 → 05 nb §F-3 업그레이드 먼저.
- `NvMapMemAllocInternalTagged ... error 12`(10-31 실습에서 이미 봄)가 Gemma 로드·YOLO 동시 구동에서 또 나오면: 환경변수 `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`을 켜고 재시도(NVIDIA 포럼 373627, 스태프 확인. 모델 의존일 수 있음). 켜기 전/후 tok/s를 둘 다 기록.
- Gemma 로드 OOM → `n_ctx` 1024로 낮춰 재시도, 결과 기록.
- `.engine` 추론이 CPU torch에서 실패 → 에러 원문 기록. 대안(ONNX Runtime, trtexec 직접 호출)은 결과 보고 결정.
