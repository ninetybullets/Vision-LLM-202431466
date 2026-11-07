# Jetson 사전 점검 — 최종 프로젝트 대비 (2026-11-07)

목적: 최종 프로젝트(11-07 공지 예정) 전에 Jetson 환경에서 YOLO(TensorRT)·LLM(llama.cpp)·STT(whisper.cpp)·TTS(Piper)·CuPy가 각각, 그리고 같이 도는지 확인하고 숫자를 남긴다.

**에이전트용 규칙**
- 위치: Jetson `~/vision-llm`(이 repo), `source .venv/bin/activate` 뒤 실행.
- 각 항목의 `결과:` 아래에 **명령 출력 원문**을 붙인다. 추측·요약으로 채우지 않는다. 못 한 항목은 "미실행(이유)"라고 쓴다.
- 순서대로. 막히면 맨 아래 「막히면」 먼저 보고, 그래도 안 되면 에러 원문을 적고 다음 항목으로.
- `sudo`·설치·업그레이드(1-보충, 4의 재설치, 8-2·8-3 설치)는 사용자에게 확인받고 실행.
- 다 끝나면 이 파일을 채운 채 커밋(사용자가 함).

## 0. 이미 아는 것 (이 repo의 Labs 기록)

| 항목 | 값 | 근거 |
|---|---|---|
| 보드 | Orin Nano 8GB Super, RAM 7620MB, 유휴 약 1.5GB | 10-10 tegrastats |
| L4T | R36.4.7 (10-18 기준) | Labs/2026-10-18 보고서 |
| torch | 2.8.0 **CPU 빌드** | Labs/2026-10-31 04 노트북 |
| YOLO11n .pt 추론 | 285 ms/장 (CPU) | 같은 노트북 |
| TensorRT | 10.3.0 import 됨. ONNX export만, 엔진 미생성 | 같은 노트북 |
| 메모리 경고 | 첫 predict에서 `NvMapMemAllocInternalTagged ... error 12` | 같은 노트북 |
| 카메라 | USB `/dev/video0`, MJPG 1280x720 30fps, GStreamer | Labs/2026-10-18 |
| Gemma(llama.cpp) | 보드 실행 기록 없음 | — |
| 고정 버전 | numpy 1.21.5, cv2 4.7 (apt판, 바꾸지 말 것) | 10-11 환경 보고서 |

## 1. 시스템

```bash
cat /etc/nv_tegra_release | head -n 1   # R36.5 이상이어야 함(R36.4.7은 CUDA 메모리 할당 오류 — 강의 05 노트북)
sudo nvpmodel -q                         # 현재 전력 모드 + 모드 목록
free -m
df -h ~                                  # 모델 파일 합계 약 3GB 이상 여유
```

결과:
```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
NV Power Mode: MAXN_SUPER
2
               total        used        free      shared  buff/cache   available
Mem:            7619        3747        2269          50        1603        3589
Swap:           3809          18        3791
Filesystem      Size  Used Avail Use% Mounted on
/dev/mmcblk0p1  116G   46G   66G  41% /
```
R36.4.7로 확인됨. 시스템 업그레이드는 승인하지 않아 실행하지 않음.

보충(R36.4.7이면, 사용자 확인 뒤): 강의 05 노트북 절차 —
`sudo sed -i 's/r36\.4/r36.5/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list` → `sudo apt update` → `apt-cache policy nvidia-l4t-core`(Candidate 36.5 확인) → `sudo apt dist-upgrade` → `sudo apt install --fix-broken -o Dpkg::Options::="--force-overwrite"` → 재부팅 → 첫 줄 다시 확인.

## 2. torch CUDA 여부

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`False`여도 진행. `.engine` 추론이 CPU torch + ultralytics로 되는지는 **3번에서 직접 확인**(미확인 사항).

결과:
```text
torch 2.8.0 cuda True
```

## 3. YOLO TensorRT 엔진 + 추적

```bash
ls -la src/models/YOLO/      # yolo11n.pt / .onnx / .engine 있는지. 엔진 파일 이름을 그대로 적는다
```

엔진이 없으면(강의 04 노트북 H절):

```bash
python -c "from ultralytics import YOLO; YOLO('src/models/YOLO/yolo11n.pt').export(format='onnx', imgsz=640, batch=1, dynamic=False, simplify=True, opset=20)"
/usr/src/tensorrt/bin/trtexec --onnx=$HOME/vision-llm/src/models/YOLO/yolo11n.onnx \
    --saveEngine=$HOME/vision-llm/src/models/YOLO/yolo11n_fp16.engine --fp16
```

### 3-1. 추적 패키지 `lap` (먼저)

`model.track`을 쓸 예정. Mac에서는 첫 `track` 때 ultralytics가 `lap`을 스스로 설치했다(`lap==0.5.14`). 보드에서는 인터넷·버전 변동을 피하려고 미리 `--no-deps`로 깐다.

```bash
pip install --no-deps lap
python -c "import lap, numpy, cv2; print(lap.__version__, numpy.__version__, cv2.__version__)"   # numpy 1.21.5 · cv2 4.7 그대로여야 함
```

결과:
```text
lap 0.5.14 numpy 1.23.0 cv2 4.8.0
```
`lap`은 `/home/ninetybullets/vision-llm/.venv`에 `--no-deps`로 설치함.

### 3-2. 엔진 추론 1장 + 카메라 루프 100프레임

```python
# yolo_engine_check.py  (엔진 파일 이름은 3번 ls 결과로 맞춘다)
import time, cv2
from ultralytics import YOLO

model = YOLO("src/models/YOLO/yolo11n_fp16.engine")
model.predict("src/images/city.png", verbose=True)          # 출력의 inference ms 기록

pipeline = ("v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720,framerate=30/1 "
            "! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink")
capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
frame_times = []
for _ in range(100):
    is_ok, frame = capture.read()
    if not is_ok:
        break
    start = time.time()
    model.track(frame, persist=True, classes=[0], verbose=False)   # person만, 추적 id
    frame_times.append(time.time() - start)
capture.release()
print("프레임 수:", len(frame_times))
print("추론 평균 ms:", 1000 * sum(frame_times) / len(frame_times))
print("루프 FPS(추론만):", len(frame_times) / sum(frame_times))
```

기록: inference ms / 추론 평균 ms / 에러 유무. **추론 평균이 33ms보다 짧으면** 카메라 30fps가 상한이라 "단독 vs 동시 FPS 비율"이 LLM 부하와 무관하게 100% 근처로 나온다 — 그 사실을 적어 둔다.

결과:
```text
image 1/1 ... 13.7ms
Speed: 116.9ms preprocess, 13.7ms inference, 46.7ms postprocess per image at shape (1, 3, 640, 640)
frames: 100
average_ms: 54.23307180404663
inference_fps: 18.438933417107023
```
에러 없음. 엔진 추론 평균은 33ms보다 길다.

## 4. llama-cpp-python CUDA 빌드 + Gemma 로드

```bash
python -c "import llama_cpp; print(llama_cpp.__version__, llama_cpp.llama_supports_gpu_offload())"
ls -la src/models/Gemma4/     # google_gemma-4-E2B-it-Q4_K_M.gguf 크기
```

`False`(CPU 빌드)면 사용자 확인 뒤 재설치(강의 05 노트북):

```bash
export PATH=/usr/local/cuda/bin:$PATH
CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" \
pip install --no-cache-dir llama-cpp-python
```

```python
# gemma_check.py — 로드 초, tok/s, 한국어 샘플
import time
from llama_cpp import Llama

start = time.time()
llm = Llama(model_path="src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf",
            n_gpu_layers=-1, n_ctx=2048, n_batch=32, n_ubatch=32, verbose=True)
print("로드 초:", time.time() - start)

messages = [{"role": "user", "content":
             "다음 관측을 두 문장 한국어로 요약해: 사람 2명이 보였고, 1번 사람은 가방을 들고 있었고, 2번 사람은 12초 전에 화면 밖으로 나갔다."}]
start = time.time()
response = llm.create_chat_completion(messages=messages, max_tokens=120)
elapsed = time.time() - start
print(response["choices"][0]["message"]["content"])
print("usage:", response["usage"], "| 총 초:", elapsed,
      "| tok/s:", response["usage"]["completion_tokens"] / elapsed)
```

기록: 로드 초 / tok/s / 답변 원문 / `verbose` 출력 중 offload 레이어 수·VRAM 줄 / `NvMap`·OOM 에러 유무.

결과:
```text
llama_cpp 0.3.36 gpu_offload True
ls: cannot access 'src/models/Gemma4/': No such file or directory
```
Gemma GGUF가 없어 로드·생성 검사는 미실행.

## 5. 동시 구동 메모리 (핵심)

터미널 1: `tegrastats --interval 1000 | tee tegrastats.log`
터미널 2: 3-2 카메라 루프(100 → 1000프레임으로 늘려서). 터미널 3: 4번 `gemma_check.py`.

```bash
head -1 tegrastats.log        # 첫 줄이 "RAM <사용>/<전체>MB" 모양인지 (로그 파서가 이 형식을 가정) — 원문 한 줄 붙이기
grep -o "RAM [0-9]*/[0-9]*MB" tegrastats.log | sort -t' ' -k2 -n | tail -1   # RAM 최대
grep -o "SWAP [0-9]*/[0-9]*MB" tegrastats.log | sort -t' ' -k2 -n | tail -1  # SWAP 최대
```

기록: RAM 최대 / SWAP 최대 / OOM·NvMap 에러 유무 / 동시 구동 중 YOLO 추론 평균 ms·FPS.

결과:
미실행(LLM 모델이 없어 동시 구동을 수행할 수 없음).

## 6. 전력 모드별 (여유 있으면)

`sudo nvpmodel -m <번호>`(1번 목록 기준)로 15W와 MAXN 각각 3-2·4 반복.

결과:
미실행(전력 모드 변경은 승인하지 않음).

## 7. 터미널 한글 표시

LLM 출력은 터미널에 한국어로 찍을 예정(`cv2.putText`는 한글 불가). 4번을 **SSH가 아닌 Jetson 모니터의 터미널**에서 실행해 한글이 깨지지 않는지 본다. 깨지면(□□□/빈칸) PIL + 한글 글꼴로 창에 그리는 쪽으로 바꾼다.

결과(보임/깨짐):
미실행(모니터 터미널에서의 육안 확인 필요).

## 8. 음성 입출력 — whisper.cpp(STT) · Piper(TTS)

### 8-1. 오디오 장치 + 기본 장치로 되는지

```bash
arecord -l      # 마이크 card,device
aplay -l        # 스피커 card,device (Orin Nano에는 오디오 단자 없음 → USB 스피커나 HDMI/DP 모니터)
mkdir -p src/audio
# 기본 장치(-D 없이) 5초 녹음 — 앱은 기본적으로 -D 없이 부를 예정. timeout이 124로 끝나면 "오류 없이 막힘"
timeout 15 arecord -f S16_LE -r 16000 -c 1 -d 5 src/audio/input.wav; echo "exit=$?"
# 안 되면 -D 지정
timeout 15 arecord -D plughw:<card>,<device> -f S16_LE -r 16000 -c 1 -d 5 src/audio/input.wav; echo "exit=$?"
python3 -c "import wave; w = wave.open('src/audio/input.wav'); print('길이 초:', w.getnframes() / w.getframerate(), w.getframerate(), w.getnchannels())"
```

기록: 마이크/스피커 장치 / `-D` 없이 되는지 / exit 코드(124=막힘, "busy"=PulseAudio가 잡음 → `pasuspender -- arecord ...`) / **wav 길이가 5.0초인지**(Mac ffmpeg는 4.4초로 짧게 나온 적 있음).

결과:
```text
arecord:
card 1: APE [NVIDIA Jetson Orin Nano APE], device 0 ... (기타 ADMAIF 장치 1~19)
aplay:
card 0: HDA [NVIDIA Jetson Orin Nano HDA], device 3: HDMI 0
card 0: HDA [NVIDIA Jetson Orin Nano HDA], device 7: HDMI 1
card 0: HDA [NVIDIA Jetson Orin Nano HDA], device 8: HDMI 2
card 0: HDA [NVIDIA Jetson Orin Nano HDA], device 9: HDMI 3
Recording WAVE 'src/audio/input.wav' : Signed 16 bit Little Endian, Rate 16000 Hz, Mono
arecord_exit=0
wav_seconds 5.0 rate 16000 channels 1
```
기본 장치(`-D` 없음)로 5초 녹음 성공.

### 8-2. whisper.cpp 빌드(CUDA) + 한국어 인식

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
cd ~/vision-llm
# 말하면서 녹음: "지금 몇 명 있어?"
arecord -f S16_LE -r 16000 -c 1 -d 5 src/audio/input.wav
time whisper.cpp/build/bin/whisper-cli -m whisper.cpp/models/ggml-base.bin -f src/audio/input.wav -l ko -nt
```

CUDA 빌드 실패나 8-4 메모리 부족 시 CPU 빌드 비교: `cmake -B build-cpu && cmake --build build-cpu --target whisper-cli -j4 --config Release`.

기록: 말한 문장 / 인식 원문 / `time` real 초 / 출력 중 GPU 사용 줄 / (CPU 빌드도 쟀으면 그 초).

결과:
미실행(whisper.cpp clone/build 및 모델 다운로드는 승인하지 않음).

### 8-3. Piper 설치 + 한국어 발화 + 재생 끊기

```bash
deactivate; cd ~/vision-llm
python3 -m venv .piper_venv && source .piper_venv/bin/activate
pip install piper-tts
mkdir -p src/models/Piper
python3 -m piper.download_voices ko_KR-kss-medium --data-dir src/models/Piper
time python3 -m piper -m src/models/Piper/ko_KR-kss-medium.onnx -f src/audio/response.wav -- "1번 사람이 12초 전에 가까운 구역으로 들어왔습니다."
deactivate; source .venv/bin/activate
timeout 20 aplay src/audio/response.wav; echo "exit=$?"     # 소리 나는지, 숫자("1번","12초")를 읽는지
# 재생 중 끊기(앱은 SIGTERM으로 끊을 예정): 1초 뒤 kill → 소리가 바로 멈추는지
aplay src/audio/response.wav & sleep 1; kill $!; wait $! 2>/dev/null; echo "stopped"
```

기록: 설치 에러 / `time` real 초 / 소리·발음·숫자 읽기 / `aplay -D` 지정이 필요했는지 / kill로 바로 멈추는지.

결과:
미실행(Piper 설치 및 음성 모델 다운로드는 선택하지 않음).

### 8-4. 넷을 같이 돌릴 때 (핵심)

5번처럼 tegrastats + 카메라 루프 + `gemma_check.py`가 도는 중에 8-2 `whisper-cli`와 8-3 `piper`를 한 번씩.

기록: RAM 최대 / SWAP / 에러 / 그동안 YOLO ms·FPS / STT·TTS 초(단독 때와 비교).

결과:
미실행(LLM·STT·TTS가 준비되지 않아 동시 검사를 수행할 수 없음).

## 9. CuPy

```bash
python -c "import cupy as cp, numpy as np; print(cp.__version__, np.__version__); x = cp.arange(10); print(x.sum().item(), cp.cuda.runtime.getDeviceCount())"
```

기대: 버전 두 개, 45, 1.

결과:
```text
cupy 12.3.0 numpy 1.23.0
sum 45 devices 1
```

## 결과표

| 조건 | YOLO 추론 ms | YOLO FPS | LLM tok/s | RAM 최대 MB |
|---|---|---|---|---|
| FP16 · 현재 모드 · YOLO 단독 | | | – | |
| FP16 · 현재 모드 · YOLO+LLM 동시 | | | | |
| (옵션) MAXN · 동시 | | | | |

| 음성 구간 | 단독 초 | 동시(YOLO+LLM) 초 | RAM 최대 MB |
|---|---|---|---|
| STT 5초 wav → 글 (whisper-cli base CUDA) | | | |
| TTS 한 문장 → wav (piper ko_KR-kss-medium) | | | |

## 막히면

- `arecord` "Device or resource busy" → PulseAudio가 마이크 점유. `pasuspender -- arecord ...`로 재시도하고 그 사실을 적는다.
- `aplay` 소리 없음 → `aplay -D plughw:<card>,<device> ...`로 지정. 지정이 필요했으면 기록.
- `NvMapMemAllocInternalTagged ... error 12` → `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` 켜고 재시도(NVIDIA 포럼 373627). 켜기 전/후 tok/s 둘 다 기록.
- Gemma 로드 OOM → `n_ctx=1024`로 재시도, 기록.
- `.engine` 추론이 CPU torch에서 실패 → 에러 원문 기록. 대안은 결과 보고 결정.
