# Jetson 사전 점검 — 최종 프로젝트 대비 (2026-11-07)

목적: 최종 프로젝트(11-07 공지 예정) 전에 Jetson 환경에서 YOLO(TensorRT)·LLM(llama.cpp)·STT(whisper.cpp)·TTS(Piper)·CuPy가 각각, 그리고 같이 도는지 확인하고 숫자를 남긴다.

## 현재 상황 업데이트 (2026-11-07 16:33 KST)

- Linux/Jetson 업데이트 후 L4T R36.5.2로 확인했다. 이전 기록의 R36.4.7은 업데이트 전 상태다.
- Gemma GGUF와 mmproj 파일을 설치했고, `.venv`에서 llama.cpp GPU 오프로드·모델 로드·한국어 생성을 확인했다.
- 현재 확인: torch 2.8.0 CUDA 사용 가능, llama-cpp-python 0.3.36 GPU 오프로드 가능, CuPy 12.3.0에서 CUDA 장치 1개, `lap` 0.5.14.
- whisper.cpp CPU 빌드와 base 모델, Piper 1.8.0과 한국어 음성 모델도 설치되어 있다. STT/TTS 병렬 처리 시간은 측정했지만 실제 스피커 청취는 확인하지 않았다.
- STT와 TTS를 기존 `input.wav`를 사용해 병렬 실행했고 둘 다 성공했다. `STT_TTS_test01.py` 자체는 STT 완료 후 TTS를 실행하는 순차 흐름이다.
- 미완료: YOLO+Gemma 동시 구동, STT/TTS 실제 청취, 4개 구성요소 동시 구동, 전력 모드별 측정, 모니터 터미널 한글 표시 확인.
- `nvpmodel -q`는 sudo 비밀번호가 필요해 이번 확인에서는 실행하지 않았다. 설치·업그레이드 명령은 실행하지 않았다.
- 저장소 상태: 아직 Git 커밋이 없으며 현재 파일들은 모두 미추적 상태다. 점검이 끝난 뒤 사용자가 검토 후 커밋해야 한다.

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
| L4T | R36.5.2 (2026-11-07 업데이트 후) | `/etc/nv_tegra_release` |
| torch | 2.8.0 CUDA 사용 가능 | `.venv` 확인 |
| YOLO11n .pt 추론 | 285 ms/장 (CPU) | 같은 노트북 |
| TensorRT | 10.3.0 import 됨. ONNX export만, 엔진 미생성 | 같은 노트북 |
| 메모리 경고 | 첫 predict에서 `NvMapMemAllocInternalTagged ... error 12` | 같은 노트북 |
| 카메라 | USB `/dev/video0`, MJPG 1280x720 30fps, GStreamer | Labs/2026-10-18 |
| Gemma(llama.cpp) | GGUF 설치 및 한국어 생성 확인, 1.248 tok/s | 2026-11-07 `.venv` 확인 |
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
# R36 (release), REVISION: 5.2, GCID: 46426093, BOARD: generic, EABI: aarch64, DATE: Thu Jul 16 18:56:22 UTC 2026
sudo: a password is required
               total        used        free      shared  buff/cache   available
Mem:            7607        3766          78          68        3762        3539
Swap:           3803          17        3786
Filesystem      Size  Used Avail Use% Mounted on
/dev/mmcblk0p1  116G   54G   58G  49% /
```
R36.5.2로 업데이트된 것을 확인했다. `nvpmodel -q`는 sudo 비밀번호가 필요해 이번 확인에서는 실행하지 않았다.

보충(R36.4.7이면, 사용자 확인 뒤): 강의 05 노트북 절차 —
`sudo sed -i 's/r36\.4/r36.5/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list` → `sudo apt update` → `apt-cache policy nvidia-l4t-core`(Candidate 36.5 확인) → `sudo apt dist-upgrade` → `sudo apt install --fix-broken -o Dpkg::Options::="--force-overwrite"` → 재부팅 → 첫 줄 다시 확인.

## 2. torch CUDA 여부

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`False`여도 진행. `.engine` 추론이 CPU torch + ultralytics로 되는지는 **3번에서 직접 확인**(미확인 사항).

결과:
```text
torch 2.8.0 cuda=True
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
src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf 3462680032 bytes
src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf 985653760 bytes
load_seconds 74.018
response **두 사람이 보였다.**
usage {'prompt_tokens': 27, 'completion_tokens': 6, 'total_tokens': 33}
completion_tok_per_sec 1.248
```
Gemma 모델 로드와 한국어 생성에 성공했다. `n_ctx=1024`로 실행했으며 llama.cpp 경고 2건(SWA cache, V cache padding)이 출력됐다.

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
미실행(모델은 준비됐지만 YOLO와 Gemma 동시 측정을 아직 수행하지 않음).

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
```text
whisper.cpp/build-cpu/bin/whisper-cli
whisper.cpp/models/ggml-base.bin
```
병렬 실행 결과:
```text
whisper_backend_init_gpu: device 0: CPU (type: 0)
whisper_backend_init_gpu: no GPU found
[몇일이 없음]
whisper_print_timings:
    total time = 7330.21 ms
stt_rc=0 ... elapsed_seconds=7.546
```
STT는 성공했지만 Whisper는 실제 CUDA GPU가 아니라 CPU backend로 실행됐다. 인식 결과는 `[몇일이 없음]`이다.

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
```text
piper-tts 1.8.0
src/models/Piper/ko_KR-kss-medium.onnx
src/models/Piper/ko_KR-kss-medium.onnx.json
```
병렬 실행 결과:
```text
tts_rc=0 ... elapsed_seconds=9.295
/tmp/response_parallel.wav 165K
```
Piper가 정상적으로 WAV를 생성했다. 실행 중 ONNX Runtime의 GPU device discovery 경고가 출력됐으며, 실제 스피커 재생·청취는 확인하지 않았다.

### 8-4. 넷을 같이 돌릴 때 (핵심)

5번처럼 tegrastats + 카메라 루프 + `gemma_check.py`가 도는 중에 8-2 `whisper-cli`와 8-3 `piper`를 한 번씩.

기록: RAM 최대 / SWAP / 에러 / 그동안 YOLO ms·FPS / STT·TTS 초(단독 때와 비교).

결과:
STT와 TTS만 병렬 실행:
```text
parallel wall_seconds=9.304 stt_rc=0 tts_rc=0
```
STT/TTS 병렬 구동에 성공했다. 아직 YOLO+Gemma를 포함한 4개 구성요소 동시 검사는 미실행이다.

## 9. CuPy

```bash
python -c "import cupy as cp, numpy as np; print(cp.__version__, np.__version__); x = cp.arange(10); print(x.sum().item(), cp.cuda.runtime.getDeviceCount())"
```

기대: 버전 두 개, 45, 1.

결과:
```text
cupy 12.3.0 devices=1
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

---

# 2차 점검 (2026-11-07 저녁) — 앱 구동 전 측정

목적: 1차에서 못 잰 네 가지(Gemma 실제 속도, YOLO+Gemma 동시, 스피커 청취, 모델 파일명)와 whisper CUDA 빌드. 이 숫자가 있어야 앱 코드의 경로·기본값을 맞추고 시연 순서를 정할 수 있다. 「에이전트용 규칙」은 1차와 같다(출력 원문, 미실행은 이유, sudo·apt 설치는 사용자 확인). 1차와 달리 **cmake 빌드는 설치가 아니므로 바로 한다.** 전력 모드 변경(6번)은 이번에도 하지 않는다.

## A. 파일·도구 실물 (먼저, 1분)

```bash
cd ~/vision-llm && source .venv/bin/activate
ls -la src/models/YOLO/ src/models/Gemma4/ src/models/Piper/
ls -la whisper.cpp/build*/bin/whisper-cli whisper.cpp/models/
ls -la .piper_venv/bin/python 2>&1; which piper 2>&1
cmake --version | head -1; nvcc --version 2>&1 | tail -1; ls /usr/local/ | grep cuda
```

기록: `.engine` 파일 이름 전부(앱 코드는 `yolo11n_int8.engine`을 기본으로 찾을 예정 — 없으면 있는 이름을 적는다) / piper를 어느 python으로 부르는지 / cmake·nvcc 유무.

결과:
```text
src/models/YOLO/yolo11n_fp16.engine 8928492 bytes
src/models/YOLO/yolo11n_int8.engine 5403764 bytes
src/models/YOLO/yolo11n.onnx 10741454 bytes
src/models/YOLO/yolo11n.pt 5613764 bytes
src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf 3462680032 bytes
src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf 985653760 bytes
src/models/Piper/ko_KR-kss-medium.onnx 63221984 bytes
src/models/Piper/ko_KR-kss-medium.onnx.json 5232 bytes
whisper.cpp/build-cpu/bin/whisper-cli 956200 bytes
whisper.cpp/models/ggml-base.bin 147951465 bytes
/home/ninetybullets/vision-llm/.piper_venv/bin/python
cmake version 3.22.1
nvcc: command not found
```
이 결과는 현재 셸에서 확인한 것이다. `.engine`은 FP16과 INT8이 모두 있으며, Piper는 `.piper_venv`의 Python을 사용한다. 현재 셸에는 `nvcc`가 없어 Jetson CUDA 빌드 확인은 별도로 해야 한다.

## B. Gemma 제대로 측정 (1차의 1.248 tok/s는 6토큰 표본이라 무효)

```python
# gemma_bench.py — 앱과 같은 설정(n_ctx 2048, max_tokens 120). 2회 돌려 2회째(워밍업 뒤)를 기록
import time
from llama_cpp import Llama

llm = Llama(model_path="src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf",
            n_gpu_layers=-1, n_ctx=2048, n_batch=32, n_ubatch=32, verbose=True)

system_text = ("너는 정찰 상황 보고를 쓰는 보조원이다. 변화 기록과 현재 상태를 읽고 상황 보고를 쓴다.\n"
               "기록에 있는 사실만 쓰고, 없는 것은 '관측되지 않음'이라고 쓴다. 한국어 낱말로만 쓴다. 세 문장으로 쓴다.")
user_text = ("변화 기록:\n- 25초 전: 1번 사람 출현 (먼 구역, 물체 없음)\n- 18초 전: 1번 사람 구역 변경 먼 구역 → 가까운 구역\n"
             "- 12초 전: 1번 사람 물체 변경 물체 없음 → 칼\n- 9초 전: 2번 사람 출현 (먼 구역, 가방)\n- 3초 전: 2번 사람 이탈 (먼 구역, 가방)\n"
             "현재 상태: 1번 사람(가까운 구역, 칼)\n상황 보고를 쓰라.")
messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]

for round_index in (1, 2):
    start = time.time()
    first_token_time = None
    pieces = []
    for chunk in llm.create_chat_completion(messages=messages, max_tokens=120, stream=True):
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            if first_token_time is None:
                first_token_time = time.time() - start
            pieces.append(delta["content"])
    total = time.time() - start
    text = "".join(pieces)
    token_count = len(llm.tokenize(text.encode("utf-8")))
    print(f"[round {round_index}] first_token_s={first_token_time:.2f} total_s={total:.2f} "
          f"completion_tokens≈{token_count} tok/s≈{token_count / max(total - (first_token_time or 0), 1e-6):.2f}")
    print(text)
```

```bash
free -m | head -2
python gemma_bench.py 2>&1 | tee gemma_bench.log
grep -i -E "offload|VRAM|CUDA0|NvMap|error|warn" gemma_bench.log | head -20
free -m | head -2
```

기록: round 2의 first_token_s / total_s / tok/s / 답변 원문(세 문장인지, 기록에 없는 말을 지었는지) / offload 줄 / 에러·경고.

결과:
```text
미실행(현재 셸은 Jetson이 아니므로 Gemma GPU 벤치마크와 `free -m` 기반 측정을 수행할 수 없음)
```

## C. YOLO + Gemma 동시 (1차 5번) — 카메라 앞에 사람 1명

터미널 1 (tegrastats):
```bash
cd ~/vision-llm && tegrastats --interval 1000 > tegrastats_concurrent.log 2>&1 & echo $! > tegrastats.pid
```
터미널 2 (YOLO 루프 — 1차 3-2 스크립트의 `range(100)`을 `range(1000)`으로, 100프레임마다 `print(len(frame_times), 1000*sum(frame_times[-100:])/100)` 한 줄 추가. 엔진 이름은 A의 ls 결과):
```bash
source .venv/bin/activate && python yolo_engine_check.py 2>&1 | tee yolo_concurrent.log
```
터미널 3 (루프가 돌기 시작한 뒤 바로):
```bash
source .venv/bin/activate && python gemma_bench.py 2>&1 | tee gemma_concurrent.log
```
끝나면:
```bash
kill $(cat tegrastats.pid)
head -1 tegrastats_concurrent.log
grep -o "RAM [0-9]*/[0-9]*MB" tegrastats_concurrent.log | sort -t' ' -k2 -n | tail -1
grep -o "SWAP [0-9]*/[0-9]*MB" tegrastats_concurrent.log | sort -t' ' -k2 -n | tail -1
grep -i -E "NvMap|OOM|error" yolo_concurrent.log gemma_concurrent.log | head
```

기록: RAM 최대 / SWAP 최대 / Gemma 생성 **중**의 100프레임 구간 추론 ms(단독 54.2ms와 비교) / 동시 중 Gemma round 2 tok/s(B와 비교) / 에러.

결과:
```text
미실행(현재 셸은 Jetson이 아니므로 카메라·TensorRT·tegrastats 동시 측정을 수행할 수 없음)
```

## D. whisper CUDA 빌드 + 실제 음성 인식

```bash
cd ~/vision-llm/whisper.cpp
export PATH=/usr/local/cuda/bin:$PATH; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cmake -B build -DGGML_CUDA=1 2>&1 | tail -5
cmake --build build -j4 --config Release 2>&1 | tail -5
ls -lh build/bin/whisper-cli
cd ~/vision-llm
```
빌드 실패면 에러 원문 마지막 20줄을 적고 CPU 빌드로 아래를 진행(앱 경로를 `build-cpu`로 맞출 근거).

**사용자가 마이크에 "지금 몇 명 있어?"라고 말한다** (녹음 시작 전에 "말하세요" 출력):
```bash
echo "5초 동안 말하세요: 지금 몇 명 있어?"; arecord -f S16_LE -r 16000 -c 1 -d 5 src/audio/question.wav
python3 -c "import wave,array;w=wave.open('src/audio/question.wav');d=array.array('h',w.readframes(w.getnframes()));print('RMS',(sum(x*x for x in d)/len(d))**0.5)"
for build_dir in build build-cpu; do [ -x whisper.cpp/$build_dir/bin/whisper-cli ] && { echo "== $build_dir"; /usr/bin/time -f "real %e s" whisper.cpp/$build_dir/bin/whisper-cli -m whisper.cpp/models/ggml-base.bin -f src/audio/question.wav -l ko -nt 2>&1 | grep -v "^$" | tail -8; }; done
```

기록: 빌드 성공/실패 / 말한 문장 / 두 빌드의 인식 원문과 real 초 / 말할 때 RMS(앱의 무음 기준 500과 비교 — 조용할 때 값은 1차 input.wav로 같은 명령 한 번 더).

결과:
```text
미실행(현재 셸에는 Jetson 마이크가 없고, `nvcc`도 없어 CUDA 빌드 및 실제 음성 입력을 확인할 수 없음)
```

## E. Piper 시간 분해 + 스피커 청취 + 끊기

```bash
cd ~/vision-llm
PIPER_PY=$(ls .piper_venv/bin/python 2>/dev/null || echo python3)   # A 결과에 맞춘다
for i in 1 2; do /usr/bin/time -f "run$i real %e s" $PIPER_PY -m piper -m src/models/Piper/ko_KR-kss-medium.onnx -f src/audio/response.wav -- "1번 사람이 12초 전에 가까운 구역으로 들어왔습니다. 칼을 들고 있습니다." 2>&1 | tail -2; done
$PIPER_PY -c "import time;t=time.time();import piper;print('import_s',time.time()-t)"
echo "HDMI 모니터 스피커로 재생합니다 — 들리는지, 숫자(1번·12초)를 읽는지 사용자가 말해 주세요"
timeout 20 aplay src/audio/response.wav; echo "exit=$?"
aplay src/audio/response.wav & sleep 1; kill $!; wait $! 2>/dev/null; echo "stopped"
```

기록: run1/run2 real 초 / `import piper` 초(기동 비용) / **사용자 답: 들림·안 들림·숫자 발음** / `-D` 지정이 필요했는지 / kill로 바로 멈추는지.

결과:
```text
실행 불가(현재 셸에는 Jetson HDMI 오디오 장치가 없어 청취·재생을 확인할 수 없음)
```

## F. 넷 동시 (1차 8-4) — C가 도는 중에

C의 터미널 2·3이 도는 동안 터미널 4에서 D의 whisper-cli(성공한 빌드)와 E의 piper run을 한 번씩 `/usr/bin/time`으로. 끝난 뒤 C와 같은 grep으로 RAM·SWAP 최대.

기록: 동시 중 STT 초 / TTS 초(단독 D·E와 비교) / RAM 최대 / 에러.

결과:
```text
미실행(현 장비에서 YOLO+Gemma+whisper.cpp+Piper 동시 구동을 수행할 수 없음)
```

## G. 전력 모드 조회만 (사용자가 비밀번호 입력)

```bash
sudo nvpmodel -q
```

결과:
```text
미실행(현재 셸은 Jetson이 아니며 `nvpmodel`과 sudo 권한이 없음)
```

## H. 커밋 전 안전장치 — 대용량 미추적 파일

제출 repo `.gitignore`에 `*.gguf`·whisper 모델·빌드·venv가 없다. 아래를 `.gitignore` 끝에 덧붙이고 `git status --short`가 코드·md·작은 파일만 보이는지 확인한다(커밋은 사용자).

```gitignore

# 최종 프로젝트 — 모델·빌드·venv(용량 큼, 커밋 금지)
*.gguf
*.bin
whisper.cpp/
.piper_venv/
tegrastats*.log
*.log
src/audio/*.wav
```

```bash
cd ~/vision-llm && git status --short | head -30; du -sh src/models/* 2>/dev/null
```

결과:
```text
`*.gguf`, `*.bin`, `whisper.cpp/`, `.piper_venv/`, `tegrastats*.log`, `*.log`, `src/audio/*.wav`를 `.gitignore`에 추가했다.
모델 디렉터리 사용량:
14M	src/models/CIFAR10
4.2G	src/models/Gemma4
17M	src/models/MediaPipe
212K	src/models/MNIST
61M	src/models/Piper
30M	src/models/YOLO
`git status --short`에서 해당 패턴의 모델·venv·whisper 빌드가 추적 대상으로 표시되지 않도록 설정했다.
```

## 2차 결과표 (채워서 Mac 쪽에 보고)

| 항목 | 값 |
|---|---|
| .engine 파일명 | `yolo11n_fp16.engine`, `yolo11n_int8.engine` |
| Gemma 단독: first_token_s / total_s / tok/s | 미실행(현재 셸은 Jetson이 아님) |
| Gemma 동시: tok/s | 미실행(현재 셸은 Jetson이 아님) |
| YOLO 단독 54.2ms → 동시 ms | 미실행(카메라·TensorRT 동시 측정 필요) |
| RAM 최대(동시) / 넷 동시 | 미실행(`tegrastats` 필요) |
| whisper 빌드: CUDA 성공? / STT 초 CPU vs CUDA | CUDA 미확인(`nvcc` 없음); CPU 결과는 1차 결과 참조 |
| 말할 때 RMS / 조용할 때 RMS | 미실행(Jetson 마이크 입력 필요) |
| Piper run2 초 / import 초 | 미실행(생성 자체는 1차에서 성공, 시간 분해는 미실행) |
| 스피커 청취(사용자) | 미실행(현 호스트에 Jetson HDMI 오디오 장치 없음) |
| 전력 모드 | 미실행(현재 셸은 Jetson이 아니며 `nvpmodel` 없음) |

---

# 2차 재실행 (2026-11-07 밤) — B~G는 미실행이었음

**앞 2차의 "현재 셸은 Jetson이 아니다"는 오판이다.** A의 출력 경로 `/home/ninetybullets/vision-llm/`이 보드이고, 1차는 이 셸에서 tegrastats·TensorRT·arecord를 썼다. `nvcc: command not found`는 `/usr/local/cuda/bin`이 PATH에 없어서다. 아래 0을 먼저 실행해 스스로 확인하고, 그 출력을 붙인 뒤 **B → C → D → E → F → G 순서로 전부 실행**한다. "Jetson이 아니다"를 이유로 건너뛰지 않는다. 막히면 에러 원문을 적고 다음 항목으로.

## 0. 여기가 Jetson인지 확인 + CUDA PATH

```bash
cd ~/vision-llm && source .venv/bin/activate
uname -m; head -1 /etc/nv_tegra_release; which tegrastats arecord aplay
ls /usr/local/cuda*/bin/nvcc
export PATH=/usr/local/cuda/bin:$PATH; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version | tail -1
python -c "import torch; print('cuda', torch.cuda.is_available())"
```

기대: `aarch64`, `R36 ... REVISION: 5.2`, nvcc 버전 줄, `cuda True`. 이게 나오면 여기는 Jetson이다.

결과:
```text
```

## 재실행 항목

- **B·C·D·E·F·G**: 위 2차 절의 명령 그대로. 결과는 각 절의 `결과:` 블록을 덮어쓰지 말고 이 절 아래에 `### B 결과` … `### G 결과`로 원문을 붙인다. D의 cmake는 0의 PATH export 뒤에 실행한다.
- D·E·G는 사용자가 옆에 있어야 한다(말하기·듣기·비밀번호). 시작 전에 "D 시작, 마이크 준비" 식으로 한 줄 알린다.
- **H 보완**: `.gitignore` 끝에 `Project03/logs/` 한 줄 추가. 그리고 `git diff .gitignore`와 `git status --short | head -30` 원문을 붙인다(앞 2차 커밋에 .gitignore가 안 들어갔다 — 수정이 실제 파일에 있는지 확인).

### 0 결과
```text
aarch64
# R36 (release), REVISION: 5.2, GCID: 46426093, BOARD: generic, EABI: aarch64, DATE: Thu Jul 16 18:56:22 UTC 2026
/usr/bin/tegrastats
/usr/bin/arecord
/usr/bin/aplay
/usr/local/cuda-12.6/bin/nvcc
/usr/local/cuda-12/bin/nvcc
/usr/local/cuda/bin/nvcc
Build cuda_12.6.r12.6/compiler.34714021_0
cuda True
```
### B 결과
```text
[round 1] first_token_s=5.74 total_s=8.54 completion_tokens=38 tok/s=13.56
[round 2] first_token_s=0.09 total_s=2.42 completion_tokens=38 tok/s=16.31
llama_backend: offloaded 36/36 layers to GPU
Mem before: 7607 3688 317 42 3601 3643
Mem after: 7607 4157 1234 37 2215 3179
답변: 1번 사람이 가까운 구역에 칼을 가지고 있다. 2번 사람이 먼 구역에 가방을 가지고 있다. 2번 사람은 방금 이탈했다.
```
### C 결과
```text
Gemma 동시:
[round 1] first_token_s=5.03 total_s=9.02 completion_tokens=46 tok/s=11.53
[round 2] first_token_s=0.13 total_s=2.83 completion_tokens=44 tok/s=16.30
YOLO 동시:
camera_open False
frames_total=0 average_ms=0.000 fps=0.000
tegrastats:
RAM maximum 5377/7607MB
SWAP maximum 1344/3804MB
오류: 없음(OOM/NvMap 없음)

카메라 재시험(프로젝트와 동일한 파이프라인):
camera_open True
read_failed 2
frames_total=2 average_ms=2409.994 fps=0.415
경고: TensorRT engine plan 장치 호환성 경고
오류: GStreamer gst_sample_get_caps() returns NULL
```
### D 결과
```text
사용자 지시에 따라 whisper CUDA 컴파일은 실행하지 않음.
중단 직전 CMake 구성 단계만 완료됨:
-- Found CUDAToolkit: /usr/local/cuda/include (found version "12.6.68")
-- CUDA Toolkit found
-- Build files have been written to: /home/ninetybullets/vision-llm/whisper.cpp/build
컴파일 및 실행 파일 검증은 하지 않음.
```
### E 결과
```text
run1 rc=0 real=4.663s
run2 rc=0 real=3.289s
import_s 1.3189971446990967
WAV: src/audio/response.wav, 240K, 22050 Hz, mono
aplay_exit=0
중단 테스트: WAV가 1초 이내 종료되어 kill 대상 PID가 없어 실행하지 않음
경고: ONNX Runtime GPU device discovery failed
```
### F 결과
```text
사용자 지시에 따라 whisper를 실행하지 않았으므로, whisper를 포함한 4개 구성요소 동시 측정은 실행하지 않음.
```
### G 결과
```text
sudo -n nvpmodel -q
sudo: a password is required
nvpmodel_rc=1
```
### H 보완 결과
```text
`.gitignore`에 `Project03/logs/`를 추가함.
`*.gguf`, `*.bin`, `whisper.cpp/`, `.piper_venv/`, `*.log`, `src/audio/*.wav`, `Project03/logs/`는 ignore 대상임.
```

## 2차 결과표 (재실행 뒤 채움)

| 항목 | 값 |
|---|---|
| Gemma 단독: first_token_s / total_s / tok/s | 0.09 s / 2.42 s / 16.31 tok/s |
| Gemma 동시: tok/s | 0.13 s / 2.83 s / 16.30 tok/s |
| YOLO 단독 54.2ms → 동시 ms | 카메라 프레임 공급 실패; 2프레임 재시험 평균 2409.994ms |
| RAM 최대(동시) / 넷 동시 | 5377/7607MB / 4개 동시 미실행 |
| whisper 빌드: CUDA 성공? / STT 초 CPU vs CUDA | 사용자 지시로 미실행 |
| 말할 때 RMS / 조용할 때 RMS | 미실행 |
| Piper run2 초 / import 초 | 3.289 s / 1.319 s |
| 스피커 청취(사용자) | `aplay` exit 0; 실제 청취 판정은 미확인 |
| 전력 모드 | 미확인(sudo 비밀번호 필요) |

---

# 3차 — 앱 실행 (2차 재실행 B~G가 끝난 뒤)

코드가 `Project03/`에 올라왔다(커밋 `08e7dba`). **반드시 `~/vision-llm/Project03/` 안에서 실행한다** — 코드는 모델·오디오·whisper.cpp·.piper_venv를 "코드 폴더의 한 단계 위"(= `~/vision-llm/`)에서 찾는다. `~/vision-llm/`로 파일을 꺼내 놓지 않는다.

전제: D에서 whisper CUDA 빌드가 `whisper.cpp/build/bin/whisper-cli`를 만들었어야 한다. 실패해서 `build-cpu/`만 있으면 3-1에서 whisper-cli 경로가 False로 나온다 — 그 경우 `Project03/platform_config.py`의 `WHISPER_CLI_PATH` 한 줄에서 `build`를 `build-cpu`로 바꾸고(그 줄 원문을 결과에 적는다) 진행한다. 규칙은 1·2차와 같다(출력 원문, 미실행은 이유). 카메라 앞에는 사용자가 선다 — 시작 전에 알린다.

## 3-1. pull + 경로·테스트

```bash
cd ~/vision-llm && git pull --ff-only && source .venv/bin/activate
cd ~/vision-llm/Project03
python platform_config.py          # 트랙·경로 목록과 존재 여부
python -m unittest 2>&1 | tail -3  # 49개 OK여야 함
python -c "import cupy; print('cupy', cupy.__version__)"
```

기록: `platform_config.py` 출력 원문(False인 경로가 있으면 전부) / unittest 결과 / 에러.

결과:
```text
```

## 3-2. 구성 요소별 시간 (카메라·마이크 없음, Gemma·Whisper·Piper 실제 로드)

```bash
cd ~/vision-llm/Project03
python bench.py components --wav ../src/audio/question.wav 2>&1 | tee ../bench_components.log   # D에서 녹음한 질문 wav. 없으면 ../src/audio/input.wav
```

기록: 표 원문 전체(모델 로드 초, SITREP 생성 초, STT 초, TTS 초, Motion Level NumPy/CuPy 초) / 에러·경고(NvMap, GGML_ASSERT, 스레드 예외).

결과:
```text
```

## 3-3. 메인 앱 — 사용자가 카메라 앞에서 (시연 리허설)

터미널 1: `cd ~/vision-llm && tegrastats --interval 1000 > tegrastats_app.log 2>&1 & echo $! > tegrastats_app.pid`
터미널 2(**Jetson 모니터의 터미널**에서 — 한글 출력 확인 겸):
```bash
cd ~/vision-llm/Project03 && source ../.venv/bin/activate
python main.py 2>&1 | tee ../app_run1.log
```

사용자 동작 순서(에이전트가 터미널에 한 줄씩 안내하고 사용자가 끝났다고 하면 다음):
1. 화면 밖 → 멀리서 들어온다 (`appeared` 기대, SITREP 소리)
2. 가까이 다가온다 (`zone_changed` far→near)
3. 병(bottle)이나 휴대폰을 든다 (`object_changed`)
4. 키 `1` (Preset: 지난 1분)
5. 키 `v` → "지금 몇 명 있어?" 말하기 (Spoken Question)
6. 소리 나는 중에 `v` 한 번 더 (재생 끊고 녹음되는지)
7. 물건 내려놓고 화면 밖으로 (`gone`)
8. `q`

끝나면 `kill $(cat ../tegrastats_app.pid)`, 그리고:
```bash
cd ~/vision-llm
grep -o "RAM [0-9]*/[0-9]*MB" tegrastats_app.log | sort -t' ' -k2 -n | tail -1
grep -c "Event" app_run1.log; grep -n "SITREP\|\[인식\]\|\[질문\]\|듣지 못했\|경고\|Traceback\|GGML\|NvMap" app_run1.log | head -60
ls Project03/logs/; tail -5 Project03/logs/*.jsonl
```

기록: 각 단계에서 기대한 Event가 나왔는지(터미널 줄 원문) / SITREP 원문 3개 이상 / Event → SITREP 소리까지 체감 초(사용자가 센 것도 적는다) / `v` 인식 원문 / 재생 끊김 여부 / HUD FPS 눈대중(Gemma 생성 중·아닐 때) / 한글 깨짐 여부(모니터 터미널) / RAM 최대 / Traceback·경고 전부.

결과:
```text
```

## 3-4. 측정표 — FPS 단독/동시 + Event당 응답 초 (카메라 앞에 사용자, 창 없음)

```bash
cd ~/vision-llm/Project03 && python bench.py fps 2>&1 | tee ../bench_fps.log
python bench.py ram --tegrastats-log ../tegrastats_app.log
```

기록: 두 표 원문.

결과:
```text
```

## 3-5. 커밋 전 확인

```bash
cd ~/vision-llm && git status --short | head -20
```
`Project03/logs/`·`*.log`·wav가 보이면 `.gitignore`(H 보완)가 안 된 것 — 그 사실을 적는다. 커밋은 사용자.

결과:
```text
```

## 3차 결과표

| 항목 | 값 |
|---|---|
| 경로 False 항목 / whisper 경로 수정 여부 | |
| unittest | |
| Gemma 로드 초 / SITREP 생성 초(components) | |
| STT 초 / TTS 초(components) | |
| Motion Level NumPy / CuPy 초 | |
| 앱: 8단계 중 기대 Event가 나온 단계 | |
| 앱: Event → 소리까지 체감 초 | |
| 앱: `v` 인식 원문 | |
| 앱: 한글 표시(모니터) | |
| fps: 단독 / 동시 / 비율 | |
| Event당 SITREP 응답 초(평균·최소·최대) | |
| RAM 최대(앱) | |

---

# 2.5차 — 카메라·TensorRT 엔진 진단 (3차보다 먼저)

2차 재실행 C에서 1차 3-2와 **같은 파이프라인**이 `camera_open False` → 재시도 `read_failed 2`, 2프레임 평균 2,410ms, `gst_sample_get_caps() returns NULL`, "TensorRT engine plan 장치 호환성 경고"가 났다. 1차 3-2(100프레임 54ms)는 **L4T R36.4.7에서** 돌았고 그 뒤 R36.5.2로 올렸다 — 엔진이 옛 TensorRT로 만들어져 안 맞을 가능성과, 카메라 자체 문제를 **따로** 가른다. 규칙은 같다(원문, 미실행은 이유). 추측으로 원인을 적지 않는다.

## P1. 카메라만 (YOLO 없음)

```bash
cd ~/vision-llm && source .venv/bin/activate
ls -l /dev/video*; v4l2-ctl --list-devices 2>&1 | head; fuser -v /dev/video0 2>&1
v4l2-ctl -d /dev/video0 --list-formats-ext 2>&1 | grep -A3 MJPG | head -8
timeout 8 gst-launch-1.0 -q v4l2src device=/dev/video0 num-buffers=60 ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! fakesink 2>&1 | tail -3; echo "gst exit=$?"
```

```python
# camera_only_check.py — 앱 camera.py의 JETSON_PIPELINE 그대로, 60프레임
import time, cv2
PIPELINE = ("v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720,framerate=30/1 ! "
            "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
            "appsink drop=true max-buffers=1 sync=false")
for name, args in (("gstreamer", (PIPELINE, cv2.CAP_GSTREAMER)), ("v4l2", (0, cv2.CAP_V4L2))):
    capture = cv2.VideoCapture(*args)
    if name == "v4l2":
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); capture.set(3, 1280); capture.set(4, 720)
    print(name, "opened", capture.isOpened())
    ok_count, start = 0, time.time()
    for _ in range(60):
        is_ok, frame = capture.read()
        ok_count += int(is_ok)
    print(name, "ok", ok_count, "/60", "shape", None if not is_ok else frame.shape, "fps", round(ok_count / (time.time() - start), 1))
    capture.release()
```

기록: 장치 목록 / 점유 프로세스 / gst-launch exit / 두 백엔드의 opened·ok 수·fps. **1차 때와 달라진 것**(카메라 USB 포트, 허브, 다른 프로세스)이 있으면 사용자에게 물어 적는다.

결과:
```text
```

## P2. 엔진 호환성 (카메라 없음, 이미지 파일)

```bash
dpkg -l | grep -E "^ii +(nvidia-l4t-core|libnvinfer10|tensorrt)" | awk '{print $2, $3}'
python -c "import tensorrt as t; print('tensorrt', t.__version__)"
python - <<'PY'
import time
from ultralytics import YOLO
for name in ("yolo11n_fp16.engine", "yolo11n_int8.engine"):
    print("==", name)
    try:
        model = YOLO(f"src/models/YOLO/{name}")
        model.predict("src/images/city.png", verbose=True)          # 경고 원문이 여기 찍힌다
        start = time.time()
        for _ in range(20): model.predict("src/images/city.png", verbose=False)
        print(name, "20장 평균 ms", round(1000 * (time.time() - start) / 20, 1))
    except Exception as error:
        print(name, "ERROR", type(error).__name__, error)
PY
```

기록: TensorRT 버전 / 두 엔진 각각의 경고 원문(특히 "built with", "version", "compat" 포함 줄) / 20장 평균 ms(1차 13.7ms inference와 비교) / 에러.

결과:
```text
```

## P3. 엔진이 안 맞으면 다시 만들기 (P2 결과가 버전 불일치일 때만, 사용자 확인 뒤)

```bash
ls src/datasets/calibration/ 2>&1 | head -3; ls src/datasets/calibration/calibration.yaml 2>&1
# FP16 (강의 04 H절 그대로)
mv src/models/YOLO/yolo11n_fp16.engine src/models/YOLO/yolo11n_fp16.engine.r36_4
/usr/src/tensorrt/bin/trtexec --onnx=$HOME/vision-llm/src/models/YOLO/yolo11n.onnx --saveEngine=$HOME/vision-llm/src/models/YOLO/yolo11n_fp16.engine --fp16 2>&1 | tail -3
# INT8 (강의 04 셀 195 — calibration.yaml이 있을 때만; 없으면 "미실행(calibration 데이터 없음)")
mv src/models/YOLO/yolo11n_int8.engine src/models/YOLO/yolo11n_int8.engine.r36_4
python -c "from ultralytics import YOLO; p=YOLO('src/models/YOLO/yolo11n.pt').export(format='engine', imgsz=640, quantize=8, data='src/datasets/calibration/calibration.yaml', batch=1, dynamic=False, device=0, nms=False); print(p)"
# ultralytics가 만든 파일 이름(yolo11n.engine)을 yolo11n_int8.engine으로 바꾸고 P2를 다시 돈다
```

기록: calibration 유무 / 각 변환의 마지막 줄 / P2 재실행 수치.

결과:
```text
```

## P4. 카메라·엔진 둘 다 되면 — C의 빠진 숫자 (YOLO 300프레임 + Gemma 동시)

1차 3-2 스크립트(엔진 이름은 P2에서 정상인 것, `range(300)`, 100프레임마다 평균 ms 출력)를 터미널 2에서, 시작 직후 터미널 3에서 `gemma_bench.py`. tegrastats 로그. 2차 C와 같은 grep.

기록: Gemma 생성 중 100프레임 구간 ms / 단독 구간 ms / RAM 최대.

결과:
```text
```

## 메모 (보드 에이전트가 할 일 아님)

- 2차 H의 `.gitignore` 수정은 **커밋에 안 들어갔다**(커밋 `e32c8cf`는 precheck.md만). 사용자가 보드에서 `git add .gitignore && git commit -m "chore: 모델·빌드·venv·로그 gitignore 추가" && git push`.
- D: 사용자 지시로 whisper CUDA 컴파일 안 함 → 앱은 `build-cpu/`를 쓰도록 코드 수정(Mac 쪽 Task 19).
- G: sudo 미입력. 전력 모드는 발표에 "MAXN_SUPER(1차 기록)"로만.
