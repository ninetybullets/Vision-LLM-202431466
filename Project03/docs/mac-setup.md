---
title: 최종 프로젝트 — Mac 트랙 설치
brand: 온디바이스 AI
nav:
  - 후보 비교 | topic-candidates.html
  - 설계 초안 | design-draft.html
  - 보드 체크 | board-check.html
  - Mac 설치 | mac-setup.html
  - 조사 보고서 | research-report.html
  - 선행 사례 | landscape-sweep.html
---

# Mac 트랙 설치 — 실험용 환경 만들기

작성 2026-11-06. Mac 트랙은 TensorRT와 CuPy만 빼고 전체 흐름을 돌리는 실험용이다(`design-draft.md` 4절 「두 트랙」). 제출용 숫자는 Jetson에서만 잰다.

<div class="warn"><b>실행 상태 (2026-11-06, Apple M2 · macOS 26.6.2)</b> — 1~5절 명령을 이 Mac에서 실제로 돌렸다. 확인됨: venv, YOLO 추론, llama-cpp-python Metal 빌드와 Gemma 4 E2B 한국어 생성, whisper.cpp 빌드(Metal), Piper 합성, Piper wav → Whisper 왕복 인식. <b>안 해 본 것</b>: 웹캠, 마이크 녹음(4절 ffmpeg avfoundation), 스피커 재생(5절 afplay) — 권한 창과 사람이 필요해서 남겨 뒀다. 고친 명령은 2절(가중치·이미지를 <code>src/</code> 아래로)뿐이다.</div>

## 0. 지금 Mac 상태 (2026-11-06 확인)

| 항목 | 상태 |
|---|---|
| `brew`, `uv`, `ffmpeg`, `afplay` | 있음 |
| `cmake`, `whisper-cli`, `piper` | 없음 |
| 기존 `.venv` | Python 3.10.20, numpy 1.21.5, cv2 4.7 (건드리지 않는다) |
| `ultralytics`, `llama_cpp` | 없음 |
| 마이크 | ffmpeg 장치 목록에서 오디오 `[0] MacBook Air 마이크` |
| 디스크 여유 | 77GB |

## 1. 폴더와 venv

Jetson의 `~/vision-llm`과 같은 상대 경로를 쓴다. 그래야 트랙마다 갈리는 값이 줄어든다.

```text
final_project/             ← Jetson의 ~/vision-llm 자리
  code/                    Mac에서의 작업 폴더 (코드는 여기서 실행)
  code/.venv/              Mac 트랙 전용 venv
  .piper_venv/             Piper 전용 venv (Jetson과 같은 이유로 분리)
  whisper.cpp/             whisper.cpp 소스와 빌드
  src/models/YOLO/         yolo11n.pt
  src/models/Gemma4/       GGUF
  src/models/Piper/        한국어 음성
  src/audio/               input.wav, response.wav
```

`.venv/`, `.piper_venv/`, `whisper.cpp/`, `src`는 `.gitignore`에 들어 있어 커밋되지 않는다.

```bash
cd ~/Desktop/26-2/OnDevice_AI
mkdir -p final_project/src/audio
cd final_project/code
uv venv --python 3.10 --seed .venv
source .venv/bin/activate
python --version
```

확인: `Python 3.10.x`. Jetson도 3.10이라 버전을 맞춘다.

## 2. YOLO (`.pt`)

```bash
pip install ultralytics lap
mkdir -p ../src/models/YOLO
curl -sSL -o ../src/bus.jpg https://ultralytics.com/images/bus.jpg
python -c "
from ultralytics import YOLO
model = YOLO('../src/models/YOLO/yolo11n.pt')
result = model.predict('../src/bus.jpg', verbose=True)
"
```

확인: 첫 실행에서 `yolo11n.pt`(5.4MB)를 `../src/models/YOLO/`에 내려받고, 출력에 `4 persons, 1 bus` 검출 줄과 inference ms가 나온다. `ultralytics`는 numpy·opencv를 새 버전으로 깔기 때문에 전용 venv를 쓰는 것이다.

가중치와 이미지를 `../src/` 아래에 두는 이유: 처음 문서처럼 `YOLO('yolo11n.pt')`와 URL을 쓰면 `yolo11n.pt`·`bus.jpg`가 `code/` 바로 아래에 생기는데, 거기는 `.gitignore`에 안 걸린다(`src`만 걸림).

Mac에서 ultralytics 기본 장치는 CPU다. `model.predict(..., device='mps')`로 Apple GPU를 쓸 수 있다(실측은 6절).

`lap`은 추적(`model.track`)에 필요한 패키지다. 2026-11-06에 `detector.py`를 처음 돌릴 때 ultralytics가 `lap>=0.5.12`가 없다며 이 venv에 `lap==0.5.14`를 스스로 설치했다. 미리 깔아 두면 첫 실행 때 인터넷이 필요 없다.

## 3. Gemma (llama-cpp-python, Metal)

```bash
CMAKE_ARGS="-DGGML_METAL=on" \
  pip install --no-cache-dir llama-cpp-python
pip install huggingface_hub python-dotenv
python -c "
import llama_cpp
print(llama_cpp.__version__,
      llama_cpp.llama_supports_gpu_offload())
"
```

확인: 버전과 `True`. 소스 빌드라 몇 분 걸리고 Xcode 명령줄 도구가 필요하다(없으면 `xcode-select --install`). 빌드가 실패하면 README의 미리 빌드된 휠을 쓴다:

```bash
pip install llama-cpp-python --extra-index-url \
  https://abetlen.github.io/llama-cpp-python/whl/metal
```

모델 받기(05 노트북 셀 178과 같은 저장소·파일명):

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='bartowski/google_gemma-4-E2B-it-GGUF',
    filename='google_gemma-4-E2B-it-Q4_K_M.gguf',
    local_dir='../src/models/Gemma4')
"
ls -lh ../src/models/Gemma4/
```

확인: `.gguf` 파일 크기가 찍힌다(3.2G). 그다음 `board-check.md` 4번의 `gemma_check.py`를 이 폴더에서 그대로 돌려 한국어 답변이 나오는지 본다.

확인됨(2026-11-06): 소스 빌드(`CMAKE_ARGS="-DGGML_METAL=on"`)가 성공해 미리 빌드된 휠은 쓰지 않았다. Hugging Face 토큰 없이 받아진다(`unauthenticated requests` 경고만 뜸). llama-cpp-python 0.3.36이 `arch = gemma4`를 읽고 `offloaded 36/36 layers to GPU`를 찍는다.

## 4. Whisper (whisper.cpp)

```bash
brew install cmake
git clone https://github.com/ggml-org/whisper.cpp.git ../whisper.cpp
cd ../whisper.cpp
cmake -B build
cmake --build build -j --config Release
sh ./models/download-ggml-model.sh base
cd ../code
```

확인: `ls -lh ../whisper.cpp/build/bin/whisper-cli ../whisper.cpp/models/ggml-base.bin` 두 파일이 있다. Apple Silicon에서는 별도 옵션 없이 GPU(Metal)로 돈다(whisper.cpp README).

5초 녹음 후 인식. `:0`은 0절의 오디오 장치 번호다. 처음 실행하면 터미널에 마이크 권한을 묻는다.

```bash
ffmpeg -y -f avfoundation -i ":0" -t 5 \
  -ar 16000 -ac 1 -c:a pcm_s16le ../src/audio/input.wav
time ../whisper.cpp/build/bin/whisper-cli \
  -m ../whisper.cpp/models/ggml-base.bin \
  -f ../src/audio/input.wav -l ko -nt
```

확인: 말한 문장이 글로 나온다. Jetson의 `arecord` 옵션(`-f S16_LE -r 16000 -c 1 -d 5`)과 같은 뜻이다: 16비트, 16kHz, mono, 5초.

이 녹음 단계는 아직 안 돌렸다(마이크 권한 창). 대신 마이크 없이 경로만 확인하는 방법: 5절의 Piper wav(22050Hz)를 16kHz mono로 바꿔 넣는다.

```bash
ffmpeg -y -i ../src/audio/response.wav \
  -ar 16000 -ac 1 -c:a pcm_s16le ../src/audio/response_16k.wav
time ../whisper.cpp/build/bin/whisper-cli \
  -m ../whisper.cpp/models/ggml-base.bin \
  -f ../src/audio/response_16k.wav -l ko -nt
```

첫 실행은 Metal 초기화로 8초쯤 걸리고, 두 번째부터 0.3초대다(6절).

## 5. Piper

```bash
deactivate
uv venv --python 3.10 --seed ../.piper_venv
source ../.piper_venv/bin/activate
pip install piper-tts
mkdir -p ../src/models/Piper
python -m piper.download_voices \
  ko_KR-kss-medium --data-dir ../src/models/Piper
time python -m piper \
  -m ../src/models/Piper/ko_KR-kss-medium.onnx \
  -f ../src/audio/response.wav \
  -- "표적 한 명이 가까운 구역으로 들어왔습니다."
afplay ../src/audio/response.wav
deactivate
source .venv/bin/activate
```

확인: 소리가 나고 알아들을 만하다. 영어 낱말도 한 번 넣어 본다(`-- "표적 1번이 near 구역에서 knife를 들었습니다."`). 영어·숫자를 어떻게 읽는지가 설계 4절 「한국어 낱말」 규칙의 근거가 된다.

2026-11-06에는 `afplay` 없이 합성만 했다: `src/audio/response.wav`(1ch 22050Hz Int16, 3.07초), `src/audio/response_mixed.wav`(영어 섞인 문장, 3.31초). 사용자가 들은 결과(2026-11-06): `response.wav`는 「표적 한 명이 가까운 구역으로 들어왔습니다.」로 또렷이 들림. `response_mixed.wav`(영어·숫자 섞인 문장)는 무슨 말인지 알아들을 수 없음 → 설계 4절 「한국어 낱말」 규칙의 근거. 마이크 권한 허용함, 웹캠은 1920x1080 프레임이 들어오는 것 확인(OpenCV 장치 0·1 둘 다).

## 6. 끝나면 적어 둘 것

2026-11-06, Apple M2 기준.

| 항목 | 결과 |
|---|---|
| 설치 버전 | Python 3.10.20 · ultralytics 8.4.203 · torch 2.14.1 · numpy 1.26.4 · opencv 4.11.0 · llama-cpp-python 0.3.36 · whisper.cpp 1.9.5-dev(`b1535f5f`) · piper-tts 1.8.0 · onnxruntime 1.23.2(Piper venv의 numpy 2.2.6) · cmake 4.4.5 |
| YOLO inference ms (`yolo11n.pt`, bus.jpg 640x480, `4 persons, 1 bus`) | 첫 실행 178.5ms(CPU). 같은 프로세스 재실행 CPU 40.3 / 38.2ms, `device='mps'` 6.6 / 4.9ms(mps 첫 회는 preprocess 357ms) |
| llama-cpp-python 버전 / GPU offload | 0.3.36 / `True`, `offloaded 36/36 layers to GPU` |
| Gemma 한국어 답변 원문 / tok/s | 로드 3.54초. 답변: 「다음은 관측을 두 문장으로 요약한 한국어 상황 보고입니다. **두 명의 사람이 있었으며, 한 명은 칼을 소지하고 있었습니다. 다른 한 명은 화면에서 이탈하는 행동을 보였습니다.**」 / prompt 45 · completion 51 토큰, 2.09초, 24.4 tok/s |
| Whisper: 말한 문장 / 인식된 문장 / real 초 | 마이크 대신 Piper wav 왕복. 「표적 한 명이 가까운 구역으로 들어왔습니다.」→「표속한 면이 가까운 구역으로 들어왔습니다.」 real 8.06초(첫 회), 0.33초(재실행). 「표적 1번이 near 구역에서 knife를 들었습니다.」→「표적 일본인을 구역에서 내리를 들었습니다.」 real 0.37초 |
| Piper: real 초 / 영어 낱말을 어떻게 읽었나 | 첫 회 4.06초, 재실행 0.47초(한국어), 0.47초(영어 섞인 문장). 귀로는 아직 안 들었다. Whisper 받아쓰기로 보면 「1번이 near」가 「일본인을」, 「knife를」이 「내리를」로 바뀌어, 영어 낱말과 숫자가 제대로 읽히지 않았을 가능성이 높다 |

이 숫자는 Mac 것이라 발표에는 쓰지 않는다. 흐름이 도는지 확인하는 용도다.
