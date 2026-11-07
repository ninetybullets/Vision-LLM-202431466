"""트랙(Jetson/Mac)마다 갈리는 값을 한 곳에 모은 설정 모듈.

실행한 OS로 트랙을 정하고(Linux → jetson, macOS → mac), 검출 모델 경로·녹음/재생 명령·배열 모듈을 내준다.
다른 모듈은 OS를 직접 보지 않고 이 파일의 상수·함수만 쓴다.
확인 실행: .venv/bin/python platform_config.py (트랙·경로·명령을 출력만 하고 장치는 열지 않는다)
"""
import platform
from pathlib import Path

# 이 파일이 있는 폴더(코드 폴더). 제출 repo는 Project03/, Mac은 final_project/code/
CODE_DIRECTORY = Path(__file__).resolve().parent
# 모델·오디오·외부 빌드(src/, whisper.cpp/, .piper_venv/)는 코드 폴더 한 단계 위 repo 루트에 둔다.
# Jetson ~/vision-llm/의 실물 배치와 같다(1차 점검으로 확인), Mac은 final_project/
REPO_ROOT = CODE_DIRECTORY.parent

# 녹음 형식: Whisper 입력에 맞춘 16kHz · mono · 16비트, 키를 누르면 5초 녹음(스펙 「Spoken Question」)
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
CHANNEL_COUNT = 1
# 16비트 little-endian PCM을 뜻하는 형식 이름. 프로그램마다 표기가 다르다(ffmpeg: mac-setup.md 4절, arecord: board-check.md 8-2)
MAC_SAMPLE_FORMAT = "pcm_s16le"
JETSON_SAMPLE_FORMAT = "S16_LE"

# Jetson 마이크 장치(예: "plughw:2,0"). board-check.md 8-2의 arecord -l 결과로 정한다. None이면 기본 장치
JETSON_RECORD_DEVICE = None  # 임시값, 실측 후 교체
# PulseAudio가 마이크를 잡아 "Device or resource busy"가 나면 True로 바꿔 arecord 앞에 pasuspender --를 붙인다(board-check.md 문제 해결)
SHOULD_USE_PASUSPENDER = False  # 임시값, 실측 후 교체


def detect_track():
    """실행 중인 OS로 트랙 이름을 정한다.

    입력: 없음(platform.system()을 본다)
    출력: "jetson"(Linux) 또는 "mac"(macOS). 그 밖의 OS는 RuntimeError
    """
    system_name = platform.system()
    if system_name == "Linux":
        return "jetson"
    if system_name == "Darwin":
        return "mac"
    raise RuntimeError(f"지원하지 않는 OS: {system_name} (Linux=Jetson 트랙, macOS=Mac 트랙만 지원)")


TRACK = detect_track()
# Project01 camera.py의 백엔드 이름과 같다("mac" | "jetson")
CAMERA_BACKEND = TRACK

# 두 트랙이 같은 상대 경로를 쓰는 파일
GEMMA_MODEL_PATH = REPO_ROOT / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
# Jetson의 whisper.cpp/build/는 CUDA 빌드, Mac은 Metal 빌드(board-check.md 8-2, mac-setup.md)
WHISPER_CLI_PATH = REPO_ROOT / "whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL_PATH = REPO_ROOT / "whisper.cpp/models/ggml-base.bin"
# Piper는 의존성 충돌을 피하려고 전용 venv의 python으로 실행한다
PIPER_PYTHON_PATH = REPO_ROOT / ".piper_venv/bin/python"
PIPER_VOICE_PATH = REPO_ROOT / "src/models/Piper/ko_KR-kss-medium.onnx"
INPUT_WAV_PATH = REPO_ROOT / "src/audio/input.wav"
RESPONSE_WAV_PATH = REPO_ROOT / "src/audio/response.wav"

# 트랙마다 다른 검출 모델: Jetson은 TensorRT INT8 엔진, Mac은 PyTorch 가중치
if TRACK == "jetson":
    # Jetson 경로: 보드 미검증
    # 강의 06 Vision_LLM_test03.py의 엔진 이름. 보드의 실제 파일명은 2차 점검에서 확인 전(2026-11-07)
    YOLO_MODEL_PATH = REPO_ROOT / "src/models/YOLO/yolo11n_int8.engine"
else:
    YOLO_MODEL_PATH = REPO_ROOT / "src/models/YOLO/yolo11n.pt"


def build_record_command(wav_path):
    """현재 트랙의 녹음 명령을 subprocess에 넘길 인자 목록으로 만든다(실행은 하지 않음).

    입력: wav_path — 녹음 결과를 저장할 wav 경로(str 또는 Path)
    출력: list[str] — 예: ["ffmpeg", "-y", ..., "input.wav"]
    """
    if TRACK == "jetson":
        # Jetson 경로: 보드 미검증
        command = ["pasuspender", "--"] if SHOULD_USE_PASUSPENDER else []
        command.append("arecord")
        if JETSON_RECORD_DEVICE is not None:
            command += ["-D", JETSON_RECORD_DEVICE]
        command += ["-f", JETSON_SAMPLE_FORMAT, "-r", str(SAMPLE_RATE), "-c", str(CHANNEL_COUNT),
                    "-d", str(RECORD_SECONDS), str(wav_path)]
        return command
    # ":0"은 avfoundation의 "영상 없음, 오디오 장치 0번(MacBook 마이크)"(mac-setup.md 4절)
    # avfoundation 마이크는 벽시계 1초에 샘플을 약 88%만 보내고 타임스탬프가 건너뛴다(샘플이 빠지는지 장치 클럭 차이인지는 미확인).
    # 그래서 -t를 타임스탬프 기준으로 세면 wav가 약 4.4초로 짧아진다. asetpts=N/SR/TB로 타임스탬프를 실제 샘플 수로 다시 매겨 -t가 소리 5초를 세게 한다
    return ["ffmpeg", "-y", "-f", "avfoundation", "-i", ":0", "-af", "asetpts=N/SR/TB", "-t", str(RECORD_SECONDS),
            "-ar", str(SAMPLE_RATE), "-ac", str(CHANNEL_COUNT), "-c:a", MAC_SAMPLE_FORMAT, str(wav_path)]


def build_play_command(wav_path):
    """현재 트랙의 wav 재생 명령을 subprocess에 넘길 인자 목록으로 만든다(실행은 하지 않음).

    입력: wav_path — 재생할 wav 경로(str 또는 Path)
    출력: list[str] — Jetson은 ["aplay", 경로], Mac은 ["afplay", 경로]
    """
    if TRACK == "jetson":
        # Jetson 경로: 보드 미검증
        return ["aplay", str(wav_path)]
    return ["afplay", str(wav_path)]


def load_array_module():
    """현재 트랙의 배열 모듈(Jetson은 CuPy, Mac은 NumPy)을 불러와 돌려준다.

    입력: 없음
    출력: cupy 또는 numpy 모듈 객체
    Jetson에서 CuPy를 못 불러오면 NumPy로 넘어가지 않고 RuntimeError로 멈춘다 —
    제출용이 CPU로 도는 것을 모른 채 시연하지 않기 위해서다(스펙 User Story 44).
    """
    if TRACK == "jetson":
        # Jetson 경로: 보드 미검증
        try:
            import cupy as cp
        except ImportError as error:
            raise RuntimeError(f"Jetson 트랙인데 CuPy를 불러오지 못했습니다(NumPy로 대체하지 않음): {error}") from error
        return cp
    import numpy as np
    return np


def describe_path(path):
    """경로와 그 파일이 있는지를 한 줄 문자열로 만든다. 입력: Path, 출력: "있음  경로" 형태의 str"""
    existence = "있음" if path.exists() else "없음"
    return f"{existence}  {path}"


def main():
    """현재 트랙의 설정값을 출력한다. 카메라·마이크·스피커는 열지 않고 명령 인자 목록만 보여 준다."""
    print(f"트랙: {TRACK}")
    print(f"카메라 백엔드: {CAMERA_BACKEND}")
    named_paths = {
        "YOLO 모델": YOLO_MODEL_PATH,
        "Gemma 모델": GEMMA_MODEL_PATH,
        "whisper-cli": WHISPER_CLI_PATH,
        "Whisper 모델": WHISPER_MODEL_PATH,
        "Piper python": PIPER_PYTHON_PATH,
        "Piper 음성": PIPER_VOICE_PATH,
        "녹음 wav": INPUT_WAV_PATH,
        "응답 wav": RESPONSE_WAV_PATH,
    }
    for name, path in named_paths.items():
        print(f"{name}: {describe_path(path)}")
    print(f"녹음 명령: {build_record_command(INPUT_WAV_PATH)}")
    print(f"재생 명령: {build_play_command(RESPONSE_WAV_PATH)}")
    print(f"배열 모듈: {load_array_module().__name__}")


if __name__ == "__main__":
    main()
