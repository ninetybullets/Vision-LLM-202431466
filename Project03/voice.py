"""음성 입출력 모듈: 녹음 → Whisper(STT) → 글, 글 → Piper(TTS) → 재생, 그리고 재생 끊기.

외부 프로그램(ffmpeg/arecord, whisper-cli, Piper, afplay/aplay)을 subprocess로 부르는 얇은 층이다(강의 06 C절 방식).
실패해도 예외로 멈추지 않고 경고 한 줄을 낸 뒤 빈 결과로 돌아온다(스펙 「임시 결정」 — 시연 중 멈추지 않게).
수동 확인: .venv/bin/python voice.py <wav 경로> | --say "<글>" | --record
"""
import argparse
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

from platform_config import (
    INPUT_WAV_PATH,
    PIPER_PYTHON_PATH,
    PIPER_VOICE_PATH,
    RESPONSE_WAV_PATH,
    WHISPER_CLI_PATH,
    WHISPER_MODEL_PATH,
    build_play_command,
    build_record_command,
)

# Whisper 인식 언어(한국어 고정, mac-setup.md 4절의 -l ko)
WHISPER_LANGUAGE = "ko"

# whisper.cpp --prompt에 넘기는 예상 어휘 힌트(Preset Question·녹음 질문에 나올 말).
# --prompt는 이 글을 디코더에 앞 문맥처럼 주어 비슷한 낱말이 나오게 유도한다. 효과 미측정 — 3차 점검 인식 원문으로 확인
WHISPER_PROMPT_HINT = "지금 몇 명 있어? 지난 1분 동안 무엇이 바뀌었나? 지금 상황은 어떤가?"

# 무음 판정 음량(RMS) 기준. 이보다 작으면 Whisper를 부르지 않는다.
# 근거: 2026-11-06 Mac 내장 마이크, 조용한 방 5초 녹음 3번의 RMS 236.0·263.6·244.4 → 가장 큰 값의 약 2배.
# 임시값, 사용자 목소리 실측 후 교체(말소리 RMS는 아직 재지 않았다)
SILENCE_RMS_THRESHOLD = 500

# 지금 재생 중인 프로세스. 다른 스레드의 stop_playback()이 이것을 끝낸다(재생 중이 아니면 None)
_playback_process = None


def print_warning(message):
    """터미널에 경고 한 줄을 낸다. 입력: message — 경고 내용(str), 출력: 없음"""
    print(f"[voice 경고] {message}", flush=True)


def run_command(command):
    """외부 프로그램을 끝날 때까지 실행하고 결과를 돌려준다.

    입력: command — subprocess에 넘길 인자 목록(list[str])
    출력: 성공하면 표준 출력 글(str), 프로그램이 없거나 0이 아닌 코드로 끝나면 경고 한 줄을 내고 None
    """
    try:
        # stdin을 막아 ffmpeg가 터미널 키 입력을 읽지 않게 한다.
        # errors="replace"는 Whisper 출력에 깨진 UTF-8 바이트가 섞여도 UnicodeDecodeError로 질문 하나가 통째로 버려지지 않게 한다
        completed = subprocess.run(command, capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL)
    except OSError as error:
        print_warning(f"{command[0]} 실행 실패: {error}")
        return None
    if completed.returncode != 0:
        # 오류 원인은 보통 표준 에러의 마지막 줄에 있다
        error_lines = completed.stderr.strip().splitlines()
        last_error_line = error_lines[-1] if error_lines else "(표준 에러 없음)"
        print_warning(f"{Path(command[0]).name} 종료 코드 {completed.returncode}: {last_error_line}")
        return None
    return completed.stdout


def record_audio(wav_path=INPUT_WAV_PATH):
    """마이크로 RECORD_SECONDS초(5초) 녹음해 wav로 저장한다(Spoken Question 입력).

    입력: wav_path — 저장할 wav 경로(기본 src/audio/input.wav)
    출력: 녹음에 성공하면 True, 실패하면 경고 한 줄 후 False
    """
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    # 명령은 트랙별로 platform_config가 만든다(Mac ffmpeg, Jetson arecord — Jetson 경로: 보드 미검증)
    return run_command(build_record_command(wav_path)) is not None


def transcribe_wav(wav_path):
    """이미 있는 wav를 whisper.cpp로 글로 옮긴다(미리 녹음한 wav 확인용, User Story 59).

    입력: wav_path — 인식할 wav 경로. whisper-cli가 16kHz로 바꿔 읽으므로 Piper의 22050Hz wav도 그대로 넣을 수 있다
    출력: 인식된 글(앞뒤 공백 제거한 str). wav가 없거나 Whisper가 실패하면 경고 한 줄 후 빈 문자열
    """
    if not Path(wav_path).exists():
        print_warning(f"wav 파일이 없습니다: {wav_path}")
        return ""
    # -nt = 시간표(타임스탬프) 없이 글만 출력(mac-setup.md 4절). Jetson은 build-cpu/의 CPU 빌드(platform_config) — 보드 미검증
    command = [str(WHISPER_CLI_PATH), "-m", str(WHISPER_MODEL_PATH),
               "-f", str(wav_path), "-l", WHISPER_LANGUAGE, "-nt", "--prompt", WHISPER_PROMPT_HINT]
    output = run_command(command)
    if output is None:
        return ""
    return output.strip()


def compute_rms(wav_path):
    """16비트 PCM wav의 음량(RMS, 표본 값의 제곱 평균의 제곱근)을 구한다.

    입력: wav_path — 16비트 wav 경로(녹음 wav·Piper wav 모두 16비트)
    출력: RMS(float, 0~32767). 파일을 읽지 못하면 경고 한 줄 후 0.0(무음으로 본다)
    """
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
    except (OSError, wave.Error) as error:
        print_warning(f"wav 음량을 읽지 못했습니다: {error}")
        return 0.0
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def speech_to_text():
    """5초 녹음한 뒤 글로 옮긴다(Spoken Question).

    입력: 없음(녹음 결과는 src/audio/input.wav에 저장)
    출력: 인식된 글(str). 녹음·인식 실패나 무음이면 빈 문자열
    """
    if not record_audio(INPUT_WAV_PATH):
        return ""
    # 무음이면 Whisper가 없는 말을 지어내므로(조용한 방 녹음에서 확인) 음량이 기준보다 작으면 Whisper를 부르지 않는다
    rms = compute_rms(INPUT_WAV_PATH)
    if rms < SILENCE_RMS_THRESHOLD:
        print(f"[voice] 음량 {rms:.1f} < 기준 {SILENCE_RMS_THRESHOLD}, 무음으로 보고 인식을 건너뜁니다.", flush=True)
        return ""
    return transcribe_wav(INPUT_WAV_PATH)


def synthesize_speech(text, wav_path=RESPONSE_WAV_PATH):
    """Piper(ko_KR-kss-medium)로 글을 wav로 합성한다. 재생은 하지 않는다(bench가 합성 시간만 재려고 분리).

    입력: text — 읽을 한국어 글(str), wav_path — 저장할 wav 경로(기본 src/audio/response.wav)
    출력: 합성에 성공하면 True, Piper가 없거나 실패하면 경고 한 줄 후 False
    """
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    # Piper는 의존성 충돌을 피해 전용 venv의 python으로 실행한다(User Story 47). "--" 뒤가 읽을 글
    command = [str(PIPER_PYTHON_PATH), "-m", "piper", "-m", str(PIPER_VOICE_PATH),
               "-f", str(wav_path), "--", text]
    return run_command(command) is not None


def play_wav(wav_path=RESPONSE_WAV_PATH):
    """wav를 재생하고 끝날 때까지 기다린다. 다른 스레드가 stop_playback()을 부르면 바로 돌아온다.

    입력: wav_path — 재생할 wav 경로
    출력: 끝까지 재생했으면 True. 끊겼으면 False(경고 없음), 재생 실패면 경고 한 줄 후 False
    """
    global _playback_process
    # Popen으로 띄워 프로세스를 모듈에 기억해 둔다 — subprocess.run이면 다른 스레드가 끊을 손잡이가 없다
    try:
        _playback_process = subprocess.Popen(build_play_command(wav_path),
                                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except OSError as error:
        print_warning(f"재생 프로그램 실행 실패: {error}")
        return False
    _, error_output = _playback_process.communicate()
    return_code = _playback_process.returncode
    _playback_process = None
    # 음수 종료 코드 = 신호로 끝남, 즉 stop_playback()이 끊은 것이라 실패로 보지 않는다
    if return_code < 0:
        return False
    if return_code != 0:
        print_warning(f"재생 종료 코드 {return_code}: {error_output.strip()}")
        return False
    return True


def stop_playback():
    """재생 중인 음성을 끊는다(재생 중 v 키). 재생 중이 아니면 아무 일도 하지 않는다.

    입력: 없음, 출력: 없음
    """
    process = _playback_process
    # 지역 변수로 한 번 받아 둔다 — 그사이 재생 스레드가 _playback_process를 None으로 바꿔도 안전하다
    if process is not None and process.poll() is None:
        process.terminate()


def text_to_speech(text):
    """글을 Piper로 합성한 뒤 재생한다(SITREP·답변 읽기). 합성이 실패하면 재생하지 않는다.

    입력: text — 읽을 한국어 글(str)
    출력: 끝까지 재생했으면 True, 합성·재생 실패 또는 끊김이면 False
    """
    if not synthesize_speech(text, RESPONSE_WAV_PATH):
        return False
    return play_wav(RESPONSE_WAV_PATH)


def parse_arguments():
    """수동 확인용 명령줄 인자를 읽는다. 출력: argparse.Namespace(wav_path, say, record)"""
    parser = argparse.ArgumentParser(description="voice 모듈 수동 확인(각 단계 걸린 초 출력)")
    parser.add_argument("wav_path", nargs="?", help="글로 옮길 wav 경로(녹음 없이 STT만)")
    parser.add_argument("--say", help="합성·재생할 글")
    parser.add_argument("--record", action="store_true", help="5초 녹음 후 글로 옮기기")
    return parser.parse_args()


def main():
    """인자에 따라 STT(wav/녹음) 또는 TTS(합성·재생)를 한 번 돌리고 단계별 걸린 초를 출력한다(bench 재료)."""
    arguments = parse_arguments()
    if arguments.say is not None:
        start_time = time.perf_counter()
        is_synthesized = synthesize_speech(arguments.say, RESPONSE_WAV_PATH)
        print(f"합성: {time.perf_counter() - start_time:.2f}초 → {RESPONSE_WAV_PATH if is_synthesized else '실패'}")
        if is_synthesized:
            start_time = time.perf_counter()
            play_wav(RESPONSE_WAV_PATH)
            print(f"재생: {time.perf_counter() - start_time:.2f}초")
    elif arguments.record:
        start_time = time.perf_counter()
        is_recorded = record_audio(INPUT_WAV_PATH)
        print(f"녹음: {time.perf_counter() - start_time:.2f}초 → {INPUT_WAV_PATH if is_recorded else '실패'}")
        if is_recorded:
            # 기준값을 잴 때 쓰는 줄: 말할 때와 조용할 때 각각 이 값을 보고 SILENCE_RMS_THRESHOLD를 정한다
            print(f"음량(RMS): {compute_rms(INPUT_WAV_PATH):.1f} (무음 기준 {SILENCE_RMS_THRESHOLD})")
            start_time = time.perf_counter()
            text = transcribe_wav(INPUT_WAV_PATH)
            print(f"인식: {time.perf_counter() - start_time:.2f}초")
            print(f"글: {text!r}")
    elif arguments.wav_path is not None:
        start_time = time.perf_counter()
        text = transcribe_wav(arguments.wav_path)
        print(f"인식: {time.perf_counter() - start_time:.2f}초")
        print(f"글: {text!r}")
    else:
        print("사용법: voice.py <wav 경로> | --say \"<글>\" | --record")


if __name__ == "__main__":
    main()
