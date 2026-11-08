"""bench: 측정표 생성 도구. 구성 요소별 걸린 시간, YOLO 루프 FPS(단독/Gemma 동시), Event당 SITREP 응답 초, RAM 피크를 마크다운 표로 출력한다.

실행: .venv/bin/python bench.py components --wav <STT에 넣을 wav 경로>   (마이크·카메라·스피커 안 씀)
      .venv/bin/python bench.py fps [--seconds 20]      (카메라를 연다, 창은 안 띄움)
      .venv/bin/python bench.py ram [--tegrastats-log tegrastats.log]       (tegrastats 로그를 읽기만 함)
RAM 측정 방법(board-check.md 5번): Jetson 터미널 1에서 `tegrastats --interval 1000 | tee tegrastats.log`를 켜 둔 채
터미널 2에서 `bench.py fps`를 돌리고, 끝나면 tegrastats를 멈춘 뒤 `bench.py ram --tegrastats-log tegrastats.log`.
발표 숫자는 Jetson에서 잰 값만 쓴다. Mac 결과는 흐름 확인용이다.
"""
import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from camera import open_camera, shrink_frame
from detector import load_model
from main import create_clock, process_frame
from motion import compute_motion_level
from platform_config import GEMMA_MODEL_PATH, REPO_ROOT, TRACK, load_array_module
from reporter import Reporter, build_sitrep_prompt, generate_text, load_gemma
from state_machine import create_state
from voice import synthesize_speech, transcribe_wav

# Motion Level 합성 프레임 크기: 카메라 MJPG 1280x720과 같다(board-check.md 카메라 행)
MOTION_FRAME_WIDTH = 1280
MOTION_FRAME_HEIGHT = 720
# Motion Level 반복 횟수: 한 번에 수 ms라 여러 번 재야 평균이 흔들리지 않는다. 임시값, 실측 후 교체
MOTION_REPEAT_COUNT = 100
# 합성 프레임 난수 씨앗: 실행마다 같은 프레임 쌍을 쓰려고 고정한다
MOTION_RANDOM_SEED = 0
# TTS·STT·SITREP의 첫 회 뒤 반복 횟수: 한 번에 0.3~수 초라 3번이면 흐름 확인에 충분하다. 임시값, 실측 후 교체
VOICE_REPEAT_COUNT = 3
SITREP_REPEAT_COUNT = 3

# TTS로 합성할 고정 한국어 한 문장(SITREP 말투)
TTS_SENTENCE = "표적 1번이 먼 구역에서 가까운 구역으로 이동했고 칼을 들고 있습니다."
# TTS 측정용 임시 wav. 앱의 response.wav를 덮지 않게 따로 두고, 측정이 끝나면 지운다
BENCH_TTS_WAV_PATH = REPO_ROOT / "src/audio/bench_tts.wav"

# SITREP 측정용 고정 가짜 Event Batch와 Snapshot 재료(reporter.py 수동 확인과 같은 흐름)
FAKE_NOW = 30.0
FAKE_EVENTS = [
    {"timestamp": 10.0, "target_id": 1, "kind": "appeared", "before": None,
     "after": {"zone": "far", "held_objects": []}},
    {"timestamp": 18.0, "target_id": 1, "kind": "zone_changed", "before": "far", "after": "near"},
    {"timestamp": 25.0, "target_id": 1, "kind": "object_changed", "before": [], "after": ["knife"]},
]
FAKE_TARGETS = {1: {"zone": "near", "held_objects": ["knife"]}}

# fps 명령의 구간 하나(단독, 동시 각각) 길이 초. 임시값, 실측 후 교체
FPS_MEASURE_SECONDS = 20

# ram 명령의 기본 로그 경로: board-check.md 5번 명령(`tee tegrastats.log`)이 지금 폴더에 만드는 파일 이름
DEFAULT_TEGRASTATS_LOG = "tegrastats.log"
# tegrastats 한 줄에서 "RAM <사용>/<전체>MB"를 찾는다. 예: "RAM 1544/7620MB (lfb 3x4MB) SWAP 0/3810MB ..."
TEGRASTATS_RAM_PATTERN = re.compile(r"RAM (\d+)/(\d+)MB")

TABLE_HEADER = "| 항목 | 트랙 | 반복 | 평균 | 최소 | 최대 |"
TABLE_DIVIDER = "|---|---|---|---|---|---|"


def make_timing_row(item_name, values, unit):
    """잰 값 목록으로 표 한 행을 만든다(시간·FPS·%·MB 공통).

    입력: item_name(항목 이름), values(잰 값 목록, 비어 있지 않음), unit(표에 붙일 단위: "ms", "초", "FPS", "%", "MB")
          unit이 "ms"일 때만 values를 초로 받아 1000을 곱해 적는다. 다른 단위는 받은 값 그대로 적는다
    출력: (항목, 트랙, 반복, 평균, 최소, 최대) 글 튜플. 반복 칸은 값 개수
    """
    scale = 1000.0 if unit == "ms" else 1.0
    scaled_values = [value * scale for value in values]
    average = sum(scaled_values) / len(scaled_values)
    return (f"{item_name} ({unit})", TRACK, str(len(scaled_values)),
            f"{average:.2f}", f"{min(scaled_values):.2f}", f"{max(scaled_values):.2f}")


def make_note_row(item_name, note):
    """숫자 없이 사유만 적는 행을 만든다(실패, N/A). 입력: 항목 이름, 사유 글. 출력: 표 행 튜플"""
    return (item_name, TRACK, "-", note, "-", "-")


def format_table(rows):
    """표 행 목록을 마크다운 표 글로 만든다. 첫 줄은 트랙과 날짜. 입력: 행 튜플 목록, 출력: str"""
    lines = [f"측정표 — 트랙 {TRACK}, {datetime.now():%Y-%m-%d %H:%M}", "", TABLE_HEADER, TABLE_DIVIDER]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def make_motion_frames():
    """Motion Level 측정용 1280x720 BGR uint8 합성 프레임 두 장을 만든다. 출력: (이전 프레임, 현재 프레임)"""
    random_generator = np.random.default_rng(MOTION_RANDOM_SEED)
    shape = (MOTION_FRAME_HEIGHT, MOTION_FRAME_WIDTH, 3)
    previous_frame = random_generator.integers(0, 256, size=shape, dtype=np.uint8)
    current_frame = random_generator.integers(0, 256, size=shape, dtype=np.uint8)
    return previous_frame, current_frame


def time_motion_level(previous_frame, current_frame, array_module, synchronize):
    """Motion Level을 MOTION_REPEAT_COUNT번 재서 초 목록을 돌려준다.

    입력: CPU 프레임 두 장, 배열 모듈(numpy/cupy), synchronize(GPU 계산이 끝날 때까지 기다리는 함수, NumPy면 아무 일 안 함)
    출력: 반복마다 걸린 초 목록. 첫 호출(준비 실행)은 재지 않는다 — CuPy는 첫 호출에 커널 컴파일·GPU 초기화가
          들어가 그 시간이 평균을 부풀리기 때문이다(NumPy도 같은 방식으로 맞춘다)
    """
    compute_motion_level(previous_frame, current_frame, array_module)
    synchronize()
    seconds_list = []
    for _ in range(MOTION_REPEAT_COUNT):
        start_time = time.perf_counter()
        # 프레임은 매번 CPU 배열로 넘긴다 — 앱(main)처럼 CPU → GPU 복사 비용이 시간에 들어가게
        compute_motion_level(previous_frame, current_frame, array_module)
        # GPU는 비동기로 돌아 함수가 돌아와도 계산이 안 끝났을 수 있다. 끝날 때까지 기다린 뒤 시간을 잰다
        synchronize()
        seconds_list.append(time.perf_counter() - start_time)
    return seconds_list


def measure_motion_level():
    """Motion Level의 NumPy 행과 CuPy 행을 잰다. 출력: 표 행 두 개(NumPy, CuPy)"""
    previous_frame, current_frame = make_motion_frames()
    # NumPy는 두 트랙 모두 numpy를 직접 써서 잰다(load_array_module은 트랙의 모듈 하나만 준다)
    numpy_seconds = time_motion_level(previous_frame, current_frame, np, synchronize=lambda: None)
    rows = [make_timing_row("Motion Level NumPy", numpy_seconds, "ms")]

    if TRACK != "jetson":
        rows.append(make_note_row("Motion Level CuPy", "N/A (이 트랙에서는 잴 수 없음: CuPy 없음)"))
        return rows
    # Jetson 경로: 보드 미검증
    try:
        cupy_module = load_array_module()
        cupy_seconds = time_motion_level(previous_frame, current_frame, cupy_module,
                                         synchronize=lambda: cupy_module.cuda.Device().synchronize())
        rows.append(make_timing_row("Motion Level CuPy", cupy_seconds, "ms"))
    except Exception as error:
        # 한 항목이 실패해도 표 전체는 끝까지 낸다
        rows.append(make_note_row("Motion Level CuPy", f"실패: {error}"))
    return rows


def time_call(function):
    """함수를 한 번 부르고 (걸린 초, 반환값)을 돌려준다. 입력: 인자 없는 함수"""
    start_time = time.perf_counter()
    result = function()
    return time.perf_counter() - start_time, result


def measure_first_and_repeats(item_name, run_once, repeat_count, failure_note):
    """첫 회 1번과 그 뒤 repeat_count번을 따로 재서 행 두 개를 만든다.

    입력: item_name, run_once(한 번 실행하고 성공 여부를 돌려주는 함수), repeat_count,
          failure_note(실패했을 때 적을 이유)
    출력: [첫 회 행, 반복 행]. 첫 회는 프로그램·모델 초기화가 들어가 느려서 평균과 섞지 않는다.
          어느 회든 실패하면 그 행은 "실패: 이유"
    """
    first_seconds, is_success = time_call(run_once)
    if not is_success:
        note = f"실패: {failure_note}"
        return [make_note_row(f"{item_name} 첫 회", note), make_note_row(item_name, note)]
    rows = [make_timing_row(f"{item_name} 첫 회", [first_seconds], "초")]
    repeat_seconds = []
    for _ in range(repeat_count):
        seconds, is_success = time_call(run_once)
        if not is_success:
            rows.append(make_note_row(item_name, f"실패: {failure_note}"))
            return rows
        repeat_seconds.append(seconds)
    rows.append(make_timing_row(item_name, repeat_seconds, "초"))
    return rows


def measure_tts():
    """고정 한 문장을 Piper로 wav 합성하는 시간을 잰다(재생 제외). 측정용 wav는 끝나면 지운다. 출력: 표 행 두 개"""
    try:
        return measure_first_and_repeats(
            "TTS 합성", lambda: synthesize_speech(TTS_SENTENCE, BENCH_TTS_WAV_PATH), VOICE_REPEAT_COUNT,
            "Piper 합성 실패(위 [voice 경고] 참고)")
    finally:
        BENCH_TTS_WAV_PATH.unlink(missing_ok=True)


def measure_stt(wav_path):
    """이미 있는 wav를 Whisper로 글로 옮기는 시간을 잰다(녹음 5초 제외). 출력: 표 행 두 개

    빈 글이 나오면 실패로 본다 — transcribe_wav는 실패 때 빈 글을 돌려주고, 측정용 wav는 말이 든 것을 쓴다.
    """
    if not Path(wav_path).exists():
        note = f"실패: wav 파일 없음 {wav_path}"
        return [make_note_row("STT 인식 첫 회", note), make_note_row("STT 인식", note)]
    return measure_first_and_repeats(
        "STT 인식", lambda: transcribe_wav(wav_path) != "", VOICE_REPEAT_COUNT,
        "Whisper 인식 실패 또는 빈 글(위 [voice 경고] 참고)")


def measure_sitrep():
    """Gemma 로드 초와 고정 가짜 Event Batch의 SITREP 생성 초(스레드 없이)를 잰다. 출력: 표 행 세 개"""
    if not GEMMA_MODEL_PATH.exists():
        note = f"실패: 모델 파일 없음 {GEMMA_MODEL_PATH}"
        return [make_note_row("Gemma 모델 로드", note), make_note_row("SITREP 생성 첫 회", note),
                make_note_row("SITREP 생성", note)]
    try:
        load_seconds, model = time_call(lambda: load_gemma(GEMMA_MODEL_PATH))
    except Exception as error:
        # llama_cpp 미설치·GGUF 손상 등. 다른 행은 계속 낸다
        note = f"실패: {error}"
        return [make_note_row("Gemma 모델 로드", note), make_note_row("SITREP 생성 첫 회", note),
                make_note_row("SITREP 생성", note)]
    rows = [make_timing_row("Gemma 모델 로드", [load_seconds], "초")]
    try:
        # 회마다 now를 1초씩 바꿔 user 글("N초 전")을 다르게 한다. 같은 프롬프트를 반복하면 llama-cpp가 직전 입력의
        # KV를 다시 써서 앱(Event Batch마다 user 글이 다르고 system 글만 같음)보다 싸게 나올 수 있기 때문이다
        prompts = iter([build_sitrep_prompt(FAKE_EVENTS, FAKE_TARGETS, FAKE_NOW + run_index)
                        for run_index in range(1 + SITREP_REPEAT_COUNT)])

        def generate_next_sitrep():
            """다음 프롬프트로 SITREP 하나를 생성하고 빈 글이 아닌지 돌려준다(프롬프트는 미리 만들어 시간 밖)."""
            system_text, user_text = next(prompts)
            return generate_text(model, system_text, user_text) != ""

        rows += measure_first_and_repeats("SITREP 생성", generate_next_sitrep, SITREP_REPEAT_COUNT,
                                          "Gemma가 빈 글을 냈음")
    except Exception as error:
        # n_ctx 초과·Metal/CUDA 메모리 오류 등. 이미 잰 다른 행이 든 표는 끝까지 낸다
        note = f"실패: {error}"
        rows += [make_note_row("SITREP 생성 첫 회", note), make_note_row("SITREP 생성", note)]
    finally:
        # 모델을 쥔 채 프로세스가 끝나면 Mac에서 ggml-metal assert(exit 134)가 난 적이 있어 먼저 닫는다(reporter.stop과 같은 이유)
        model.close()
    return rows


def run_components(wav_path):
    """구성 요소를 차례로 재서 표 행 목록을 돌려준다. 입력: STT에 넣을 wav 경로. 출력: 행 튜플 목록"""
    rows = []
    print("[bench] Motion Level 측정 중")
    rows += measure_motion_level()
    print("[bench] TTS 측정 중")
    rows += measure_tts()
    print("[bench] STT 측정 중")
    rows += measure_stt(wav_path)
    print("[bench] Gemma 로드·SITREP 측정 중")
    rows += measure_sitrep()
    return rows


def run_detection_loop(capture, model, state, get_now, seconds, on_frame):
    """카메라 → 검출 → Target State 판정 → 상태기계(main.process_frame)를 seconds초 동안 돌린다.

    입력: capture(열린 카메라), model(YOLO), state(상태기계 상태), get_now(main.create_clock 결과),
          seconds(구간 길이), on_frame(프레임마다 처리 뒤 부를 함수 — 동시 구간의 가짜 Event 넣기, 단독이면 아무 일 안 함)
    출력: (frame_seconds, elapsed_seconds, new_state). frame_seconds는 프레임마다 읽기~처리 끝까지 초 목록,
          elapsed_seconds는 구간 전체 초. 프레임을 못 읽으면 그때까지 잰 것으로 멈춘다
    창은 띄우지 않는다 — imshow·HUD 그리기 비용을 빼고 YOLO 루프 자체의 FPS만 재기 위해서다.
    이번 프레임의 실제 Event는 버린다(동시 구간에서 생성할 Event Batch는 가짜로 따로 넣는다).
    """
    frame_seconds = []
    loop_start = time.perf_counter()
    while time.perf_counter() - loop_start < seconds:
        frame_start = time.perf_counter()
        is_read, frame = capture.read()
        if not is_read:
            print("[bench] 카메라 프레임을 읽지 못해 이 구간을 멈춥니다.", flush=True)
            break
        # main과 같은 처리 크기로 재야 앱 FPS와 비교된다
        _, state, _ = process_frame(model, shrink_frame(frame), state, get_now())
        on_frame()
        frame_seconds.append(time.perf_counter() - frame_start)
    return frame_seconds, time.perf_counter() - loop_start, state


def make_loop_rows(label, frame_seconds, elapsed_seconds):
    """한 구간의 프레임 시간 행과 FPS 행을 만든다.

    입력: label("단독" 또는 "동시"), frame_seconds(프레임별 초 목록), elapsed_seconds(구간 전체 초)
    출력: (행 두 개, FPS 또는 None). FPS는 처리한 프레임 수 ÷ 구간 초(프레임별 FPS의 평균이 아님)
    """
    if not frame_seconds:
        note = "실패: 처리한 프레임 없음"
        return [make_note_row(f"YOLO 프레임 시간 {label}", note), make_note_row(f"YOLO 루프 FPS {label}", note)], None
    fps = len(frame_seconds) / elapsed_seconds
    return [make_timing_row(f"YOLO 프레임 시간 {label}", frame_seconds, "ms"),
            make_timing_row(f"YOLO 루프 FPS {label}", [fps], "FPS")], fps


def make_fake_event_batch(batch_index, now):
    """동시 구간에서 넣을 가짜 Event Batch(Event 한 건)와 Snapshot 재료를 만든다.

    입력: batch_index(몇 번째 묶음, 0부터), now(지금 시각, 실행 시작 기준 초)
    출력: (Event 목록, {target_id: Target State}). 묶음마다 Target id를 바꿔 프롬프트를 다르게 한다
          (같은 프롬프트면 llama-cpp가 직전 입력을 다시 써서 생성이 싸게 나올 수 있다 — measure_sitrep와 같은 이유)
    """
    target_id = batch_index + 1
    target_state = {"zone": "far", "held_objects": []}
    events = [{"timestamp": now, "target_id": target_id, "kind": "appeared", "before": None, "after": target_state}]
    return events, {target_id: target_state}


def make_fake_batch_feeder(reporter, get_now):
    """직전 가짜 Event Batch의 SITREP이 나오면 다음 묶음을 넣는 함수를 만든다(동시 구간의 on_frame).

    입력: reporter(시작된 Reporter), get_now.
    출력: (넣는 함수, 지금까지 넣은 묶음 수를 돌려주는 함수). 둘 다 인자 없음. 넣은 수는 생성 실패 건수 계산용.
    "끝나면 바로 다음"으로 넣는 이유: 생성이 쉬지 않고 돌게 하면서, 묶음이 대기열에서 앞 생성을 기다리는 시간이
    응답 초에 섞이지 않게 하려고(앱처럼 Event가 생길 때 넣은 것과 같은 정의 — 넣은 시각 → SITREP 출력).
    끝났는지는 reporter.sitrep_seconds 길이로 본다(SITREP 하나가 나올 때마다 한 칸 늘어난다).
    """
    submitted_count = 0

    def feed_if_previous_done():
        """직전 묶음의 SITREP이 나왔으면(sitrep_seconds 길이로 판단) 가짜 Event 묶음 하나를 넣는다. 입력·출력 없음."""
        nonlocal submitted_count
        if len(reporter.sitrep_seconds) < submitted_count:
            return
        events, targets = make_fake_event_batch(submitted_count, get_now())
        reporter.submit_events(events, targets)
        submitted_count += 1

    return feed_if_previous_done, lambda: submitted_count


def measure_concurrent(capture, model, state, get_now, seconds, standalone_fps):
    """Gemma가 SITREP을 계속 생성하는 동안 같은 YOLO 루프 FPS와 Event당 응답 초를 잰다.

    입력: run_detection_loop와 같은 재료, standalone_fps(단독 FPS 또는 None — 비율 계산용)
    출력: 표 행 목록(프레임 시간·FPS 동시, 비율, 응답 초). Gemma를 못 불러오면 그 행들은 "실패: 이유"
    """
    failure_names = ["YOLO 프레임 시간 동시", "YOLO 루프 FPS 동시", "동시/단독 FPS 비율", "Event → SITREP 응답"]
    # Event Log 자리는 None: bench는 질문을 넣지 않고, Event Log는 질문 답을 만들 때만 읽힌다(logs/에 파일도 안 생김)
    reporter = Reporter(None, get_now, is_voice_enabled=False)
    try:
        load_seconds = reporter.start()
        print(f"[bench] Gemma 로드: {load_seconds:.2f}초", flush=True)
    except Exception as error:
        # 모델 파일 없음·llama_cpp 미설치 등. 단독 구간 행은 이미 있으니 표는 끝까지 낸다
        return [make_note_row(name, f"실패: Gemma 로드 {error}") for name in failure_names]
    try:
        print(f"[bench] 동시 구간 {seconds}초 측정 중(Gemma 생성 중)", flush=True)
        feed_if_previous_done, get_submitted_count = make_fake_batch_feeder(reporter, get_now)
        frame_seconds, elapsed_seconds, _ = run_detection_loop(
            capture, model, state, get_now, seconds, feed_if_previous_done)
        # 구간 끝에 생성 중이던 마지막 SITREP까지 기다린다. 넣는 쪽(이 루프)이 끝난 뒤 같은 메인 스레드에서 부르므로
        # 기다리는 동안 새 일을 넣는 스레드가 없다
        reporter.wait_until_idle()
    finally:
        # 모델을 쥔 작업 스레드를 남긴 채 끝나면 Mac에서 ggml-metal assert(exit 134)가 나므로 꼭 멈춘다
        reporter.stop()

    rows, concurrent_fps = make_loop_rows("동시", frame_seconds, elapsed_seconds)
    # reporter는 생성에 실패한 묶음을 sitrep_seconds에 남기지 않는다. wait_until_idle() 뒤에는 넣은 묶음이 모두
    # 성공 또는 실패로 끝났으므로 (넣은 수 - 나온 수)가 정확한 실패 건수다
    failed_count = get_submitted_count() - len(reporter.sitrep_seconds)
    if failed_count > 0:
        # 한 번 실패하면 넣는 함수가 다음 묶음을 영영 넣지 않아 남은 동시 구간은 Gemma가 쉰 채 돈다.
        # 그 FPS·비율은 부풀었을 수 있어 숫자 대신 실패 사유를 적는다(rows[1]은 make_loop_rows의 FPS 행)
        note = f"실패: SITREP 생성 실패 {failed_count}건 — 동시 구간에 생성이 멈췄을 수 있음"
        rows[1] = make_note_row("YOLO 루프 FPS 동시", note)
        rows.append(make_note_row("동시/단독 FPS 비율", note))
    elif standalone_fps and concurrent_fps:
        rows.append(make_timing_row("동시/단독 FPS 비율", [concurrent_fps / standalone_fps * 100], "%"))
    else:
        rows.append(make_note_row("동시/단독 FPS 비율", "N/A (단독 또는 동시 FPS 없음)"))
    if reporter.sitrep_seconds:
        rows.append(make_timing_row("Event → SITREP 응답", reporter.sitrep_seconds, "초"))
    else:
        rows.append(make_note_row("Event → SITREP 응답", "실패: 구간 안에 나온 SITREP 없음"))
    return rows


def run_fps(seconds):
    """단독 구간 → 동시 구간 순서로 YOLO 루프를 재서 표 행 목록을 돌려준다.

    입력: seconds(구간 하나 길이). 출력: 행 튜플 목록
    허용 비율 X%는 스펙에서 미정이라 판정하지 않고 숫자만 적는다.
    """
    model = load_model()
    get_now = create_clock()
    state = create_state()
    try:
        # 카메라는 Jetson GStreamer 파이프라인(camera.JETSON_PIPELINE)으로만 연다
        capture = open_camera()
    except RuntimeError as error:
        return [make_note_row("YOLO 루프 FPS", f"실패: {error}")]
    try:
        # 준비 실행 한 번: 첫 추론에는 모델·추적기 초기화가 들어가 첫 프레임 시간이 평균을 부풀리기 때문이다
        is_read, frame = capture.read()
        if is_read:
            _, state, _ = process_frame(model, shrink_frame(frame), state, get_now())
        print(f"[bench] 단독 구간 {seconds}초 측정 중", flush=True)
        frame_seconds, elapsed_seconds, state = run_detection_loop(
            capture, model, state, get_now, seconds, on_frame=lambda: None)
        rows, standalone_fps = make_loop_rows("단독", frame_seconds, elapsed_seconds)
        rows += measure_concurrent(capture, model, state, get_now, seconds, standalone_fps)
    finally:
        capture.release()
    return rows


def read_ram_used_values(log_path):
    """tegrastats 로그의 줄마다 "RAM <사용>/<전체>MB"의 사용 MB를 읽는다.

    입력: log_path(로그 파일 경로). 출력: 사용 MB 정수 목록(RAM 항목이 없는 줄은 건너뜀)
    """
    used_values = []
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = TEGRASTATS_RAM_PATTERN.search(line)
        if match:
            used_values.append(int(match.group(1)))
    return used_values


def measure_ram(log_path):
    """tegrastats 로그의 RAM 사용량으로 표 한 행을 만든다. 최대 칸이 RAM 피크다.

    입력: log_path. 출력: 행 한 개 목록. 파일이 없거나(Mac 트랙) RAM 줄이 없으면 "N/A (이유)" 행
    7.0GB 기준(스펙 성공 판정) 판정은 하지 않고 숫자만 적는다.
    """
    item_name = "RAM 사용, 최대가 피크"
    # Jetson 경로: 보드 미검증 — 실제 tegrastats 출력 한 줄 전체로는 아직 파싱해 보지 않았다
    if not Path(log_path).exists():
        return [make_note_row(f"{item_name} (MB)", f"N/A (tegrastats 로그 없음: {log_path})")]
    used_values = read_ram_used_values(log_path)
    if not used_values:
        return [make_note_row(f"{item_name} (MB)", f"N/A (로그에 RAM 줄 없음: {log_path})")]
    return [make_timing_row(item_name, used_values, "MB")]


def parse_arguments():
    """명령줄 인자를 읽는다. 출력: argparse.Namespace(command와 하위 명령별 인자)"""
    parser = argparse.ArgumentParser(description="Recon SITREP Assistant 측정표 생성")
    subparsers = parser.add_subparsers(dest="command", required=True)
    components_parser = subparsers.add_parser("components", help="구성 요소별 걸린 시간")
    components_parser.add_argument("--wav", required=True, help="STT 시간을 잴 wav 경로(말이 든 wav, 마이크 없이)")
    fps_parser = subparsers.add_parser("fps", help="YOLO 루프 FPS 단독/Gemma 동시 + Event당 SITREP 응답 초(카메라 사용)")
    fps_parser.add_argument("--seconds", type=float, default=FPS_MEASURE_SECONDS, help="구간 하나(단독·동시 각각) 길이 초")
    ram_parser = subparsers.add_parser("ram", help="tegrastats 로그의 RAM 피크")
    ram_parser.add_argument("--tegrastats-log", default=DEFAULT_TEGRASTATS_LOG, help="tegrastats 로그 파일 경로")
    return parser.parse_args()


def main():
    """하위 명령에 맞는 측정을 돌리고 마크다운 표를 출력한다."""
    arguments = parse_arguments()
    if arguments.command == "components":
        rows = run_components(arguments.wav)
    elif arguments.command == "fps":
        rows = run_fps(arguments.seconds)
    else:
        rows = measure_ram(arguments.tegrastats_log)
    print()
    print(format_table(rows))


if __name__ == "__main__":
    main()
