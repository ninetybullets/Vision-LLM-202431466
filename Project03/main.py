"""Recon SITREP Assistant 메인 루프: 카메라 → 검출 → Target State 판정 → 상태기계 → Event Log + 터미널 한 줄 → reporter(SITREP) → HUD.

Event가 생긴 프레임에만 터미널에 Event 한 줄을 찍고 reporter에 넘긴다(작업 스레드의 Gemma가 SITREP + Evidence를 터미널에 출력).
창에는 HUD(박스·Target id·Zone·Held Object·FPS·Motion Level)를 그린 영상을 띄운다. 키 v는 녹음 질문, 1·2는 Preset Question, q는 종료.
실행: .venv/bin/python main.py
"""
import time

import cv2

from camera import open_camera
from detector import detect_frame, load_model
from event_log import EventLog, format_event
from hud import draw_hud
from motion import compute_motion_level
from platform_config import load_array_module
from reporter import Reporter
from state_machine import build_observations, create_state, read_targets, step
from voice import stop_playback

WINDOW_NAME = "Recon SITREP Assistant"
# cv2.waitKey 대기 ms. 1ms만 기다려 키 입력을 받고 바로 다음 프레임으로 간다(Project01과 같음)
KEY_WAIT_MILLISECONDS = 1
QUIT_KEY = "q"
# FPS 지수이동평균 가중치. 작을수록 숫자가 덜 튄다(Project01 main과 같은 값)
FPS_SMOOTHING = 0.1

# Preset Question 두 개(스펙 「음성」: 마이크가 안 될 때를 위한 키 1·2). 질문 글은 Spoken Question과 같은 길로 reporter에 간다
PRESET_QUESTION_RECENT = "지난 1분 동안 무엇이 바뀌었나?"
PRESET_QUESTION_CURRENT = "지금 상황은 어떤가?"
PRESET_QUESTIONS = {"1": PRESET_QUESTION_RECENT, "2": PRESET_QUESTION_CURRENT}
# 녹음 질문 키(스펙 「음성」: v → 5초 녹음 → Whisper → 질문)
SPOKEN_QUESTION_KEY = "v"


def create_clock():
    """실행 시작 시각을 한 번 잡고, 그 기준 상대 초를 돌려주는 함수를 만든다.

    입력: 없음
    출력: get_now() 함수 — 부를 때마다 실행 시작부터 지난 초(float)를 준다.
        Event의 timestamp와 reporter의 "N초 전"이 같은 기준을 쓰도록 이 함수 하나를 같이 넘긴다.
    """
    # 벽시계(time.time)는 시스템 시각 조정에 따라 뒤로 갈 수 있어, 늘 앞으로만 가는 monotonic을 쓴다
    start_time = time.monotonic()

    def get_now():
        return time.monotonic() - start_time

    return get_now


def process_frame(model, frame, state, now):
    """프레임 한 장을 검출 → Target State 판정 → 상태기계 한 단계로 처리한다.

    입력: model — detector.load_model()의 모델(매 프레임 같은 객체를 넘겨야 추적 id가 이어진다),
        frame — BGR 프레임, state — 이전 상태기계 상태, now — 이 프레임의 시각(실행 시작 기준 초)
    출력: (persons, new_state, events)
        persons — 이번 프레임에 보인 사람 [(track_id, (x1, y1, x2, y2))] (HUD가 박스를 그릴 재료)
        new_state — 다음 프레임에 넘길 상태, events — 이번 프레임에 생긴 Event 목록(없으면 빈 목록)
    """
    persons, objects = detect_frame(model, frame)
    observations = build_observations(persons, objects, frame.shape[0])
    new_state, events = step(state, observations, now)
    return persons, new_state, events


def update_fps(fps, previous_time, now):
    """직전 프레임과 이번 프레임의 시각 차로 순간 FPS를 구해 지수이동평균에 섞는다(Project01 main 방식).

    입력: fps — 지금까지의 FPS(첫 프레임이면 None), previous_time·now — 직전·이번 프레임 시각(초)
    출력: 새 FPS(float). 첫 프레임은 순간 FPS 그대로.
    """
    instant_fps = 1 / (now - previous_time) if now > previous_time else 0.0
    if fps is None:
        return instant_fps
    return (1 - FPS_SMOOTHING) * fps + FPS_SMOOTHING * instant_fps


def record_events(event_log, events):
    """Event를 Event Log에 쌓고 터미널에 한 줄씩 찍는다. 입력: EventLog, Event 목록. 출력: 없음"""
    for event in events:
        event_log.append(event)
        print(format_event(event), flush=True)


def submit_preset_question(reporter, pressed_key, targets):
    """누른 키가 1·2이면 해당 Preset Question을 reporter에 넣는다. 다른 키는 아무것도 하지 않는다.

    입력: reporter(시작된 Reporter), pressed_key(cv2.waitKey 결과 & 0xFF), targets(지금 Snapshot 재료)
    출력: 없음. 질문을 처리하는 중이면(스펙 「임시 결정」: 한 번에 질문 하나) 넣지 않고 무시했다고 터미널에 한 줄 알린다.
    """
    question = PRESET_QUESTIONS.get(chr(pressed_key))
    if question is None:
        return
    if reporter.is_question_busy():
        print(f"질문을 처리하는 중이라 키 {chr(pressed_key)} 입력을 무시합니다.", flush=True)
        return
    reporter.submit_question(question, targets)


def submit_spoken_question(reporter, pressed_key, targets):
    """누른 키가 v이면 녹음 질문을 reporter에 넣고 재생 중인 소리를 끊는다. 다른 키는 아무것도 하지 않는다.

    입력: reporter(시작된 Reporter), pressed_key(cv2.waitKey 결과 & 0xFF), targets(v를 누른 프레임의 Snapshot 재료)
    출력: 없음. 녹음·질문 처리 중이면 넣지 않고 무시했다고 한 줄 알린다(재생 중이면 처리 중이 아니라서 받는다).
    """
    if chr(pressed_key) != SPOKEN_QUESTION_KEY:
        return
    if reporter.is_question_busy():
        print(f"질문을 처리하는 중이라 키 {SPOKEN_QUESTION_KEY} 입력을 무시합니다.", flush=True)
        return
    # 녹음 질문을 먼저 넣고 그다음 끊는다: 반대 순서면 끊긴 직후 작업 스레드가 대기 중인 다른 일을 먼저 집을 수 있다
    reporter.submit_spoken_question(targets)
    stop_playback()


def close_resources(capture, reporter, event_log):
    """카메라 → reporter → Event Log 파일 → 창 순서로 정리한다.

    입력: capture(열지 못했으면 None), reporter(시작된 Reporter), event_log. 출력: 없음.
    앞 단계가 예외를 내도 reporter.stop()과 event_log.close()는 불리도록 try/finally로 잇는다.
    창 닫기를 맨 뒤에 두는 이유: 창을 띄울 수 없는 환경에서는 destroyAllWindows가 예외를 낼 수 있다.
    """
    try:
        if capture is not None:
            capture.release()
    finally:
        try:
            # 모델을 쥔 작업 스레드가 살아 있는 채로 끝나면 Mac에서 ggml-metal assert(exit 134)가 나므로 꼭 멈춘다.
            # 생성 중이면 그 글이 끝날 때까지(수 초) 기다린다
            reporter.stop()
        finally:
            event_log.close()
    cv2.destroyAllWindows()


def run():
    """카메라 루프. q를 누르거나 프레임을 못 읽을 때까지 프레임을 처리·표시하고, Event·질문(v·1·2)을 reporter에 넘긴다.

    입력: 없음.
    출력: 없음. 끝나면 카메라·reporter·Event Log 파일·창을 닫는다.
    """
    # 모델과 배열 모듈을 먼저 읽는다: 둘 중 하나가 실패해도 열린 채 남는 카메라가 없게
    model = load_model()
    # Jetson 경로: 보드 미검증 — Jetson이면 cupy(못 불러오면 RuntimeError로 멈춤), Mac이면 numpy
    array_module = load_array_module()
    event_log = EventLog()
    print(f"Event Log: {event_log.path}", flush=True)
    # 시각 기준은 이 get_now 하나: Event timestamp와 reporter 프롬프트의 "N초 전"이 같은 기준이어야 한다
    get_now = create_clock()
    reporter = Reporter(event_log, get_now)
    # Gemma 로드(수 초)도 카메라를 열기 전에 끝낸다. 실패하면 여기서 멈추고, 열린 것은 Event Log 파일뿐이다
    load_seconds = reporter.start()
    print(f"Gemma 로드: {load_seconds:.2f}초", flush=True)
    state = create_state()
    fps = None
    previous_frame = None
    capture = None
    try:
        # 카메라 열기는 try 안: 실패해도 finally에서 reporter 작업 스레드를 멈춘다
        capture = open_camera()
        # 카메라를 연 뒤에 잡는다: 여는 데 걸린 시간이 첫 FPS에 섞이지 않게
        previous_time = get_now()
        while True:
            is_read, frame = capture.read()
            if not is_read:
                print("카메라 프레임을 읽지 못해 종료합니다.", flush=True)
                break
            now = get_now()
            persons, state, events = process_frame(model, frame, state, now)
            # Snapshot 재료와 HUD 재료는 같은 값이라 프레임마다 한 번만 읽는다
            targets = read_targets(state)
            # 순서: Event Log 기록(record_events) → reporter. reporter는 Event Log에 쓰지 않고 질문 때 읽기만 한다
            record_events(event_log, events)
            # 바로 돌아온다(대기열에 넣기만). 빈 목록이면 reporter가 무시한다
            reporter.submit_events(events, targets)

            # HUD가 frame에 글자를 그리기 전에 Motion Level을 구하고 원본을 복사해 둔다(다음 프레임과 비교할 때 글자가 차이로 잡히지 않게)
            motion_level = compute_motion_level(previous_frame, frame, array_module)
            previous_frame = frame.copy()
            fps = update_fps(fps, previous_time, now)
            previous_time = now
            draw_hud(frame, persons, targets, fps, motion_level)

            cv2.imshow(WINDOW_NAME, frame)
            # waitKey 결과에서 하위 8비트(키 코드)만 남긴다 — 키가 없으면 -1이 255가 되어 앱이 쓰는 어느 키와도 같지 않다
            pressed_key = cv2.waitKey(KEY_WAIT_MILLISECONDS) & 0xFF
            if pressed_key == ord(QUIT_KEY):
                break
            submit_spoken_question(reporter, pressed_key, targets)
            submit_preset_question(reporter, pressed_key, targets)
    finally:
        close_resources(capture, reporter, event_log)


def main():
    """카메라 루프를 시작한다. 카메라는 camera.JETSON_PIPELINE으로 연다."""
    run()


if __name__ == "__main__":
    main()
