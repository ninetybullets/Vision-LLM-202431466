"""Project01 — 항복 의사 검증 HUD: 정찰 자산(드론·로봇) 영상에서 대상(최대 3명)의 양손(펴짐·든 물체·가림)과 손 높이로
거짓 항복·공격 준비 여부를 DANGER / UNKNOWN / SURRENDER로 판단해 overlay.

실행: python main.py --camera mac   (맥 내장 웹캠)
      python main.py --camera jetson (Jetson USB MJPG 1280x720)
종료: q
"""
import argparse
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2

from camera import open_camera
from hand_analyzer import HandAnalyzer, HandObservation, HandState
from held_object import HeldObjectDetector, does_overlap_hand, find_object_contour
from hud import draw_hud
from person_detector import PersonDetector, compute_body_center, compute_shoulder_width, get_visible_points_by_index, is_human
from preprocess import resize_for_processing
from target_tracker import MultiTargetTracker, TrackState
from threat_classifier import ThreatDebouncer, ThreatState, are_hands_raised, classify_person

WINDOW_NAME = "Project01 Safety HUD"
# fps 지수이동평균 가중치. 작을수록 숫자가 덜 튄다
FPS_SMOOTHING = 0.1
HAND_SIDES = ("left", "right")
# 물체 검출이 끊겨도 이 프레임 수 동안 OBJECT를 유지한다. 검출률이 13~84%로 깜빡여(실측) 그대로 쓰면 디바운스(5연속)를 못 넘긴다. 맥 실측 15fps 기준 약 0.7초
OBJECT_HOLD_FRAMES = 10
# 손 분석(HandLandmarker)은 홀수 프레임, 물체 검출기는 짝수 프레임(held_object.DETECT_EVERY_FRAMES=2 전제)에 돌려 프레임당 비용을 고르게 나눈다
HAND_ANALYSIS_FRAME_PARITY = 1
# 손목은 보이는데 crop에서 손을 이 프레임 수(15fps 기준 ≈1초) 연속 못 찾으면 '가린 손'(HIDDEN)으로 본다.
# 프레임 기준이라 손 분석을 건너뛰고 직전 결과를 재사용한 프레임도 센다(한 손은 4프레임마다 분석되므로 실제 HandLandmarker 판정 약 4회).
# 실측: 손을 등 뒤로 숨겨도 Pose가 손목 위치를 추정해 visibility가 안 떨어지고(HIDDEN 4/58) NOT_FOUND(54/58)로 나온 적이 있다.
# 거리 보정 실측: 2.5m 빈손 내림은 NOT_FOUND 0, 자세 전환 중 최장 연속 12 → 15 유지(전환 중 오탐 없이 1초 내 가림 판정)
NOT_FOUND_FRAMES_AS_HIDDEN = 15


@dataclass
class PersonView:
    """HUD가 그릴 사람 한 명분 결과. PREDICTING이면 points_by_index·hands가 비고 threat_state는 직전 안정 상태."""

    person_id: int
    track_state: TrackState
    center: tuple
    threat_state: ThreatState
    points_by_index: dict = field(default_factory=dict)
    hands: dict = field(default_factory=dict)  # {"left"|"right": HandObservation}
    lock_seconds: Optional[float] = None  # run()이 채움. HUD 라벨용


@dataclass
class PersonState:
    """프레임을 넘어 유지하는 사람별 상태: 디바운서, 마지막 양손 분석 결과, 손별 물체 유지 카운터."""

    debouncer: ThreatDebouncer = field(default_factory=ThreatDebouncer)
    hands: dict = field(default_factory=dict)  # {"left"|"right": HandObservation} 물체 덮어쓰기 전의 손 분석 결과
    object_hold: dict = field(default_factory=lambda: {side: 0 for side in HAND_SIDES})  # 남은 OBJECT 유지 프레임
    object_info: dict = field(default_factory=dict)  # {side: (검출기 라벨, 색 분할 윤곽)} 마지막으로 겹친 검출
    not_found_frames: dict = field(default_factory=lambda: {side: 0 for side in HAND_SIDES})  # 손별 NOT_FOUND 연속 프레임 수


def apply_hidden_hands(hands, person_state):
    """NOT_FOUND가 NOT_FOUND_FRAMES_AS_HIDDEN 프레임 연속이면 그 손을 HIDDEN으로 바꾼 새 {side: HandObservation}을 돌려준다.
    person_state.not_found_frames를 제자리 갱신(다른 상태가 나오면 0으로)."""
    updated_hands = {}
    for side, observation in hands.items():
        if observation.state == HandState.NOT_FOUND:
            person_state.not_found_frames[side] += 1
            if person_state.not_found_frames[side] >= NOT_FOUND_FRAMES_AS_HIDDEN:
                observation = HandObservation(HandState.HIDDEN, observation.crop_box)
        else:
            person_state.not_found_frames[side] = 0
        updated_hands[side] = observation
    return updated_hands


def update_hands(frame, landmarks, hand_analyzer, frame_size, previous_hands, frame_index):
    """양손 결과 {side: HandObservation}을 돌려준다. 처음 본 사람은 양손 모두 분석하고, 그 뒤로는 홀수 프레임마다 한 손씩 번갈아
    분석하며 나머지는 직전 결과를 재사용한다. HandLandmarker가 비싸서(손 crop당 약 23ms) 호출 수를 줄이기 위함.
    5프레임 디바운스가 있어 한두 프레임 늦는 손 상태는 결과에 거의 영향이 없다.
    """
    if len(previous_hands) < len(HAND_SIDES):
        return {side: hand_analyzer.analyze(frame, landmarks, side, frame_size) for side in HAND_SIDES}
    hands = dict(previous_hands)
    if frame_index % 2 == HAND_ANALYSIS_FRAME_PARITY:
        side = HAND_SIDES[(frame_index // 2) % len(HAND_SIDES)]
        hands[side] = hand_analyzer.analyze(frame, landmarks, side, frame_size)
    return hands


def apply_held_objects(frame, hands, detections, person_state):
    """손 박스와 겹치는 검출이 있으면 그 손을 OBJECT로 덮어쓴 새 {side: HandObservation}을 돌려준다.
    활짝 편 손(OPEN)은 물건을 쥘 수 없으므로 제외한다 — 빈손을 펴 들었을 때 오검출(remote 등)로 DANGER가 뜨던 것을 막는다(실측 14%).
    검출이 끊겨도 OBJECT_HOLD_FRAMES 동안은 OBJECT를 유지한다. person_state.object_hold·object_info를 제자리 갱신.
    OBJECT일 때 검출기 라벨과 색 분할 윤곽을 HandObservation에 붙인다(HUD 표시용).
    """
    updated_hands = {}
    for side, observation in hands.items():
        if observation.state == HandState.OPEN:
            person_state.object_hold[side] = 0
            updated_hands[side] = observation
            continue
        matched = next((detection for detection in detections if does_overlap_hand(detection.box, observation.crop_box)), None)
        is_held = matched is not None or person_state.object_hold[side] > 0
        if matched is not None:
            person_state.object_hold[side] = OBJECT_HOLD_FRAMES
            person_state.object_info[side] = (matched.label, find_object_contour(frame, matched.box))
        elif person_state.object_hold[side] > 0:
            person_state.object_hold[side] -= 1
        if is_held and observation.crop_box is not None:
            label, contour = person_state.object_info.get(side, (None, None))
            observation = HandObservation(HandState.OBJECT, observation.crop_box, label, contour)
        updated_hands[side] = observation
    return updated_hands


def process_frame(frame, detector, hand_analyzer, object_detector, tracker, person_states, timestamp_ms, frame_index):
    """한 프레임을 처리해 (축소 프레임, PersonView 목록)을 돌려준다. person_states는 {person_id: PersonState}로 호출자가 보관.

    흐름: 축소 → Pose(여러 명) → 사람 판정 → 몸 중심 → 다중 Kalman 매칭 → 물체 검출(전체 프레임) →
          사람별 양손 분석(번갈아) → 오래 못 찾은 손은 HIDDEN → 손↔검출 겹침으로 OBJECT → 위협 규칙 → 디바운스.
    frame_index는 손 분석·물체 검출을 몇 번째 프레임에 돌릴지 정하는 프레임 번호.
    """
    processing_frame = resize_for_processing(frame)
    frame_size = (processing_frame.shape[1], processing_frame.shape[0])

    poses = [landmarks for landmarks in detector.detect(processing_frame, timestamp_ms) if is_human(landmarks, frame_size)]
    centers = [compute_body_center(landmarks, frame_size) for landmarks in poses]
    shoulder_widths = [compute_shoulder_width(landmarks, frame_size) for landmarks in poses]
    tracked_persons = tracker.update(centers, shoulder_widths)
    detections = object_detector.detect(processing_frame, timestamp_ms, frame_index) if tracked_persons else []

    person_views = []
    for tracked in tracked_persons:
        person_state = person_states.setdefault(tracked.person_id, PersonState())
        view = PersonView(tracked.person_id, tracked.state, tracked.center, person_state.debouncer.state)
        if tracked.measurement_index is not None:
            landmarks = poses[tracked.measurement_index]
            view.points_by_index = get_visible_points_by_index(landmarks, frame_size)
            person_state.hands = update_hands(processing_frame, landmarks, hand_analyzer, frame_size, person_state.hands, frame_index)
            hands = apply_hidden_hands(person_state.hands, person_state)
            view.hands = apply_held_objects(processing_frame, hands, detections, person_state)
            candidate = classify_person(view.hands["left"].state, view.hands["right"].state, are_hands_raised(landmarks))
            view.threat_state = person_state.debouncer.update(candidate)
        person_views.append(view)

    # 사라진 사람의 상태는 버린다(ID는 재사용되지 않으므로 다시 쓸 일 없음)
    alive_ids = {view.person_id for view in person_views}
    for person_id in list(person_states):
        if person_id not in alive_ids:
            del person_states[person_id]
    return processing_frame, person_views


def run(backend, camera_index):
    """카메라 루프. q를 누를 때까지 프레임을 읽어 처리·표시한다."""
    capture = open_camera(backend, camera_index)
    detector = PersonDetector()
    hand_analyzer = HandAnalyzer()
    object_detector = HeldObjectDetector()
    tracker = MultiTargetTracker()
    person_states = {}
    lock_start_times = {}  # person_id → 처음 잡힌 시각 (lock 유지 시간 표시용)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # 발표 때 창을 늘릴 수 있게

    start_time = time.monotonic()
    previous_time = start_time
    previous_timestamp_ms = -1
    frame_index = 0
    fps = None
    try:
        while True:
            is_read, frame = capture.read()
            if not is_read:
                break
            now = time.monotonic()
            # PoseLandmarker VIDEO 모드는 단조 증가하는 ms 타임스탬프를 요구한다. 같은 ms가 두 번 나오지 않게 보정
            timestamp_ms = max(int((now - start_time) * 1000), previous_timestamp_ms + 1)
            previous_timestamp_ms = timestamp_ms
            frame_index += 1

            processing_frame, person_views = process_frame(
                frame, detector, hand_analyzer, object_detector, tracker, person_states, timestamp_ms, frame_index
            )

            alive_ids = {view.person_id for view in person_views}
            lock_start_times = {
                person_id: start for person_id, start in lock_start_times.items() if person_id in alive_ids
            }
            for view in person_views:
                view.lock_seconds = now - lock_start_times.setdefault(view.person_id, now)

            instant_fps = 1 / (now - previous_time) if now > previous_time else 0.0
            fps = instant_fps if fps is None else (1 - FPS_SMOOTHING) * fps + FPS_SMOOTHING * instant_fps
            previous_time = now

            draw_hud(processing_frame, person_views, fps)
            cv2.imshow(WINDOW_NAME, processing_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        object_detector.close()
        hand_analyzer.close()
        detector.close()
        capture.release()
        cv2.destroyAllWindows()


def main():
    """명령행 인자(--camera)를 읽어 HUD 루프를 시작한다."""
    parser = argparse.ArgumentParser(description="접근 전 안전 확인 HUD (DANGER / UNKNOWN / SURRENDER)")
    parser.add_argument("--camera", choices=("mac", "jetson"), default="mac", help="카메라 백엔드")
    parser.add_argument("--camera-index", type=int, default=0, help="mac 백엔드 장치 번호 (내장 0, 아이폰 연속성 카메라 등 1~)")
    arguments = parser.parse_args()
    run(arguments.camera, arguments.camera_index)


if __name__ == "__main__":
    main()
