"""HUD overlay: 사람별 조준점·상태 라벨·손 박스·뼈대·예측 점선, 좌상단 fps, 우상단 인원 요약을 프레임 위에 그린다.

사용: draw_hud(frame, person_views, fps)  — person_views는 main.PersonView 목록
"""
import cv2

from person_detector import POSE_CONNECTIONS
from target_tracker import TrackState
from threat_classifier import ThreatState

# BGR 색상. 사람 상태별: 빨강=위험, 노랑=판단 보류, 초록=항복. 뼈대는 흰색 얇게
THREAT_COLORS = {
    ThreatState.DANGER: (0, 0, 255),
    ThreatState.UNKNOWN: (0, 255, 255),
    ThreatState.SURRENDER: (0, 255, 0),
}
COLOR_SKELETON = (255, 255, 255)
COLOR_TEXT = (255, 255, 255)
# 손에 든 물체의 색 분할 윤곽. 상태 색(빨강)과 구분되게 자홍
COLOR_OBJECT_CONTOUR = (255, 0, 255)
# 조준점 반지름(px, 640폭 기준)
CROSSHAIR_RADIUS = 20
# 십자선 팔 길이: 원 밖으로 반지름만큼 더 나가게 지름 길이로
CROSSHAIR_ARM_LENGTH = CROSSHAIR_RADIUS * 2
# PREDICTING 때 그릴 점선 박스 반폭(px, 640폭 기준). 2~3m 거리 몸통 폭 대략치
PREDICT_BOX_HALF_SIZE = 60
# 선 굵기(px): 보조선(뼈대·손 박스·점선)은 얇게, 조준점 원과 사람 라벨은 굵게
THIN_LINE = 1
THICK_LINE = 2
# 점선 한 토막 길이(px). OpenCV에 점선 API가 없어 선분을 이어 그린다
DASH_LENGTH = 10
# 뼈대 관절 점 반지름(px)
JOINT_RADIUS = 2
# 텍스트: 손 라벨은 작게, 사람·상단 상태는 크게. 박스 위 라벨은 4px 띄움, 가장자리 여백 10px, 첫 줄 기준선 y=24
FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.4
STATUS_FONT_SCALE = 0.6
LABEL_OFFSET = 4
MARGIN = 10
STATUS_BASELINE_Y = 24
# 사람 라벨을 조준점 위로 띄우는 거리(px)
PERSON_LABEL_OFFSET = CROSSHAIR_RADIUS * 2 + LABEL_OFFSET


def draw_skeleton(frame, points_by_index):
    """보이는 관절끼리만 연결선을 그린다. points_by_index: {랜드마크 인덱스: (x, y)}."""
    for connection in POSE_CONNECTIONS:
        start = points_by_index.get(connection.start)
        end = points_by_index.get(connection.end)
        if start is not None and end is not None:
            cv2.line(frame, start, end, COLOR_SKELETON, THIN_LINE)
    for point in points_by_index.values():
        cv2.circle(frame, point, JOINT_RADIUS, COLOR_SKELETON, cv2.FILLED)


def draw_crosshair(frame, center, color):
    """중심에 원 + 십자선 조준점을 그린다."""
    x, y = center
    cv2.circle(frame, center, CROSSHAIR_RADIUS, color, THICK_LINE)
    cv2.line(frame, (x - CROSSHAIR_ARM_LENGTH, y), (x + CROSSHAIR_ARM_LENGTH, y), color, THIN_LINE)
    cv2.line(frame, (x, y - CROSSHAIR_ARM_LENGTH), (x, y + CROSSHAIR_ARM_LENGTH), color, THIN_LINE)


def draw_dashed_box(frame, center, color):
    """예측 중임을 나타내는 점선 사각형."""
    x, y = center
    left, top = x - PREDICT_BOX_HALF_SIZE, y - PREDICT_BOX_HALF_SIZE
    right, bottom = x + PREDICT_BOX_HALF_SIZE, y + PREDICT_BOX_HALF_SIZE
    for start in range(left, right, DASH_LENGTH * 2):
        cv2.line(frame, (start, top), (min(start + DASH_LENGTH, right), top), color, THIN_LINE)
        cv2.line(frame, (start, bottom), (min(start + DASH_LENGTH, right), bottom), color, THIN_LINE)
    for start in range(top, bottom, DASH_LENGTH * 2):
        cv2.line(frame, (left, start), (left, min(start + DASH_LENGTH, bottom)), color, THIN_LINE)
        cv2.line(frame, (right, start), (right, min(start + DASH_LENGTH, bottom)), color, THIN_LINE)


def draw_hands(frame, hands, color):
    """손 crop 박스와 소분류 라벨(OPEN/CLOSED/OBJECT/NOT_FOUND)을 그린다. HIDDEN/OUT은 박스가 없어 생략.
    OBJECT면 검출기 라벨을 참고로 덧붙이고(`OBJECT(book?)`, 클래스는 신뢰하지 않음) 색 분할 윤곽을 그린다."""
    for observation in hands.values():
        if observation.crop_box is None:
            continue
        x, y, width, height = observation.crop_box
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, THIN_LINE)
        label = observation.state.value
        if observation.object_label:
            label += f"({observation.object_label}?)"
        cv2.putText(frame, label, (x, y - LABEL_OFFSET), FONT, LABEL_FONT_SCALE, color, THIN_LINE)
        if observation.object_contour is not None:
            cv2.drawContours(frame, [observation.object_contour], -1, COLOR_OBJECT_CONTOUR, THICK_LINE)


def draw_person(frame, view):
    """사람 한 명: 뼈대 → 조준점(상태 색) → 예측 점선 → 손 박스 → `P1 DANGER 3.2s` 라벨."""
    color = THREAT_COLORS[view.threat_state]
    if view.points_by_index:
        draw_skeleton(frame, view.points_by_index)
    draw_crosshair(frame, view.center, color)
    if view.track_state == TrackState.PREDICTING:
        draw_dashed_box(frame, view.center, color)
    draw_hands(frame, view.hands, color)
    label = f"P{view.person_id} {view.threat_state.value}"
    if view.lock_seconds is not None:
        label += f" {view.lock_seconds:.1f}s"
    x, y = view.center
    cv2.putText(frame, label, (x - CROSSHAIR_RADIUS, y - PERSON_LABEL_OFFSET), FONT, STATUS_FONT_SCALE, color, THICK_LINE)


def draw_status(frame, person_views, fps):
    """좌상단 fps, 우상단 `N PERSONS  k DANGER` 요약. 위험 인원이 있으면 빨강, 없으면 흰색."""
    cv2.putText(frame, f"FPS {fps:4.1f}", (MARGIN, STATUS_BASELINE_Y), FONT, STATUS_FONT_SCALE, COLOR_TEXT, THIN_LINE)
    danger_count = sum(1 for view in person_views if view.threat_state == ThreatState.DANGER)
    label = f"{len(person_views)} PERSONS  {danger_count} DANGER"
    color = THREAT_COLORS[ThreatState.DANGER] if danger_count else COLOR_TEXT
    text_width = cv2.getTextSize(label, FONT, STATUS_FONT_SCALE, THICK_LINE)[0][0]
    cv2.putText(frame, label, (frame.shape[1] - text_width - MARGIN, STATUS_BASELINE_Y), FONT, STATUS_FONT_SCALE, color, THICK_LINE)


def draw_hud(frame, person_views, fps):
    """모든 overlay를 한 번에 그린다. frame을 제자리에서 수정한다."""
    for view in person_views:
        draw_person(frame, view)
    draw_status(frame, person_views, fps)
