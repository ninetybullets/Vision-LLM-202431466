"""손 분석: Pose 손목 위치로 손 crop 박스를 정하고(이미지 조정), crop에 HandLandmarker를 돌려 손가락 관절 각도로 펴짐 여부를 판정한다.

사용: analyzer = HandAnalyzer(); observation = analyzer.analyze(frame_bgr, pose_landmarks, "left", frame_size)
"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from person_detector import VISIBILITY_THRESHOLD, compute_shoulder_width
from preprocess import crop_frame

HAND_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
# 손별 Pose 랜드마크 인덱스 (손목, 검지 뿌리). MediaPipe Pose 규약: 왼손 15·19, 오른손 16·20
HAND_LANDMARK_INDICES = {"left": (15, 19), "right": (16, 20)}
# crop 반폭 = 어깨 너비 × CROP_SCALE. 손을 편 길이가 어깨 너비의 절반 정도이고, 든 물체는 손 밖으로 튀어나오므로 0.9로 넉넉히
CROP_SCALE = 0.9
# 640폭 기준 반폭 하한·상한(px). 너무 작으면 HandLandmarker가 손을 못 찾고, 너무 크면 옆 사람 손까지 들어온다
CROP_MIN_HALF_SIZE = 40
CROP_MAX_HALF_SIZE = 200
# 어깨가 안 보여 너비를 모를 때 쓰는 반폭. 2~3m 거리 손 크기 대략치
DEFAULT_CROP_HALF_SIZE = 60
# 손가락 둘째 마디 각도가 이 값 이상이면 '편 손가락' (Camera_test7.py 값)
OPEN_ANGLE_THRESHOLD = 160
# 각도를 잴 (뿌리, 둘째 마디, 셋째 마디) 손 랜드마크. 검지·중지·약지·소지만. 엄지(1~4)는 각도 기준이 불안정해 제외
FINGER_ANGLE_POINTS = ((5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19))


class HandState(Enum):
    """손 하나의 소분류. threat_classifier가 양손 소분류로 사람 상태를 정한다."""

    HIDDEN = "HIDDEN"  # Pose 손목이 안 보임(몸 뒤·주머니) → 위험
    OUT = "OUT"  # 손목이 화면 밖 → 판단 보류
    NOT_FOUND = "NOT_FOUND"  # 손목은 보이는데 crop에서 손을 못 찾음 → 판단 보류
    CLOSED = "CLOSED"  # 손을 찾았지만 펴지 않음
    OPEN = "OPEN"  # 네 손가락 모두 폄
    OBJECT = "OBJECT"  # 손 박스와 겹치는 객체 검출 있음 → 위험 (held_object 규칙으로 main이 덮어씀)


@dataclass
class HandObservation:
    """손 하나의 분석 결과. crop_box는 (x, y, w, h), locate_hand가 정한 HIDDEN/OUT이면 None(NOT_FOUND에서 바뀐 HIDDEN은 박스 유지).
    object_label·object_contour는 state가 OBJECT일 때 main이 채움(검출기 라벨 참고 표시용, 색 분할 윤곽)."""

    state: HandState
    crop_box: Optional[tuple]
    object_label: Optional[str] = None
    object_contour: Optional[object] = None


def calculate_angle(point1, point2, point3):
    """point2를 꼭짓점으로 하는 세 점의 각도(도)를 돌려준다. 벡터 내적 → arccos. (Camera_test7.py와 동일)"""
    vector1 = np.array(point1, dtype=np.float64) - np.array(point2, dtype=np.float64)
    vector2 = np.array(point3, dtype=np.float64) - np.array(point2, dtype=np.float64)
    cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    # 부동소수 오차로 1.0을 살짝 넘으면 arccos가 nan이 되므로 잘라 준다
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def is_hand_open(hand_landmarks, crop_size):
    """21점 손 랜드마크에서 검지~소지 네 손가락이 모두 OPEN_ANGLE_THRESHOLD 이상 펴져 있으면 True. crop_size는 (폭, 높이).

    정규화 좌표에 crop 크기를 곱해 픽셀로 바꾼 뒤 각도를 잰다. 프레임 가장자리에서 잘린 crop은 정사각형이 아니라
    정규화 좌표 그대로 재면 가로·세로 비율이 달라져 각도가 틀어지기 때문.
    """
    crop_width, crop_height = crop_size
    for base, middle, tip in FINGER_ANGLE_POINTS:
        points = [(hand_landmarks[index].x * crop_width, hand_landmarks[index].y * crop_height) for index in (base, middle, tip)]
        if calculate_angle(*points) < OPEN_ANGLE_THRESHOLD:
            return False
    return True


def compute_crop_half_size(shoulder_width):
    """어깨 너비(px)에 비례한 crop 반폭을 돌려준다. 너비를 모르면 기본값, 알면 하한·상한으로 자른다."""
    if shoulder_width is None:
        return DEFAULT_CROP_HALF_SIZE
    return int(min(max(shoulder_width * CROP_SCALE, CROP_MIN_HALF_SIZE), CROP_MAX_HALF_SIZE))


def locate_hand(landmarks, side, frame_size):
    """손목 상태를 먼저 가리고, 괜찮으면 손 crop 박스를 돌려준다. 반환 (HandState 또는 None, (x, y, w, h) 또는 None).

    손목 정규화 좌표가 0~1 밖 → (OUT, None). 손목 visibility < 임계값 → (HIDDEN, None).
    화면 밖 검사를 먼저 하는 이유: MediaPipe는 화면 밖 랜드마크도 visibility를 낮게 주므로, 순서를 바꾸면 화면 밖 손이 전부 HIDDEN(위험)이 된다.
    그 외 → (None, 박스). 박스 중심은 손목·검지 중점(손은 손목 바깥쪽으로 뻗으므로), 검지가 안 보이면 손목.
    중심은 프레임 안으로 고정하고 박스는 프레임 경계 안으로 잘린다.
    """
    width, height = frame_size
    wrist_index, index_finger_index = HAND_LANDMARK_INDICES[side]
    wrist = landmarks[wrist_index]
    if not (0.0 <= wrist.x <= 1.0 and 0.0 <= wrist.y <= 1.0):
        return HandState.OUT, None
    if wrist.visibility < VISIBILITY_THRESHOLD:
        return HandState.HIDDEN, None

    center_x, center_y = int(wrist.x * width), int(wrist.y * height)
    index_finger = landmarks[index_finger_index]
    if index_finger.visibility >= VISIBILITY_THRESHOLD:
        center_x = (center_x + int(index_finger.x * width)) // 2
        center_y = (center_y + int(index_finger.y * height)) // 2
    # 검지가 화면 밖이면 중점이 프레임 밖으로 나가 음수 폭 박스가 생길 수 있어 중심을 프레임 안으로 고정
    center_x, center_y = min(max(center_x, 0), width - 1), min(max(center_y, 0), height - 1)

    half_size = compute_crop_half_size(compute_shoulder_width(landmarks, frame_size))
    left, top = max(0, center_x - half_size), max(0, center_y - half_size)
    right, bottom = min(width, center_x + half_size), min(height, center_y + half_size)
    return None, (left, top, right - left, bottom - top)


class HandAnalyzer:
    """HandLandmarker 래퍼. 한 프레임에서 사람 수 × 2개의 crop을 따로 넣어야 하므로 타임스탬프가 필요 없는 IMAGE 모드를 쓴다
    (VIDEO 모드는 같은 타임스탬프를 두 번 받지 못함)."""

    def __init__(self, model_path=HAND_MODEL_PATH):
        """HandLandmarker를 IMAGE 모드로 연다(crop마다 독립 추론)."""
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,  # crop 하나에 손 하나만 기대
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def analyze(self, frame_bgr, landmarks, side, frame_size):
        """Pose 랜드마크의 side('left'|'right') 손을 분석해 HandObservation을 돌려준다."""
        state, crop_box = locate_hand(landmarks, side, frame_size)
        if state is not None:
            return HandObservation(state, None)
        crop = crop_frame(frame_bgr, crop_box)
        if crop.size == 0:
            return HandObservation(HandState.NOT_FOUND, crop_box)
        # MediaPipe는 RGB 입력을 기대하므로 OpenCV BGR을 변환한다 (색공간 변환)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return HandObservation(HandState.NOT_FOUND, crop_box)
        crop_size = (crop.shape[1], crop.shape[0])
        state = HandState.OPEN if is_hand_open(result.hand_landmarks[0], crop_size) else HandState.CLOSED
        return HandObservation(state, crop_box)

    def close(self):
        """모델 리소스 해제."""
        self._landmarker.close()
