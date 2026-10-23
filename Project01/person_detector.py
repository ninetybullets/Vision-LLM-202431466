"""신체 인식: MediaPipe PoseLandmarker로 여러 사람의 33개 관절을 찾고, 보이는 관절 수와 어깨 너비로 '사람' 여부를 판정한다.

사용: detector = PersonDetector(); poses = detector.detect(frame_bgr, timestamp_ms)  # 사람별 랜드마크 목록의 리스트
"""
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# 모델 파일은 이 파일과 같은 폴더의 models/ 아래. __file__ 기준 절대경로라 어느 cwd에서 실행해도 찾는다
POSE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task"
# 동시에 인식할 최대 인원. 늘릴수록 Pose·Hand 추론이 비례해 늘어 fps가 떨어진다 (Jetson 측정 후 조정)
MAX_PERSONS = 3
# visibility: 해당 관절이 가려지지 않고 보일 확률(0~1). 0.5 미만이면 '안 보임'으로 취급
VISIBILITY_THRESHOLD = 0.5
# 보이는 관절이 이 개수 이상이면 사람으로 판정. 얼굴 5점(코·눈·귀 일부)만 보여도 통과하도록 5로 둠 → 앉거나 부분 가림 대응
MIN_VISIBLE_LANDMARKS = 5
# 어깨 너비(px, 640폭 기준)가 이보다 작으면 사람으로 안 봄. 책상 위 마우스·케이블에 Pose가 관절 20개 이상을 찍은 사례에서
# 어깨 너비가 10~39px였고, 실제 사람은 2~3m에서 약 80~150px. 손 crop 최소 반폭(40)과 같은 값
MIN_SHOULDER_WIDTH = 40
# 몸 중심 계산에 쓰는 관절: 왼/오른 어깨(11·12), 왼/오른 엉덩이(23·24). MediaPipe Pose 인덱스 규약
TORSO_LANDMARK_INDICES = (11, 12, 23, 24)
SHOULDER_INDICES = (11, 12)
# 화면에 그릴 뼈대 연결선 (MediaPipe가 제공하는 표준 연결 목록)
POSE_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS


def is_human(landmarks, frame_size):
    """visibility ≥ 임계값인 관절이 MIN_VISIBLE_LANDMARKS개 이상이고, 어깨 너비가 MIN_SHOULDER_WIDTH 이상이면 True.

    어깨가 안 보여 너비를 모르면(얼굴만 보이는 부분 가림) 크기 검사는 건너뛴다. landmarks가 None이면 False.
    """
    if not landmarks:
        return False
    visible_count = sum(1 for landmark in landmarks if landmark.visibility >= VISIBILITY_THRESHOLD)
    if visible_count < MIN_VISIBLE_LANDMARKS:
        return False
    shoulder_width = compute_shoulder_width(landmarks, frame_size)
    return shoulder_width is None or shoulder_width >= MIN_SHOULDER_WIDTH


def get_visible_points_by_index(landmarks, frame_size):
    """보이는 관절만 {랜드마크 인덱스: (x, y) 픽셀 좌표}로 바꾼다. frame_size는 (폭, 높이).

    정규화 좌표(0~1)에 프레임 크기를 곱한다. 인덱스를 남기는 이유는 뼈대 연결선을 그릴 때 필요해서.
    """
    width, height = frame_size
    return {
        index: (int(landmark.x * width), int(landmark.y * height))
        for index, landmark in enumerate(landmarks)
        if landmark.visibility >= VISIBILITY_THRESHOLD
    }


def compute_body_center(landmarks, frame_size):
    """Kalman 측정값이 될 몸 중심 픽셀 좌표를 돌려준다. 보이는 관절이 하나도 없으면 None.

    1순위: 양 어깨 중점. 앉은 자세에서 엉덩이(23·24)가 보였다 안 보였다 하면 '어깨 평균 ↔ 어깨+엉덩이 평균' 사이를
           약 100px씩 점프해 ID 매칭(100px)이 깨지는 것을 실측으로 확인 → 거의 항상 보이고 안정적인 어깨만 쓴다.
    2순위: 어깨 한쪽이라도 안 보이면 어깨·엉덩이 중 보이는 관절 평균.
    3순위: 몸통이 전부 가려지면(얼굴만 보일 때 등) 보이는 관절 전체 평균 → 사람 판정과 lock이 어긋나지 않게.
    """
    width, height = frame_size
    shoulders = [landmarks[index] for index in SHOULDER_INDICES if landmarks[index].visibility >= VISIBILITY_THRESHOLD]
    torso = [landmarks[index] for index in TORSO_LANDMARK_INDICES if landmarks[index].visibility >= VISIBILITY_THRESHOLD]
    anchors = (shoulders if len(shoulders) == len(SHOULDER_INDICES) else torso) or [
        landmark for landmark in landmarks if landmark.visibility >= VISIBILITY_THRESHOLD
    ]
    if not anchors:
        return None
    center_x = sum(landmark.x for landmark in anchors) / len(anchors) * width
    center_y = sum(landmark.y for landmark in anchors) / len(anchors) * height
    return int(center_x), int(center_y)


def compute_shoulder_width(landmarks, frame_size):
    """양 어깨(11·12) 사이 픽셀 거리. 손 crop 크기의 기준(사람이 멀면 작게, 가까우면 크게). 한쪽이라도 안 보이면 None."""
    width, height = frame_size
    left, right = (landmarks[index] for index in SHOULDER_INDICES)
    if left.visibility < VISIBILITY_THRESHOLD or right.visibility < VISIBILITY_THRESHOLD:
        return None
    return (((left.x - right.x) * width) ** 2 + ((left.y - right.y) * height) ** 2) ** 0.5


class PersonDetector:
    """PoseLandmarker 래퍼. VIDEO 모드라 프레임마다 단조 증가하는 timestamp_ms가 필요하다."""

    def __init__(self, model_path=POSE_MODEL_PATH, max_persons=MAX_PERSONS):
        """PoseLandmarker를 VIDEO 모드로 연다. max_persons는 한 프레임에서 찾을 최대 인원."""
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max_persons,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr, timestamp_ms):
        """BGR 프레임에서 찾은 사람들의 33개 랜드마크 목록 리스트를 돌려준다. 없으면 빈 리스트."""
        # MediaPipe는 RGB 입력을 기대하므로 OpenCV BGR을 변환한다 (색공간 변환)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        return list(result.pose_landmarks)

    def close(self):
        """모델 리소스 해제."""
        self._landmarker.close()
