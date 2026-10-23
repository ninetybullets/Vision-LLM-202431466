"""손에 든 물체 판별: MediaPipe ObjectDetector(EfficientDet-Lite0, COCO)를 전체 프레임에 돌려 'person이 아닌 검출'이 손 박스와 겹치면
물체를 든 것으로 본다. 검출 박스 안은 LAB 색 분할(Color Segmentation)로 물체 윤곽(Contour Detection)을 뽑아 HUD에 그린다.

왜 클래스를 안 보나: 실측(README「왜 객체 검출기인데 클래스를 안 보나」)에서 펜→toothbrush, 텀블러→cell phone처럼 이름은 거의 틀렸지만
"뭔가 있다"는 잘 잡혔다. 접근 전 안전 판단엔 "손에 뭔가 들렸는가"면 충분하다.

사용: detector = HeldObjectDetector(); detections = detector.detect(frame_bgr, timestamp_ms, frame_index)
      does_overlap_hand(detection.box, hand_crop_box) → True면 그 손은 OBJECT
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from preprocess import blur_frame, convert_to_lab, crop_frame, enhance_lightness

OBJECT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "efficientdet_lite0.tflite"
# 검출 점수 하한. 실측: 0.2면 빈손 오검출 8~10%, 0.3이면 2%지만 펜 검출률이 84%→49%로 떨어짐. 그 사이 값
MIN_DETECTION_SCORE = 0.25
# 프레임당 최대 검출 수. 사람 3명 + 손에 든 물체 몇 개면 충분
MAX_DETECTIONS = 6
# 이 라벨은 물체가 아니므로 무시
IGNORED_LABEL = "person"
# 모델은 N프레임마다 한 번만 돌리고 사이 프레임은 직전 결과를 쓴다. 1회 약 37ms(맥 M2)라 매 프레임 돌리면 fps가 반토막
DETECT_EVERY_FRAMES = 2
# 손 crop 박스를 이 배수로 키운 영역과 검출 박스가 겹치면 '손 근처'. 1.0 = 확대 없음. 1.5로 두니 손을 내렸을 때 책상 위 마우스·컵까지
# 겹쳐 DANGER가 떠서(실측 59%) 키우지 않는다. crop 자체가 이미 어깨 너비 0.9배라 든 물체는 충분히 들어온다. 배수는 실험용으로 남김
HAND_BOX_GROW = 1.0
# 검출 박스 면적 중 이 비율 이상이 키운 손 박스 안에 있어야 겹침으로 인정
MIN_OVERLAP_RATIO = 0.2
# 색 분할: 검출 박스 중심부(가로세로 절반)의 a·b 중앙값을 물체 색으로 보고 ±허용폭 안 픽셀만 남긴다.
# 피부(a≈140,b≈148)와 파란 노트(134,110)의 b 차이가 약 38이므로 12면 서로 섞이지 않는다 (실측 후 조정)
COLOR_TOLERANCE = 12
# 아주 어두운 픽셀은 색 정보가 잡음이라 제외
MIN_LIGHTNESS = 40
# open/close로 지울 잡음·메울 구멍 크기(px). 검출 박스가 수십~수백 px라 5px면 점 잡음만 지우고 물체 형태는 남는다
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


@dataclass(frozen=True)
class DetectedObject:
    """검출 하나. box는 프레임 좌표 (x, y, w, h)."""

    label: str
    score: float  # 판정엔 안 쓴다(점수 하한은 모델 옵션에서 걸러짐). 실측 로그용
    box: tuple


def does_overlap_hand(detection_box, hand_box):
    """검출 박스가 HAND_BOX_GROW배 키운 손 박스와 MIN_OVERLAP_RATIO 이상 겹치면 True. hand_box가 None이면 False."""
    if hand_box is None:
        return False
    detection_x, detection_y, detection_width, detection_height = detection_box
    hand_x, hand_y, hand_width, hand_height = hand_box
    center_x, center_y = hand_x + hand_width / 2, hand_y + hand_height / 2
    half_width, half_height = hand_width * HAND_BOX_GROW / 2, hand_height * HAND_BOX_GROW / 2
    overlap_width = max(0.0, min(detection_x + detection_width, center_x + half_width) - max(detection_x, center_x - half_width))
    overlap_height = max(0.0, min(detection_y + detection_height, center_y + half_height) - max(detection_y, center_y - half_height))
    detection_area = detection_width * detection_height
    return detection_area > 0 and overlap_width * overlap_height >= MIN_OVERLAP_RATIO * detection_area


def clip_box(box, frame_size):
    """박스를 프레임 안으로 자른다. 완전히 밖이면 None."""
    width, height = frame_size
    x, y, box_width, box_height = box
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + box_width), min(height, y + box_height)
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def find_object_contour(frame, detection_box):
    """검출 박스 안에서 물체 색을 분할해 가장 큰 윤곽을 프레임 좌표로 돌려준다. 못 찾으면 None.

    흐름: 박스 crop → 블러 → LAB → L CLAHE → 중심부 a·b 중앙값 ± COLOR_TOLERANCE로 inRange → open/close → findContours → 최대 면적.
    검출기 박스는 거칠어 배경이 섞이므로, 색으로 물체 영역만 정밀하게 남기는 단계.
    """
    box = clip_box(detection_box, (frame.shape[1], frame.shape[0]))
    if box is None:
        return None
    x, y, box_width, box_height = box
    lab = enhance_lightness(convert_to_lab(blur_frame(crop_frame(frame, box))))
    center_region = lab[box_height // 4 : box_height - box_height // 4, box_width // 4 : box_width - box_width // 4]
    if center_region.size == 0:
        center_region = lab
    median_a, median_b = np.median(center_region[:, :, 1]), np.median(center_region[:, :, 2])
    lower = np.array([MIN_LIGHTNESS, max(0, median_a - COLOR_TOLERANCE), max(0, median_b - COLOR_TOLERANCE)], dtype=np.uint8)
    upper = np.array([255, min(255, median_a + COLOR_TOLERANCE), min(255, median_b + COLOR_TOLERANCE)], dtype=np.uint8)
    mask = cv2.inRange(lab, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour + np.array([[[x, y]]], dtype=largest_contour.dtype)  # crop 좌표 → 프레임 좌표


class HeldObjectDetector:
    """ObjectDetector 래퍼. VIDEO 모드(Pose와 같은 타임스탬프), DETECT_EVERY_FRAMES마다 한 번 추론하고 사이엔 캐시를 돌려준다."""

    def __init__(self, model_path=OBJECT_MODEL_PATH):
        """EfficientDet 모델을 VIDEO 모드로 연다. 점수 하한·최대 검출 수는 모듈 상수."""
        options = vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            max_results=MAX_DETECTIONS,
            score_threshold=MIN_DETECTION_SCORE,
        )
        self._detector = vision.ObjectDetector.create_from_options(options)
        self._last_detections = []
        self._last_run_frame_index = -1  # 마지막으로 모델을 돌린 프레임 번호. 캐시가 오래됐는지 판단용

    def detect(self, frame_bgr, timestamp_ms, frame_index):
        """person을 뺀 DetectedObject 목록. frame_index가 DETECT_EVERY_FRAMES의 배수일 때만 모델을 돌리고 사이 프레임은 캐시.
        사람이 없어 한동안 호출되지 않았다면(캐시가 DETECT_EVERY_FRAMES보다 오래됨) 옛 박스가 새 손과 겹치지 않도록 빈 목록을 돌려준다."""
        if frame_index % DETECT_EVERY_FRAMES != 0:
            is_cache_fresh = frame_index - self._last_run_frame_index < DETECT_EVERY_FRAMES
            return self._last_detections if is_cache_fresh else []
        self._last_run_frame_index = frame_index
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # MediaPipe는 RGB 입력 (색공간 변환)
        result = self._detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        self._last_detections = [
            DetectedObject(
                detection.categories[0].category_name,
                detection.categories[0].score,
                (detection.bounding_box.origin_x, detection.bounding_box.origin_y, detection.bounding_box.width, detection.bounding_box.height),
            )
            for detection in result.detections
            if detection.categories[0].category_name != IGNORED_LABEL
        ]
        return self._last_detections

    def close(self):
        """모델 리소스 해제."""
        self._detector.close()
