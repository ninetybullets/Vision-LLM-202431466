"""held_object 검증: 검출 박스↔손 박스 겹침 규칙, 검출 박스 안 색 분할 윤곽(합성 이미지), 격프레임 캐시 (실제 모델은 smoke에서)."""
import unittest

import cv2
import numpy as np

from held_object import (
    DETECT_EVERY_FRAMES,
    HAND_BOX_GROW,
    MIN_OVERLAP_RATIO,
    DetectedObject,
    HeldObjectDetector,
    does_overlap_hand,
    find_object_contour,
)
from tests.test_preprocess import BLUE_BGR, make_frame

GRAY_BGR = (128, 128, 128)


class DoesOverlapHandTest(unittest.TestCase):
    def test_detection_inside_hand_box_overlaps(self):
        self.assertTrue(does_overlap_hand((110, 110, 20, 20), (100, 100, 60, 60)))

    def test_detection_far_away_does_not_overlap(self):
        self.assertFalse(does_overlap_hand((400, 100, 20, 20), (100, 100, 60, 60)))

    def test_detection_partly_inside_hand_box_overlaps_only_above_ratio(self):
        # 손 박스 (100,100,60,60) = 100~160. 검출 박스 140~170은 2/3 겹침 → True, 161~191은 안 겹침 → False
        hand_box = (100, 100, 60, 60)
        self.assertTrue(does_overlap_hand((140, 120, 30, 30), hand_box))
        self.assertFalse(does_overlap_hand((161, 120, 30, 30), hand_box))

    def test_no_hand_box_never_overlaps(self):
        self.assertFalse(does_overlap_hand((110, 110, 20, 20), None))

    def test_constants_are_sane(self):
        # 책상 위 물건(손 아래)이 겹치지 않도록 손 박스를 키우지 않는다(실측 후 결정)
        self.assertEqual(HAND_BOX_GROW, 1.0)
        self.assertGreater(MIN_OVERLAP_RATIO, 0.0)


class FindObjectContourTest(unittest.TestCase):
    def test_blue_block_inside_detection_box_gives_contour_at_block(self):
        # 회색 배경, 검출 박스 (100,100,100,100) 안에 파란 사각형 (120,120)~(180,180)
        frame = make_frame(GRAY_BGR, (120, 120, 60, 60, BLUE_BGR), size=(320, 240))
        contour = find_object_contour(frame, (100, 100, 100, 100))
        self.assertIsNotNone(contour)
        moments = cv2.moments(contour)
        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        self.assertAlmostEqual(center[0], 150, delta=3)  # 프레임 좌표계로 돌려놓았는지
        self.assertAlmostEqual(center[1], 150, delta=3)
        self.assertAlmostEqual(cv2.contourArea(contour), 60 * 60, delta=60 * 60 * 0.15)

    def test_box_outside_frame_is_clipped_not_crashing(self):
        frame = make_frame(GRAY_BGR, size=(320, 240))
        self.assertIsNotNone(find_object_contour(frame, (300, 220, 100, 100)))

    def test_empty_box_returns_none(self):
        frame = make_frame(GRAY_BGR, size=(320, 240))
        self.assertIsNone(find_object_contour(frame, (330, 250, 10, 10)))


class FakeLandmarker:
    """detect_for_video 호출 횟수를 세는 가짜 ObjectDetector."""

    def __init__(self):
        self.calls = 0

    def detect_for_video(self, image, timestamp_ms):
        self.calls += 1

        class Category:
            category_name, score = "book", 0.5

        class BoundingBox:
            origin_x, origin_y, width, height = 10, 10, 20, 20

        class Detection:
            categories = [Category()]
            bounding_box = BoundingBox()

        class Result:
            detections = [Detection()]

        return Result()


class HeldObjectDetectorCadenceTest(unittest.TestCase):
    def test_model_runs_every_n_frames_and_result_is_cached_between(self):
        detector = HeldObjectDetector.__new__(HeldObjectDetector)
        detector._detector = FakeLandmarker()
        detector._last_detections = []
        frame = make_frame(GRAY_BGR, size=(64, 48))
        results = [detector.detect(frame, timestamp_ms=index, frame_index=index) for index in range(DETECT_EVERY_FRAMES * 2)]
        self.assertEqual(detector._detector.calls, 2)
        self.assertTrue(all(results))  # 모델을 안 돌린 프레임도 직전 결과를 돌려준다
        self.assertEqual(results[0][0], DetectedObject("book", 0.5, (10, 10, 20, 20)))

    def test_stale_cache_is_not_returned_after_frames_without_detection(self):
        # 사람이 없어 detect()가 한동안 안 불리면(캐시가 DETECT_EVERY_FRAMES보다 오래됨) 옛 박스 대신 빈 목록
        detector = HeldObjectDetector.__new__(HeldObjectDetector)
        detector._detector = FakeLandmarker()
        detector._last_detections = []
        detector._last_run_frame_index = -1
        frame = make_frame(GRAY_BGR, size=(64, 48))
        self.assertTrue(detector.detect(frame, timestamp_ms=0, frame_index=DETECT_EVERY_FRAMES))
        self.assertTrue(detector.detect(frame, timestamp_ms=1, frame_index=DETECT_EVERY_FRAMES + 1))  # 바로 다음 프레임은 캐시
        stale_index = DETECT_EVERY_FRAMES * 5 + 1  # 모델을 안 돌리는 프레임 번호, 마지막 추론에서 한참 뒤
        self.assertEqual(detector.detect(frame, timestamp_ms=2, frame_index=stale_index), [])
        self.assertEqual(detector._detector.calls, 1)


if __name__ == "__main__":
    unittest.main()
