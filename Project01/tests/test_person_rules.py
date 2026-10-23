"""person_detector 순수 규칙 검증: 합성 랜드마크로 HUMAN 판정·몸 중심·어깨 너비 계산 확인."""
import unittest
from collections import namedtuple

from person_detector import (
    MIN_SHOULDER_WIDTH,
    MIN_VISIBLE_LANDMARKS,
    VISIBILITY_THRESHOLD,
    compute_body_center,
    compute_shoulder_width,
    get_visible_points_by_index,
    is_human,
)

# MediaPipe NormalizedLandmark와 같은 필드만 가진 가짜 랜드마크
FakeLandmark = namedtuple("FakeLandmark", "x y visibility")
FRAME_SIZE = (640, 360)  # (폭, 높이)


def make_landmarks(visible_count, x=0.5, y=0.5):
    """33개 랜드마크 중 앞 visible_count개만 visibility 0.9, 나머지는 0.1인 목록을 만든다."""
    return [FakeLandmark(x, y, 0.9 if index < visible_count else 0.1) for index in range(33)]


def make_person(shoulder_width_pixels, visible_count=MIN_VISIBLE_LANDMARKS):
    """어깨 11·12가 FRAME_SIZE 기준 shoulder_width_pixels 떨어져 보이는 랜드마크(앞 visible_count개도 보임)."""
    landmarks = make_landmarks(visible_count)
    half_width = shoulder_width_pixels / FRAME_SIZE[0] / 2
    landmarks[11] = FakeLandmark(0.5 - half_width, 0.3, 0.9)
    landmarks[12] = FakeLandmark(0.5 + half_width, 0.3, 0.9)
    return landmarks


class IsHumanTest(unittest.TestCase):
    def test_enough_visible_landmarks_and_wide_shoulders_is_human(self):
        self.assertTrue(is_human(make_person(MIN_SHOULDER_WIDTH), FRAME_SIZE))

    def test_too_few_visible_landmarks_is_not_human(self):
        self.assertFalse(is_human(make_landmarks(MIN_VISIBLE_LANDMARKS - 1), FRAME_SIZE))

    def test_tiny_shoulder_width_is_not_human(self):
        # 책상 위 마우스를 사람으로 잡은 사례: 관절은 20개 이상 "보여도" 어깨 너비가 10~39px
        self.assertFalse(is_human(make_person(MIN_SHOULDER_WIDTH - 1, visible_count=25), FRAME_SIZE))

    def test_hidden_shoulders_with_enough_landmarks_is_still_human(self):
        # 얼굴만 보이는 부분 가림(test1 요구): 어깨 너비를 모를 땐 크기 검사를 건너뛴다
        self.assertTrue(is_human(make_landmarks(MIN_VISIBLE_LANDMARKS), FRAME_SIZE))

    def test_no_landmarks_is_not_human(self):
        self.assertFalse(is_human(None, FRAME_SIZE))


class GetVisiblePointsByIndexTest(unittest.TestCase):
    def test_normalized_coordinates_scaled_to_pixels_and_hidden_dropped(self):
        landmarks = [FakeLandmark(0.25, 0.5, 0.9), FakeLandmark(0.75, 0.5, VISIBILITY_THRESHOLD - 0.01)]
        self.assertEqual(get_visible_points_by_index(landmarks, FRAME_SIZE), {0: (160, 180)})


class ComputeBodyCenterTest(unittest.TestCase):
    def test_center_is_shoulder_midpoint(self):
        landmarks = make_landmarks(0)
        # 어깨 11·12만 보이고 엉덩이 23·24는 가려진 상황
        landmarks[11] = FakeLandmark(0.25, 0.5, 0.9)
        landmarks[12] = FakeLandmark(0.75, 0.5, 0.9)
        self.assertEqual(compute_body_center(landmarks, FRAME_SIZE), (320, 180))

    def test_flickering_hip_does_not_move_center_when_shoulders_visible(self):
        # 앉은 자세에서 엉덩이가 보였다 안 보였다 해도 어깨가 보이면 중심은 그대로 (ID 매칭이 깨지던 원인)
        landmarks = make_landmarks(0)
        landmarks[11] = FakeLandmark(0.25, 0.5, 0.9)
        landmarks[12] = FakeLandmark(0.75, 0.5, 0.9)
        landmarks[24] = FakeLandmark(0.6, 0.9, 0.9)
        self.assertEqual(compute_body_center(landmarks, FRAME_SIZE), (320, 180))

    def test_one_shoulder_hidden_falls_back_to_visible_torso_mean(self):
        landmarks = make_landmarks(0)
        landmarks[11] = FakeLandmark(0.25, 0.5, 0.9)
        landmarks[23] = FakeLandmark(0.25, 1.0, 0.9)  # y 평균 0.75 → 270px (부동소수 오차 없는 값)
        self.assertEqual(compute_body_center(landmarks, FRAME_SIZE), (160, 270))

    def test_hidden_torso_falls_back_to_mean_of_visible_landmarks(self):
        landmarks = make_landmarks(0)
        landmarks[0] = FakeLandmark(0.5, 0.25, 0.9)  # 코만 보이는 상황(얼굴만 노출)
        self.assertEqual(compute_body_center(landmarks, FRAME_SIZE), (320, 90))

    def test_nothing_visible_returns_none(self):
        self.assertIsNone(compute_body_center(make_landmarks(0), FRAME_SIZE))


class ComputeShoulderWidthTest(unittest.TestCase):
    def test_width_is_pixel_distance_between_shoulders(self):
        landmarks = make_landmarks(0)
        landmarks[11] = FakeLandmark(0.4, 0.3, 0.9)
        landmarks[12] = FakeLandmark(0.6, 0.3, 0.9)
        self.assertAlmostEqual(compute_shoulder_width(landmarks, FRAME_SIZE), 128.0, places=3)

    def test_hidden_shoulder_returns_none(self):
        landmarks = make_landmarks(0)
        landmarks[11] = FakeLandmark(0.4, 0.3, 0.9)
        self.assertIsNone(compute_shoulder_width(landmarks, FRAME_SIZE))


if __name__ == "__main__":
    unittest.main()
