"""hand_analyzer 순수 규칙 검증: 관절 각도, 손 펴짐, 손목 기준 crop 위치·크기·경계 (HandLandmarker 모델 불필요)."""
import unittest
from collections import namedtuple

from hand_analyzer import (
    CROP_MAX_HALF_SIZE,
    CROP_MIN_HALF_SIZE,
    DEFAULT_CROP_HALF_SIZE,
    HandState,
    calculate_angle,
    compute_crop_half_size,
    is_hand_open,
    locate_hand,
)

FakeLandmark = namedtuple("FakeLandmark", "x y visibility")
FakeHandPoint = namedtuple("FakeHandPoint", "x y")
FRAME_SIZE = (640, 360)
CROP_SIZE = (100, 100)  # 정사각형 crop이면 정규화 좌표와 픽셀 좌표의 각도가 같다


def make_hand(is_open):
    """21점 손 랜드마크. 펴진 손은 네 손가락이 직선, 주먹은 둘째 마디에서 꺾임. 엄지(1~4)는 항상 꺾어 둬 무시되는지 확인."""
    points = [FakeHandPoint(0.5, 0.9)]
    points += [FakeHandPoint(0.3, 0.8), FakeHandPoint(0.25, 0.75), FakeHandPoint(0.3, 0.7), FakeHandPoint(0.35, 0.72)]
    for base_x in (0.4, 0.47, 0.54, 0.61):
        if is_open:
            points += [FakeHandPoint(base_x, 0.7 - 0.1 * joint) for joint in range(4)]
        else:
            points += [FakeHandPoint(base_x, 0.7), FakeHandPoint(base_x, 0.6), FakeHandPoint(base_x + 0.05, 0.65), FakeHandPoint(base_x + 0.05, 0.7)]
    return points


def make_pose(wrist=(0.5, 0.5, 0.9), index=(0.55, 0.45, 0.9), shoulders=((0.4, 0.3, 0.9), (0.6, 0.3, 0.9))):
    """왼손(15·19)과 어깨(11·12)만 채운 33개 포즈 랜드마크."""
    landmarks = [FakeLandmark(0.5, 0.5, 0.1) for _ in range(33)]
    landmarks[11], landmarks[12] = FakeLandmark(*shoulders[0]), FakeLandmark(*shoulders[1])
    landmarks[15], landmarks[19] = FakeLandmark(*wrist), FakeLandmark(*index)
    return landmarks


class CalculateAngleTest(unittest.TestCase):
    def test_straight_line_is_180(self):
        self.assertAlmostEqual(calculate_angle((0, 0), (1, 0), (2, 0)), 180.0, places=3)

    def test_right_angle_is_90(self):
        self.assertAlmostEqual(calculate_angle((0, 1), (0, 0), (1, 0)), 90.0, places=3)


class IsHandOpenTest(unittest.TestCase):
    def test_straight_fingers_are_open_even_with_bent_thumb(self):
        self.assertTrue(is_hand_open(make_hand(is_open=True), CROP_SIZE))

    def test_bent_fingers_are_closed(self):
        self.assertFalse(is_hand_open(make_hand(is_open=False), CROP_SIZE))

    def test_angles_measured_in_pixels_not_normalized(self):
        # 손가락이 정규화 좌표에서는 살짝 꺾여(≈153°) 보여도, 세로로 긴 crop에서는 픽셀 기준 거의 직선(≥160°)이다
        points = make_hand(is_open=True)
        points[7] = FakeHandPoint(0.4 + 0.05, 0.5)  # 검지 셋째 마디를 옆으로 조금 밀어 둠
        self.assertFalse(is_hand_open(points, (100, 100)))
        self.assertTrue(is_hand_open(points, (100, 400)))


class ComputeCropHalfSizeTest(unittest.TestCase):
    def test_scales_with_shoulder_width(self):
        self.assertEqual(compute_crop_half_size(100), 90)

    def test_clamped_to_min_and_max(self):
        self.assertEqual(compute_crop_half_size(10), CROP_MIN_HALF_SIZE)
        self.assertEqual(compute_crop_half_size(1000), CROP_MAX_HALF_SIZE)

    def test_unknown_shoulder_width_uses_default(self):
        self.assertEqual(compute_crop_half_size(None), DEFAULT_CROP_HALF_SIZE)


class LocateHandTest(unittest.TestCase):
    def test_hidden_wrist(self):
        state, box = locate_hand(make_pose(wrist=(0.5, 0.5, 0.2)), "left", FRAME_SIZE)
        self.assertEqual((state, box), (HandState.HIDDEN, None))

    def test_wrist_outside_frame_is_out_even_with_low_visibility(self):
        # MediaPipe는 화면 밖 랜드마크의 visibility도 낮게 주므로, 화면 밖은 HIDDEN이 아니라 OUT이어야 한다
        state, box = locate_hand(make_pose(wrist=(1.2, 0.5, 0.1)), "left", FRAME_SIZE)
        self.assertEqual((state, box), (HandState.OUT, None))

    def test_index_finger_outside_frame_still_gives_positive_box(self):
        _, box = locate_hand(make_pose(wrist=(0.0, 0.5, 0.9), index=(-0.3, 0.5, 0.9)), "left", FRAME_SIZE)
        x, y, width, height = box
        self.assertEqual(x, 0)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)

    def test_box_centered_between_wrist_and_index_with_shoulder_scaled_size(self):
        # 어깨 0.4~0.6 × 640 = 128px → 반폭 int(0.9×128)=115. 손목 (320,180)·검지 (352,162) 중점 (336,171)
        state, box = locate_hand(make_pose(), "left", FRAME_SIZE)
        self.assertIsNone(state)
        self.assertEqual(box, (336 - 115, 171 - 115, 230, 230))

    def test_box_uses_wrist_when_index_hidden(self):
        _, box = locate_hand(make_pose(index=(0.55, 0.45, 0.1)), "left", FRAME_SIZE)
        self.assertEqual(box, (320 - 115, 180 - 115, 230, 230))

    def test_box_clipped_to_frame_edge(self):
        _, box = locate_hand(make_pose(wrist=(0.02, 0.5, 0.9), index=(0.02, 0.5, 0.9)), "left", FRAME_SIZE)
        x, y, width, height = box
        self.assertEqual(x, 0)
        self.assertEqual(width, 12 + 115)


if __name__ == "__main__":
    unittest.main()
