"""Motion Level 계산(NumPy 경로) 자동 테스트 — seam (c).

실행: 코드 폴더에서 python -m unittest tests.test_motion
"""

import unittest

import numpy as np

from motion import compute_motion_level

# 테스트용 작은 해상도(폭, 높이). 값은 평균이라 해상도와 무관해서 빠르게 돌도록 작게 잡는다.
TEST_FRAME_SIZE = (64, 48)


def make_frame(background_bgr, size=TEST_FRAME_SIZE):
    """단색 BGR uint8 프레임을 만든다. 입력: (B, G, R) 값, (폭, 높이). 출력: (높이, 폭, 3) 배열."""
    width, height = size
    return np.full((height, width, 3), background_bgr, dtype=np.uint8)


class ComputeMotionLevelTest(unittest.TestCase):
    def test_same_frames_give_zero(self):
        frame = make_frame((40, 80, 120))
        self.assertEqual(compute_motion_level(frame, frame.copy()), 0.0)

    def test_all_zero_vs_all_ten_gives_ten(self):
        self.assertEqual(compute_motion_level(make_frame((0, 0, 0)), make_frame((10, 10, 10))), 10.0)

    def test_bigger_difference_gives_bigger_level(self):
        black = make_frame((0, 0, 0))
        small_change = compute_motion_level(black, make_frame((10, 10, 10)))
        big_change = compute_motion_level(black, make_frame((50, 50, 50)))
        self.assertGreater(big_change, small_change)

    def test_no_previous_frame_gives_zero(self):
        self.assertEqual(compute_motion_level(None, make_frame((10, 10, 10))), 0.0)

    def test_gray_is_channel_mean(self):
        # 한 채널만 30 다르면 채널 평균 흑백에서는 30 / 3 = 10 차이
        self.assertEqual(compute_motion_level(make_frame((0, 0, 0)), make_frame((0, 30, 0))), 10.0)

    def test_returns_python_float(self):
        level = compute_motion_level(make_frame((0, 0, 0)), make_frame((10, 10, 10)))
        self.assertIs(type(level), float)
        self.assertIs(type(compute_motion_level(None, make_frame((0, 0, 0)))), float)


if __name__ == "__main__":
    unittest.main()
