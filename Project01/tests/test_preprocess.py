"""preprocess 모듈 검증: 축소 크기·LAB 변환·L 채널 CLAHE가 a/b 채널을 건드리지 않는지 확인."""
import unittest

import cv2
import numpy as np

from preprocess import PROCESS_WIDTH, convert_to_lab, enhance_lightness, resize_for_processing

# 합성 프레임용 색. 파랑은 무기 모형 색, 피부는 손 배경 대용
BLUE_BGR = (255, 0, 0)
SKIN_BGR = (105, 172, 224)


def make_frame(background_bgr, patch=None, size=(1280, 720)):
    """단색 배경에 선택적으로 (x, y, w, h, bgr) 사각형을 그린 프레임을 만든다. size는 (폭, 높이)."""
    width, height = size
    frame = np.full((height, width, 3), background_bgr, dtype=np.uint8)
    if patch is not None:
        x, y, patch_width, patch_height, color = patch
        cv2.rectangle(frame, (x, y), (x + patch_width, y + patch_height), color, cv2.FILLED)
    return frame


class ResizeForProcessingTest(unittest.TestCase):
    def test_width_becomes_process_width_and_aspect_kept(self):
        resized = resize_for_processing(make_frame(BLUE_BGR))
        self.assertEqual(resized.shape[1], PROCESS_WIDTH)
        self.assertEqual(resized.shape[0], PROCESS_WIDTH * 720 // 1280)


class LabPipelineTest(unittest.TestCase):
    def test_convert_to_lab_keeps_shape_and_dtype(self):
        lab = convert_to_lab(make_frame(BLUE_BGR, size=(64, 48)))
        self.assertEqual(lab.shape, (48, 64, 3))
        self.assertEqual(lab.dtype, np.uint8)

    def test_enhance_lightness_stretches_l_and_keeps_a_b(self):
        # 100~130 사이 좁은 밝기 그라데이션(저대비). CLAHE는 L의 대비(표준편차)를 키우고 a/b는 그대로 둬야 한다
        gray_levels = np.linspace(100, 130, 64, dtype=np.uint8)
        frame = np.repeat(np.repeat(gray_levels[None, :, None], 48, axis=0), 3, axis=2)
        lab = convert_to_lab(frame)
        enhanced = enhance_lightness(lab)
        self.assertGreater(enhanced[:, :, 0].std(), lab[:, :, 0].std())
        np.testing.assert_array_equal(enhanced[:, :, 1], lab[:, :, 1])
        np.testing.assert_array_equal(enhanced[:, :, 2], lab[:, :, 2])


if __name__ == "__main__":
    unittest.main()
