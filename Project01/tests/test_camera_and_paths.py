"""camera 백엔드 선택과 모델 파일 경로 검증 (실제 카메라는 열지 않음)."""
import unittest

from camera import build_capture_arguments
from hand_analyzer import HAND_MODEL_PATH
from held_object import OBJECT_MODEL_PATH
from person_detector import POSE_MODEL_PATH


class BuildCaptureArgumentsTest(unittest.TestCase):
    def test_mac_backend_uses_camera_index_zero(self):
        self.assertEqual(build_capture_arguments("mac"), (0,))

    def test_mac_backend_passes_through_external_camera_index(self):
        self.assertEqual(build_capture_arguments("mac", camera_index=1), (1,))

    def test_jetson_backend_uses_gstreamer_mjpg_pipeline(self):
        arguments = build_capture_arguments("jetson")
        pipeline, api = arguments
        self.assertIn("v4l2src device=/dev/video0", pipeline)
        self.assertIn("image/jpeg", pipeline)
        self.assertIn("width=1280", pipeline)
        self.assertIn("appsink", pipeline)
        import cv2
        self.assertEqual(api, cv2.CAP_GSTREAMER)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            build_capture_arguments("windows")


class ModelPathTest(unittest.TestCase):
    def test_model_paths_are_absolute_and_exist_regardless_of_cwd(self):
        for path in (POSE_MODEL_PATH, HAND_MODEL_PATH, OBJECT_MODEL_PATH):
            self.assertTrue(path.is_absolute())
            self.assertTrue(path.exists(), f"모델 없음: {path}")


if __name__ == "__main__":
    unittest.main()
