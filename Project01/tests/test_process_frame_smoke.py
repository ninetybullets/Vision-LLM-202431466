"""main.process_frame 통합 smoke: 실제 Pose·Hand·ObjectDetector 모델을 로드해 합성 프레임이 처리되는지, HandLandmarker crop 경로와
HUD 그리기가 도는지 확인 (카메라 불필요)."""
import unittest

from hand_analyzer import HandAnalyzer, HandObservation, HandState
from held_object import HeldObjectDetector
from hud import draw_hud
from main import PersonView, process_frame
from person_detector import PersonDetector
from target_tracker import MultiTargetTracker, TrackState
from tests.test_hand_analyzer import make_pose
from tests.test_preprocess import BLUE_BGR, SKIN_BGR, make_frame
from threat_classifier import ThreatState


class ProcessFrameSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = PersonDetector()
        cls.hand_analyzer = HandAnalyzer()
        cls.object_detector = HeldObjectDetector()

    @classmethod
    def tearDownClass(cls):
        cls.detector.close()
        cls.hand_analyzer.close()
        cls.object_detector.close()

    def test_blue_frame_has_no_person_and_hud_draws(self):
        frame, person_views = process_frame(
            make_frame(BLUE_BGR), self.detector, self.hand_analyzer, self.object_detector, MultiTargetTracker(), {},
            timestamp_ms=1, frame_index=0,
        )
        self.assertEqual(person_views, [])
        draw_hud(frame, person_views, fps=0.0)

    def test_object_detector_runs_on_plain_frame_without_person_label(self):
        detections = self.object_detector.detect(make_frame(SKIN_BGR, size=(640, 360)), timestamp_ms=10, frame_index=0)
        self.assertTrue(all(detection.label != "person" for detection in detections))

    def test_hand_analyzer_runs_landmarker_on_crop_and_reports_not_found(self):
        # 손 없는 단색 crop → HandLandmarker가 손을 못 찾아 NOT_FOUND, 박스는 있어야 한다
        frame = make_frame(SKIN_BGR, size=(640, 360))
        observation = self.hand_analyzer.analyze(frame, make_pose(), "left", (640, 360))
        self.assertEqual(observation.state, HandState.NOT_FOUND)
        self.assertIsNotNone(observation.crop_box)

    def test_hud_draws_predicting_person_with_hand_boxes_and_object_contour(self):
        import numpy as np

        frame = make_frame(SKIN_BGR, size=(640, 360))
        contour = np.array([[[120, 120]], [[160, 120]], [[160, 160]], [[120, 160]]], dtype=np.int32)
        hands = {
            "left": HandObservation(HandState.OBJECT, (100, 100, 80, 80), "book", contour),
            "right": HandObservation(HandState.HIDDEN, None),
        }
        view = PersonView(1, TrackState.PREDICTING, (320, 180), ThreatState.DANGER, {11: (300, 150), 12: (340, 150)}, hands, 2.5)
        draw_hud(frame, [view], fps=20.0)


if __name__ == "__main__":
    unittest.main()
