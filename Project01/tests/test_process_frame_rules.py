"""main.process_frame의 사람별 상태 관리 검증(가짜 검출기 사용): 디바운스 후 SURRENDER, 가림(PREDICTING) 중 상태 유지,
트랙 소멸 시 사람 상태 정리, 손 분석을 홀수 프레임에 한 손씩 번갈아 하는지, 검출 물체가 손과 겹치면 OBJECT→DANGER가 되고 hold 뒤 풀리는지."""
import unittest
from collections import namedtuple

from hand_analyzer import HandObservation, HandState
from held_object import DetectedObject
from main import HAND_SIDES, NOT_FOUND_FRAMES_AS_HIDDEN, OBJECT_HOLD_FRAMES, process_frame
from target_tracker import MAX_PREDICT_FRAMES, MultiTargetTracker, TrackState
from tests.test_preprocess import SKIN_BGR, make_frame
from threat_classifier import STABLE_FRAMES, ThreatState

FakeLandmark = namedtuple("FakeLandmark", "x y visibility")
LEFT_HAND_BOX = (100, 50, 80, 80)
RIGHT_HAND_BOX = (400, 50, 80, 80)


def make_surrender_pose(center_x=0.5):
    """어깨·엉덩이가 보이고(어깨 너비 128px) 양 손목이 어깨 위에 있는 33개 랜드마크(사람 판정 통과)."""
    landmarks = [FakeLandmark(center_x, 0.5, 0.1) for _ in range(33)]
    landmarks[11], landmarks[12] = FakeLandmark(center_x - 0.1, 0.4, 0.9), FakeLandmark(center_x + 0.1, 0.4, 0.9)
    landmarks[23], landmarks[24] = FakeLandmark(center_x - 0.1, 0.7, 0.9), FakeLandmark(center_x + 0.1, 0.7, 0.9)
    landmarks[15], landmarks[16] = FakeLandmark(center_x - 0.2, 0.2, 0.9), FakeLandmark(center_x + 0.2, 0.2, 0.9)
    return landmarks


class FakeDetector:
    """detect()가 미리 정한 포즈 목록을 돌려준다."""

    def __init__(self):
        self.poses = []

    def detect(self, frame, timestamp_ms):
        return self.poses


class FakeHandAnalyzer:
    """손별로 정해 둔 상태(기본 OPEN)를 고정 박스와 함께 돌려준다. 어느 손을 분석했는지 기록한다."""

    def __init__(self):
        self.analyzed_sides = []
        self.states = {"left": HandState.OPEN, "right": HandState.OPEN}

    def analyze(self, frame, landmarks, side, frame_size):
        self.analyzed_sides.append(side)
        return HandObservation(self.states[side], LEFT_HAND_BOX if side == "left" else RIGHT_HAND_BOX)


class FakeObjectDetector:
    """detect()가 미리 정한 검출 목록을 돌려준다."""

    def __init__(self):
        self.detections = []

    def detect(self, frame, timestamp_ms, frame_index):
        return self.detections


class ProcessFrameRulesTest(unittest.TestCase):
    def setUp(self):
        self.frame = make_frame(SKIN_BGR)
        self.detector, self.hand_analyzer, self.object_detector = FakeDetector(), FakeHandAnalyzer(), FakeObjectDetector()
        self.tracker, self.person_states = MultiTargetTracker(), {}

    def step(self, poses, frame_index, detections=()):
        self.detector.poses = poses
        self.object_detector.detections = list(detections)
        return process_frame(
            self.frame, self.detector, self.hand_analyzer, self.object_detector, self.tracker, self.person_states,
            timestamp_ms=frame_index, frame_index=frame_index,
        )[1]

    def test_surrender_after_debounce_then_kept_while_predicting_then_cleaned_up(self):
        pose = make_surrender_pose()
        for frame_index in range(STABLE_FRAMES - 1):
            self.assertEqual(self.step([pose], frame_index)[0].threat_state, ThreatState.UNKNOWN)
        self.assertEqual(self.step([pose], STABLE_FRAMES)[0].threat_state, ThreatState.SURRENDER)

        # 사람이 가려져 Pose가 끊김 → PREDICTING 동안 직전 상태 유지, 손 분석 없음
        for frame_index in range(MAX_PREDICT_FRAMES):
            views = self.step([], 100 + frame_index)
            self.assertEqual(views[0].track_state, TrackState.PREDICTING)
            self.assertEqual(views[0].threat_state, ThreatState.SURRENDER)
            self.assertEqual(views[0].hands, {})
        # 예측 한도를 넘기면 트랙과 사람 상태가 함께 사라진다
        self.assertEqual(self.step([], 200), [])
        self.assertEqual(self.person_states, {})

    def test_hands_are_analyzed_alternately_on_odd_frames_after_first_frame(self):
        pose = make_surrender_pose()
        self.step([pose], 0)
        self.assertEqual(sorted(self.hand_analyzer.analyzed_sides), sorted(HAND_SIDES))  # 처음 본 사람은 양손 모두
        self.hand_analyzer.analyzed_sides.clear()
        for frame_index in range(1, 5):
            views = self.step([pose], frame_index)
            self.assertEqual(set(views[0].hands), set(HAND_SIDES))  # 분석 안 한 손은 직전 결과 재사용
        self.assertEqual(self.hand_analyzer.analyzed_sides, ["left", "right"])  # 홀수 프레임(1, 3)에 한 손씩

    def test_object_overlapping_closed_left_hand_makes_danger_and_releases_after_hold(self):
        pose = make_surrender_pose()
        self.hand_analyzer.states["left"] = HandState.CLOSED  # 물건을 쥔 손은 주먹 모양
        detection = DetectedObject("book", 0.4, (110, 60, 40, 40))  # 왼손 박스 안
        for frame_index in range(STABLE_FRAMES):
            views = self.step([pose], frame_index, [detection])
        self.assertEqual(views[0].hands["left"].state, HandState.OBJECT)
        self.assertEqual(views[0].hands["left"].object_label, "book")
        self.assertEqual(views[0].hands["right"].state, HandState.OPEN)
        self.assertEqual(views[0].threat_state, ThreatState.DANGER)

        # 검출이 끊겨도 OBJECT_HOLD_FRAMES 동안 OBJECT 유지, 그 뒤 풀림
        for frame_index in range(OBJECT_HOLD_FRAMES):
            views = self.step([pose], 100 + frame_index)
            self.assertEqual(views[0].hands["left"].state, HandState.OBJECT)
        views = self.step([pose], 200)
        self.assertEqual(views[0].hands["left"].state, HandState.CLOSED)

    def test_detection_on_open_hand_is_ignored(self):
        # 활짝 편 손은 물건을 쥘 수 없다 → 오검출이 겹쳐도 OPEN 유지, SURRENDER 가능
        pose = make_surrender_pose()
        detection = DetectedObject("remote", 0.4, (110, 60, 40, 40))
        for frame_index in range(STABLE_FRAMES):
            views = self.step([pose], frame_index, [detection])
        self.assertEqual(views[0].hands["left"].state, HandState.OPEN)
        self.assertEqual(views[0].threat_state, ThreatState.SURRENDER)

    def test_detection_far_from_hands_is_ignored(self):
        pose = make_surrender_pose()
        for frame_index in range(STABLE_FRAMES):
            views = self.step([pose], frame_index, [DetectedObject("bottle", 0.9, (300, 300, 30, 30))])
        self.assertEqual({hand.state for hand in views[0].hands.values()}, {HandState.OPEN})

    def test_hand_not_found_for_long_becomes_hidden_and_danger(self):
        # 등 뒤로 숨긴 손: Pose는 손목을 "보인다"고 추정하지만 crop엔 손이 없어 NOT_FOUND가 이어진다 → 가림으로 간주
        pose = make_surrender_pose()
        self.hand_analyzer.states["left"] = HandState.NOT_FOUND
        for frame_index in range(NOT_FOUND_FRAMES_AS_HIDDEN - 1):
            views = self.step([pose], frame_index)
            self.assertEqual(views[0].hands["left"].state, HandState.NOT_FOUND)
        views = self.step([pose], NOT_FOUND_FRAMES_AS_HIDDEN - 1)
        self.assertEqual(views[0].hands["left"].state, HandState.HIDDEN)
        for frame_index in range(STABLE_FRAMES):
            views = self.step([pose], 100 + frame_index)
        self.assertEqual(views[0].threat_state, ThreatState.DANGER)

    def test_two_people_get_separate_states(self):
        poses = [make_surrender_pose(0.25), make_surrender_pose(0.75)]
        for frame_index in range(STABLE_FRAMES):
            views = self.step(poses, frame_index)
        self.assertEqual([view.person_id for view in views], [1, 2])
        self.assertEqual(set(self.person_states), {1, 2})


if __name__ == "__main__":
    unittest.main()
