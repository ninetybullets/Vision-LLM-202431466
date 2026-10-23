"""threat_classifier 규칙 검증: 손 소분류 → 사람 상태, 손 들기 판정, 디바운스."""
import unittest
from collections import namedtuple

from hand_analyzer import HandState
from threat_classifier import STABLE_FRAMES, ThreatDebouncer, ThreatState, are_hands_raised, classify_person

FakeLandmark = namedtuple("FakeLandmark", "x y visibility")


def make_landmarks(wrist_y=0.3, shoulder_y=0.5, shoulder_visibility=0.9):
    """어깨(11·12)와 손목(15·16)만 의미 있는 33개 랜드마크. y가 작을수록 화면 위."""
    landmarks = [FakeLandmark(0.5, 0.5, 0.1) for _ in range(33)]
    landmarks[11] = FakeLandmark(0.4, shoulder_y, shoulder_visibility)
    landmarks[12] = FakeLandmark(0.6, shoulder_y, shoulder_visibility)
    landmarks[15] = FakeLandmark(0.3, wrist_y, 0.9)
    landmarks[16] = FakeLandmark(0.7, wrist_y, 0.9)
    return landmarks


class ClassifyPersonTest(unittest.TestCase):
    def test_any_hidden_or_object_hand_is_danger(self):
        for dangerous in (HandState.HIDDEN, HandState.OBJECT):
            for other in HandState:
                self.assertEqual(classify_person(dangerous, other, True), ThreatState.DANGER)
                self.assertEqual(classify_person(other, dangerous, True), ThreatState.DANGER)

    def test_both_open_and_raised_is_surrender(self):
        self.assertEqual(classify_person(HandState.OPEN, HandState.OPEN, True), ThreatState.SURRENDER)

    def test_both_open_but_not_raised_is_unknown(self):
        self.assertEqual(classify_person(HandState.OPEN, HandState.OPEN, False), ThreatState.UNKNOWN)

    def test_one_open_one_closed_is_unknown(self):
        self.assertEqual(classify_person(HandState.OPEN, HandState.CLOSED, True), ThreatState.UNKNOWN)

    def test_out_or_not_found_is_unknown(self):
        self.assertEqual(classify_person(HandState.OUT, HandState.OPEN, True), ThreatState.UNKNOWN)
        self.assertEqual(classify_person(HandState.NOT_FOUND, HandState.NOT_FOUND, True), ThreatState.UNKNOWN)


class AreHandsRaisedTest(unittest.TestCase):
    def test_both_wrists_above_shoulders(self):
        self.assertTrue(are_hands_raised(make_landmarks(wrist_y=0.3, shoulder_y=0.5)))

    def test_wrists_below_shoulders(self):
        self.assertFalse(are_hands_raised(make_landmarks(wrist_y=0.7, shoulder_y=0.5)))

    def test_one_wrist_below_is_not_raised(self):
        landmarks = make_landmarks(wrist_y=0.3)
        landmarks[16] = FakeLandmark(0.7, 0.8, 0.9)
        self.assertFalse(are_hands_raised(landmarks))

    def test_hidden_shoulders_is_not_raised(self):
        self.assertFalse(are_hands_raised(make_landmarks(shoulder_visibility=0.1)))


class ThreatDebouncerTest(unittest.TestCase):
    def test_starts_unknown(self):
        self.assertEqual(ThreatDebouncer().state, ThreatState.UNKNOWN)

    def test_switches_only_after_stable_frames(self):
        debouncer = ThreatDebouncer()
        for _ in range(STABLE_FRAMES - 1):
            self.assertEqual(debouncer.update(ThreatState.DANGER), ThreatState.UNKNOWN)
        self.assertEqual(debouncer.update(ThreatState.DANGER), ThreatState.DANGER)

    def test_flicker_does_not_switch(self):
        debouncer = ThreatDebouncer()
        for _ in range(STABLE_FRAMES * 3):
            debouncer.update(ThreatState.DANGER)
            debouncer.update(ThreatState.SURRENDER)
        self.assertEqual(debouncer.state, ThreatState.UNKNOWN)

    def test_one_frame_of_current_state_resets_the_count(self):
        debouncer = ThreatDebouncer()
        for _ in range(STABLE_FRAMES - 1):
            debouncer.update(ThreatState.DANGER)
        debouncer.update(ThreatState.UNKNOWN)  # 현재 상태가 한 번 끼어들면 카운트 초기화
        for _ in range(STABLE_FRAMES - 1):
            self.assertEqual(debouncer.update(ThreatState.DANGER), ThreatState.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
