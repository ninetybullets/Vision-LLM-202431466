"""target_tracker 상태머신 검증: SEARCHING → LOCKED → PREDICTING → SEARCHING 전이와 재획득, 예측 위치."""
import unittest

from target_tracker import MAX_PREDICT_FRAMES, TargetTracker, TrackState


class TargetTrackerTest(unittest.TestCase):
    def test_starts_searching_without_measurement(self):
        tracker = TargetTracker()
        self.assertEqual(tracker.update(None).state, TrackState.SEARCHING)

    def test_measurement_locks_target_at_measured_position(self):
        tracker = TargetTracker()
        result = tracker.update((100, 200))
        self.assertEqual(result.state, TrackState.LOCKED)
        self.assertEqual(result.center, (100, 200))

    def test_lost_measurement_predicts_then_gives_up_after_max_frames(self):
        tracker = TargetTracker()
        tracker.update((100, 200))
        for _ in range(MAX_PREDICT_FRAMES):
            self.assertEqual(tracker.update(None).state, TrackState.PREDICTING)
        self.assertEqual(tracker.update(None).state, TrackState.SEARCHING)

    def test_reacquire_during_prediction_returns_to_locked_without_searching(self):
        tracker = TargetTracker()
        tracker.update((100, 200))
        states = [tracker.update(None).state for _ in range(10)]
        states.append(tracker.update((110, 200)).state)
        self.assertNotIn(TrackState.SEARCHING, states)
        self.assertEqual(states[-1], TrackState.LOCKED)

    def test_prediction_continues_constant_velocity_motion(self):
        tracker = TargetTracker()
        for x in range(100, 200, 10):  # 매 프레임 +10px로 오른쪽 이동
            tracker.update((x, 150))
        predicted = tracker.update(None).center
        self.assertGreater(predicted[0], 190)
        self.assertAlmostEqual(predicted[1], 150, delta=5)


if __name__ == "__main__":
    unittest.main()
