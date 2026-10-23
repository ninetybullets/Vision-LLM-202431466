"""MultiTargetTracker 검증: 여러 몸 중심을 ID로 이어 붙이고(최근접 매칭), 사라지면 예측 후 소멸, 새 사람은 새 ID."""
import unittest

from target_tracker import MATCH_DISTANCE, MAX_PREDICT_FRAMES, MultiTargetTracker, TrackState


class MultiTargetTrackerTest(unittest.TestCase):
    def test_two_centers_get_ids_in_order_with_measurement_index(self):
        tracker = MultiTargetTracker()
        tracked = tracker.update([(100, 100), (500, 100)])
        self.assertEqual([person.person_id for person in tracked], [1, 2])
        self.assertEqual([person.measurement_index for person in tracked], [0, 1])
        self.assertTrue(all(person.state == TrackState.LOCKED for person in tracked))

    def test_ids_follow_position_when_input_order_swaps(self):
        tracker = MultiTargetTracker()
        tracker.update([(100, 100), (500, 100)])
        tracked = tracker.update([(505, 102), (103, 99)])
        by_id = {person.person_id: person for person in tracked}
        self.assertEqual(by_id[1].measurement_index, 1)
        self.assertEqual(by_id[2].measurement_index, 0)

    def test_missing_person_predicts_then_disappears(self):
        tracker = MultiTargetTracker()
        tracker.update([(100, 100), (500, 100)])
        for _ in range(MAX_PREDICT_FRAMES):
            tracked = tracker.update([(100, 100)])
            by_id = {person.person_id: person for person in tracked}
            self.assertEqual(by_id[2].state, TrackState.PREDICTING)
            self.assertIsNone(by_id[2].measurement_index)
        tracked = tracker.update([(100, 100)])
        self.assertEqual([person.person_id for person in tracked], [1])

    def test_far_center_becomes_new_person_not_reassigned(self):
        tracker = MultiTargetTracker()
        tracker.update([(100, 100)])
        tracked = tracker.update([(100 + MATCH_DISTANCE + 50, 100)])
        ids = sorted(person.person_id for person in tracked)
        self.assertEqual(ids, [1, 2])  # 1은 PREDICTING으로 남고 2가 새로 생김

    def test_match_distance_scales_with_shoulder_width(self):
        # 가까운 사람(어깨 300px)은 한 프레임에 100px 넘게 움직여도 같은 사람. 허용 거리 = max(100, 0.8×300) = 240
        tracker = MultiTargetTracker()
        tracker.update([(300, 200)], shoulder_widths=[300])
        tracked = tracker.update([(300 + 200, 200)], shoulder_widths=[300])
        self.assertEqual([person.person_id for person in tracked], [1])

    def test_default_match_distance_without_shoulder_width(self):
        tracker = MultiTargetTracker()
        tracker.update([(300, 200)])
        tracked = tracker.update([(300 + MATCH_DISTANCE + 10, 200)])
        self.assertEqual(sorted(person.person_id for person in tracked), [1, 2])

    def test_ids_are_never_reused(self):
        tracker = MultiTargetTracker()
        tracker.update([(100, 100)])
        for _ in range(MAX_PREDICT_FRAMES + 1):
            tracker.update([])
        tracked = tracker.update([(100, 100)])
        self.assertEqual(tracked[0].person_id, 2)


if __name__ == "__main__":
    unittest.main()
