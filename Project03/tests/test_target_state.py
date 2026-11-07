"""state_machine Target State 판정 검증: 사람·물체 박스 + 프레임 높이 → 관측 목록 (seam (a)).

YOLO 모델 없이 박스 숫자만 넣는다. near/far 경계는 NEAR_HEIGHT_RATIO 이름으로만 부른다.
"""
import unittest

from state_machine import (
    DEBOUNCE_SECONDS,
    NEAR_HEIGHT_RATIO,
    build_observations,
    create_state,
    judge_zone,
    step,
)

# 합성 프레임 높이(px). 1000이면 박스 높이 1px = 비율 0.001이라 경계 계산이 읽기 쉽다.
FRAME_HEIGHT = 1000


def make_person_box(height, x1=100, y1=0, width=200):
    """왼쪽 위 (x1, y1)에서 시작하는 사람 박스 (x1, y1, x2, y2)를 만든다."""
    return (x1, y1, x1 + width, y1 + height)


def make_object_box(center_x, center_y, half_size=10):
    """중심이 (center_x, center_y)인 정사각형 물체 박스를 만든다."""
    return (center_x - half_size, center_y - half_size, center_x + half_size, center_y + half_size)


class JudgeZoneTest(unittest.TestCase):
    def test_height_ratio_just_above_threshold_is_near(self):
        height = NEAR_HEIGHT_RATIO * FRAME_HEIGHT + 1
        self.assertEqual(judge_zone(make_person_box(height), FRAME_HEIGHT), "near")

    def test_height_ratio_just_below_threshold_is_far(self):
        height = NEAR_HEIGHT_RATIO * FRAME_HEIGHT - 1
        self.assertEqual(judge_zone(make_person_box(height), FRAME_HEIGHT), "far")

    def test_height_ratio_equal_to_threshold_is_near(self):
        # 스펙 「이상이면 near」
        height = NEAR_HEIGHT_RATIO * FRAME_HEIGHT
        self.assertEqual(judge_zone(make_person_box(height), FRAME_HEIGHT), "near")


class BuildObservationsTest(unittest.TestCase):
    def test_no_person_gives_empty_list(self):
        objects = [("knife", make_object_box(150, 300))]
        self.assertEqual(build_observations([], objects, FRAME_HEIGHT), [])

    def test_no_object_gives_empty_held_objects(self):
        persons = [(1, make_person_box(800)), (2, make_person_box(200, x1=600))]
        self.assertEqual(
            build_observations(persons, [], FRAME_HEIGHT),
            [
                {"track_id": 1, "zone": "near", "held_objects": []},
                {"track_id": 2, "zone": "far", "held_objects": []},
            ],
        )

    def test_object_center_inside_person_box_is_held(self):
        persons = [(1, make_person_box(800))]  # x 100~300, y 0~800
        objects = [("knife", make_object_box(150, 400))]
        observations = build_observations(persons, objects, FRAME_HEIGHT)
        self.assertEqual(observations[0]["held_objects"], ["knife"])

    def test_object_center_outside_person_box_is_not_held(self):
        persons = [(1, make_person_box(800))]  # x 100~300
        # 박스 가장자리는 사람 박스에 걸치지만 중심(x=320)은 밖
        objects = [("knife", make_object_box(320, 400, half_size=30))]
        observations = build_observations(persons, objects, FRAME_HEIGHT)
        self.assertEqual(observations[0]["held_objects"], [])

    def test_object_inside_two_person_boxes_goes_to_closer_center_only(self):
        # 사람 1 중심 x=200, 사람 2 중심 x=300. 겹치는 구간 x 200~300. 물체 중심 x=280은 사람 2에 더 가깝다.
        persons = [(1, make_person_box(800, x1=100)), (2, make_person_box(800, x1=200))]
        objects = [("knife", make_object_box(280, 400))]
        observations = build_observations(persons, objects, FRAME_HEIGHT)
        held_by_id = {observation["track_id"]: observation["held_objects"] for observation in observations}
        self.assertEqual(held_by_id, {1: [], 2: ["knife"]})

    def test_object_inside_two_person_boxes_goes_to_closer_center_listed_first(self):
        # 위와 반대: 물체 중심 x=220은 목록 앞의 사람 1(중심 x=200)에 더 가깝다. "마지막에 맞은 사람"을 고르는 구현을 걸러낸다.
        persons = [(1, make_person_box(800, x1=100)), (2, make_person_box(800, x1=200))]
        objects = [("knife", make_object_box(220, 400))]
        observations = build_observations(persons, objects, FRAME_HEIGHT)
        held_by_id = {observation["track_id"]: observation["held_objects"] for observation in observations}
        self.assertEqual(held_by_id, {1: ["knife"], 2: []})

    def test_same_class_twice_listed_once_and_classes_sorted(self):
        persons = [(1, make_person_box(800))]
        objects = [
            ("knife", make_object_box(150, 200)),
            ("cell phone", make_object_box(200, 300)),
            ("knife", make_object_box(250, 400)),
            ("bottle", make_object_box(180, 500)),
        ]
        observations = build_observations(persons, objects, FRAME_HEIGHT)
        self.assertEqual(observations[0]["held_objects"], ["bottle", "cell phone", "knife"])


class BuildObservationsIntoStepTest(unittest.TestCase):
    def test_person_with_knife_appears_after_debounce(self):
        persons = [(7, make_person_box(800))]
        objects = [("knife", make_object_box(150, 400))]
        observations = build_observations(persons, objects, FRAME_HEIGHT)

        state, first_events = step(create_state(), observations, 0.0)
        state, confirmed_events = step(state, observations, DEBOUNCE_SECONDS)

        self.assertEqual(first_events, [])
        self.assertEqual(
            confirmed_events,
            [
                {
                    "timestamp": DEBOUNCE_SECONDS,
                    "target_id": 7,
                    "kind": "appeared",
                    "before": None,
                    "after": {"zone": "near", "held_objects": ["knife"]},
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
