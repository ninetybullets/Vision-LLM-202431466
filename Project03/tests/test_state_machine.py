"""state_machine 디바운스·Event 규칙 검증: 관측 목록 + 시각 → Event 목록 (seam (a)).

프로토타입(final_project/prototype-state-machine.html)의 따라 하기 탭 시나리오를 그대로 옮겼다.
시각은 인자로 주고 실제로 기다리지 않는다. T는 DEBOUNCE_SECONDS 이름으로만 부른다.
"""
import unittest

from state_machine import DEBOUNCE_SECONDS, create_state, read_targets, step

# 프로토타입 DEFAULT_CONFIG.frameIntervalMs = 100 → 초당 10프레임
FRAME_SECONDS = 0.1
# 부동소수 덧셈 오차를 흡수하는 여유. 시각 비교 창의 양 끝에만 쓴다.
TIME_TOLERANCE = 1e-6


def make_observation(track_id, zone="far", held_objects=()):
    """이번 프레임에 보인 사람 한 명의 관측 dict를 만든다."""
    return {"track_id": track_id, "zone": zone, "held_objects": list(held_objects)}


class FrameRunner:
    """프로토타입 페이지의 observe/wait를 흉내 내는 테스트용 시계 + 상태 보관."""

    def __init__(self, frame_seconds=FRAME_SECONDS):
        self.frame_seconds = frame_seconds
        self.state = create_state()
        self.clock = 0.0

    def observe(self, observations):
        """장면이 바뀐 순간: 현재 시각에 프레임 한 장을 보고 그 프레임의 Event를 돌려준다."""
        self.state, events = step(self.state, observations, self.clock)
        return events

    def wait(self, observations, seconds):
        """같은 장면을 seconds 동안 프레임 간격마다 보고, 그동안 생긴 Event 전부를 돌려준다."""
        frame_count = max(1, round(seconds / self.frame_seconds))
        events = []
        for _ in range(frame_count):
            self.clock += self.frame_seconds
            events.extend(self.observe(observations))
        return events

    def settle(self, observations):
        """장면을 바꾸고 T를 넘길 때까지 본다. (T 직전까지의 Event, T 직후 두 프레임의 Event)를 돌려준다."""
        events_before = self.observe(observations)
        events_before += self.wait(observations, DEBOUNCE_SECONDS - self.frame_seconds)
        events_after = self.wait(observations, 2 * self.frame_seconds)
        return events_before, events_after


def run_basic_flow():
    """기본 흐름(탭 1)을 처음부터 끝까지 돌려 Event 전부를 돌려준다."""
    runner = FrameRunner()
    events = []
    for observations in (
        [make_observation(1, "far")],
        [make_observation(1, "near")],
        [make_observation(1, "near", ["knife"])],
        [],
    ):
        events_before, events_after = runner.settle(observations)
        events += events_before + events_after
    return events


class BasicFlowTest(unittest.TestCase):
    """탭 1. 기본 흐름: far 출현 → near 접근 → 칼 꺼냄 → 이탈."""

    def test_each_change_gives_one_event_only_after_debounce(self):
        runner = FrameRunner()

        before, after = runner.settle([make_observation(1, "far")])
        self.assertEqual(before, [])
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0]["kind"], "appeared")
        self.assertEqual(after[0]["target_id"], 1)
        self.assertIsNone(after[0]["before"])
        self.assertEqual(after[0]["after"], {"zone": "far", "held_objects": []})

        before, after = runner.settle([make_observation(1, "near")])
        self.assertEqual(before, [])
        self.assertEqual([(event["kind"], event["before"], event["after"]) for event in after],
                         [("zone_changed", "far", "near")])

        before, after = runner.settle([make_observation(1, "near", ["knife"])])
        self.assertEqual(before, [])
        self.assertEqual([(event["kind"], event["before"], event["after"]) for event in after],
                         [("object_changed", [], ["knife"])])

        before, after = runner.settle([])
        self.assertEqual(before, [])
        self.assertEqual([(event["kind"], event["before"], event["after"]) for event in after],
                         [("gone", {"zone": "near", "held_objects": ["knife"]}, None)])

    def test_event_has_exactly_five_keys(self):
        for event in run_basic_flow():
            self.assertEqual(set(event), {"timestamp", "target_id", "kind", "before", "after"})


class ShortDropoutTest(unittest.TestCase):
    """탭 2. 짧은 끊김과 이탈, 같은 추적 id로 재진입."""

    def test_short_dropout_is_silent_and_long_absence_is_gone_then_reappear(self):
        runner = FrameRunner()
        near = [make_observation(1, "near")]
        runner.settle(near)

        dropout_events = runner.observe([]) + runner.wait([], DEBOUNCE_SECONDS / 2)
        dropout_events += runner.observe(near) + runner.wait(near, DEBOUNCE_SECONDS * 2)
        self.assertEqual(dropout_events, [])

        before, after = runner.settle([])
        self.assertEqual(before, [])
        self.assertEqual([event["kind"] for event in after], ["gone"])

        before, after = runner.settle(near)
        self.assertEqual(before, [])
        self.assertEqual([(event["kind"], event["target_id"]) for event in after], [("appeared", 1)])


class GlanceTest(unittest.TestCase):
    """탭 3. 스쳐 지나감: T보다 짧게 보였다 사라지면 Event 0건."""

    def test_brief_sighting_gives_no_event_at_all(self):
        runner = FrameRunner()
        seen = [make_observation(2)]
        events = runner.observe(seen) + runner.wait(seen, DEBOUNCE_SECONDS / 2)
        events += runner.observe([]) + runner.wait([], DEBOUNCE_SECONDS * 2)
        self.assertEqual(events, [])
        self.assertEqual(read_targets(runner.state), {})


class SimultaneousChangesTest(unittest.TestCase):
    """탭 4. 한순간에 여러 변화."""

    def test_appearing_with_object_puts_zone_and_object_in_one_event(self):
        runner = FrameRunner()
        _, after = runner.settle([make_observation(1, "far", ["backpack"])])
        self.assertEqual([(event["kind"], event["after"]) for event in after],
                         [("appeared", {"zone": "far", "held_objects": ["backpack"]})])

    def test_zone_and_object_change_together_give_two_events_same_timestamp(self):
        runner = FrameRunner()
        runner.settle([make_observation(1, "far", ["backpack"])])
        _, after = runner.settle([make_observation(1, "near", ["backpack", "cell phone"])])
        self.assertEqual([(event["kind"], event["before"], event["after"]) for event in after],
                         [("zone_changed", "far", "near"),
                          ("object_changed", ["backpack"], ["backpack", "cell phone"])])
        self.assertEqual(after[0]["timestamp"], after[1]["timestamp"])

    def test_dropping_two_objects_together_gives_one_event(self):
        runner = FrameRunner()
        runner.settle([make_observation(1, "near", ["cell phone", "backpack"])])
        _, after = runner.settle([make_observation(1, "near")])
        self.assertEqual([(event["kind"], event["before"], event["after"]) for event in after],
                         [("object_changed", ["backpack", "cell phone"], [])])


class ZoneFlickerTest(unittest.TestCase):
    """탭 5. near/far 경계 떨림."""

    def test_flicker_is_silent_then_steady_near_gives_one_event(self):
        runner = FrameRunner()
        far = [make_observation(1, "far")]
        near = [make_observation(1, "near")]
        runner.settle(far)

        flicker_events = []
        for _ in range(2):
            for observations in (near, far):
                flicker_events += runner.observe(observations)
                flicker_events += runner.wait(observations, DEBOUNCE_SECONDS * 0.4)
        self.assertEqual(flicker_events, [])

        before, after = runner.settle(near)
        self.assertEqual(before, [])
        self.assertEqual([(event["kind"], event["before"], event["after"]) for event in after],
                         [("zone_changed", "far", "near")])


class PerClassObjectFlickerTest(unittest.TestCase):
    """탭 6b. 물체 깜빡임(종류별): 칼 깜빡임은 칼에만 영향, 휴대폰은 T 뒤 기록."""

    def test_phone_is_recorded_while_knife_detection_flickers(self):
        runner = FrameRunner()
        runner.settle([make_observation(1, "near", ["knife"])])

        with_knife = [make_observation(1, "near", ["knife", "cell phone"])]
        without_knife = [make_observation(1, "near", ["cell phone"])]
        phone_start = runner.clock
        events = runner.observe(with_knife)
        for _ in range(4):
            for observations in (without_knife, with_knife):
                events += runner.observe(observations)
                events += runner.wait(observations, DEBOUNCE_SECONDS * 0.3)
        events += runner.wait(with_knife, DEBOUNCE_SECONDS)

        self.assertEqual([(event["kind"], event["before"], event["after"]) for event in events],
                         [("object_changed", ["knife"], ["cell phone", "knife"])])
        elapsed = events[0]["timestamp"] - phone_start
        self.assertGreaterEqual(elapsed, DEBOUNCE_SECONDS - TIME_TOLERANCE)
        self.assertLess(elapsed, DEBOUNCE_SECONDS + FRAME_SECONDS + TIME_TOLERANCE)


class LowFrameRateTest(unittest.TestCase):
    """탭 7. FPS가 떨어져도 확정까지 걸리는 시간은 T초(프레임 수 기준이 아님)."""

    def test_confirmation_time_is_seconds_not_frames(self):
        for frame_seconds in (0.1, 0.5):
            with self.subTest(frame_seconds=frame_seconds):
                runner = FrameRunner(frame_seconds)
                seen = [make_observation(1, "far")]
                events = runner.observe(seen) + runner.wait(seen, DEBOUNCE_SECONDS * 2)
                self.assertEqual([event["kind"] for event in events], ["appeared"])
                elapsed = events[0]["timestamp"]
                self.assertGreaterEqual(elapsed, DEBOUNCE_SECONDS - TIME_TOLERANCE)
                self.assertLess(elapsed, DEBOUNCE_SECONDS + frame_seconds + TIME_TOLERANCE)


class AppearanceFrameObjectsTest(unittest.TestCase):
    """출현 확정 프레임에 보인 물체만 appeared에 담고, 나중에 보인 물체는 T 뒤 object_changed."""

    def test_object_missing_on_confirm_frame_comes_later_as_object_changed(self):
        with_knife = [make_observation(1, "far", ["knife"])]
        without_knife = [make_observation(1, "far")]
        state = create_state()
        state, _ = step(state, with_knife, 0.0)
        state, _ = step(state, with_knife, DEBOUNCE_SECONDS / 2)
        state, events = step(state, without_knife, DEBOUNCE_SECONDS)
        self.assertEqual([(event["kind"], event["after"]) for event in events],
                         [("appeared", {"zone": "far", "held_objects": []})])

        knife_start = DEBOUNCE_SECONDS * 2
        state, events = step(state, with_knife, knife_start)
        self.assertEqual(events, [])
        state, events = step(state, with_knife, knife_start + DEBOUNCE_SECONDS - 0.01)
        self.assertEqual(events, [])
        state, events = step(state, with_knife, knife_start + DEBOUNCE_SECONDS + 0.01)
        self.assertEqual([(event["kind"], event["before"], event["after"]) for event in events],
                         [("object_changed", [], ["knife"])])


class BoundaryTest(unittest.TestCase):
    """T 바로 아래에서는 안 바뀌고 T 이상에서 바뀐다."""

    def test_appearance_confirms_exactly_at_debounce_seconds(self):
        seen = [make_observation(1)]
        state, _ = step(create_state(), seen, 0.0)
        state, events = step(state, seen, DEBOUNCE_SECONDS - 0.01)
        self.assertEqual(events, [])
        state, events = step(state, seen, DEBOUNCE_SECONDS)
        self.assertEqual([event["kind"] for event in events], ["appeared"])

    def test_gone_confirms_only_after_debounce_seconds(self):
        seen = [make_observation(1)]
        state, _ = step(create_state(), seen, 0.0)
        state, _ = step(state, seen, DEBOUNCE_SECONDS)
        hide_start = DEBOUNCE_SECONDS * 2
        state, events = step(state, [], hide_start)
        self.assertEqual(events, [])
        state, events = step(state, [], hide_start + DEBOUNCE_SECONDS - 0.01)
        self.assertEqual(events, [])
        state, events = step(state, [], hide_start + DEBOUNCE_SECONDS + 0.01)
        self.assertEqual([event["kind"] for event in events], ["gone"])


class DeterminismTest(unittest.TestCase):
    def test_same_input_sequence_gives_same_events(self):
        first_run = run_basic_flow()
        self.assertEqual(len(first_run), 4)
        self.assertEqual(first_run, run_basic_flow())


class MultipleTargetsTest(unittest.TestCase):
    def test_events_in_one_frame_are_sorted_by_target_id(self):
        runner = FrameRunner()
        _, after = runner.settle([make_observation(3, "near"), make_observation(1, "far")])
        self.assertEqual([(event["kind"], event["target_id"]) for event in after],
                         [("appeared", 1), ("appeared", 3)])


class ReadTargetsTest(unittest.TestCase):
    """read_targets: 확정된 Target만 {target_id: Target State}로. Snapshot·HUD 재료."""

    def test_reads_only_confirmed_targets(self):
        runner = FrameRunner()
        self.assertEqual(read_targets(runner.state), {})
        runner.settle([make_observation(1, "near", ["knife", "bottle"])])
        runner.observe([make_observation(1, "near", ["knife", "bottle"]), make_observation(2, "far")])
        self.assertEqual(read_targets(runner.state),
                         {1: {"zone": "near", "held_objects": ["bottle", "knife"]}})

    def test_gone_target_is_removed(self):
        runner = FrameRunner()
        runner.settle([make_observation(1)])
        runner.settle([])
        self.assertEqual(read_targets(runner.state), {})


if __name__ == "__main__":
    unittest.main()
