"""reporter 프롬프트 생성 테스트(seam b). 모델 없이 프롬프트 문자열만 검사한다.

실행: cd final_project/code && .venv/bin/python -m unittest tests.test_prompt
"""

import subprocess
import sys
import unittest
from pathlib import Path

from reporter import (
    MAX_BATCH_EVENTS,
    QUESTION_WINDOW_SECONDS,
    build_question_prompt,
    build_sitrep_prompt,
    describe_snapshot,
)

CODE_DIRECTORY = Path(__file__).resolve().parent.parent


def make_event(timestamp, target_id=1, kind="zone_changed", before="far", after="near"):
    """테스트용 Event dict 하나를 만든다."""
    return {"timestamp": timestamp, "target_id": target_id, "kind": kind, "before": before, "after": after}


TWO_TARGETS = {
    1: {"zone": "near", "held_objects": ["knife"]},
    2: {"zone": "far", "held_objects": []},
}


class SystemRuleTest(unittest.TestCase):
    def test_system_text_has_fixed_rules(self):
        sitrep_system, _ = build_sitrep_prompt([make_event(1.0)], TWO_TARGETS, now=5.0)
        question_system, _ = build_question_prompt("지금 상황은?", [], TWO_TARGETS, now=5.0)
        for system_text in (sitrep_system, question_system):
            self.assertIn("로그에 있는 사실만", system_text)
            self.assertIn("관측되지 않음", system_text)
            self.assertIn("한국어 낱말", system_text)


class KoreanLabelTest(unittest.TestCase):
    def test_english_values_become_korean_labels(self):
        events = [
            make_event(1.0, target_id=1, kind="zone_changed", before="far", after="near"),
            make_event(2.0, target_id=1, kind="object_changed", before=[], after=["knife"]),
        ]
        _, user_text = build_sitrep_prompt(events, {}, now=5.0)
        self.assertIn("가까운 구역", user_text)
        self.assertIn("칼", user_text)
        self.assertIn("표적 1번", user_text)
        self.assertNotIn("near", user_text)
        self.assertNotIn("knife", user_text)

    def test_appeared_and_gone_use_target_state_labels(self):
        events = [
            make_event(1.0, kind="appeared", before=None, after={"zone": "far", "held_objects": ["bottle"]}),
            make_event(2.0, kind="gone", before={"zone": "near", "held_objects": []}, after=None),
        ]
        _, user_text = build_sitrep_prompt(events, {}, now=5.0)
        self.assertIn("먼 구역", user_text)
        self.assertIn("병", user_text)
        self.assertIn("물체 없음", user_text)
        self.assertNotIn("bottle", user_text)
        self.assertNotIn("far", user_text)

    def test_unknown_class_stays_english(self):
        events = [make_event(1.0, kind="object_changed", before=[], after=["umbrella"])]
        _, user_text = build_sitrep_prompt(events, {}, now=5.0)
        self.assertIn("umbrella", user_text)


class BatchLimitTest(unittest.TestCase):
    def test_over_limit_keeps_latest_and_marks_omitted(self):
        total_count = MAX_BATCH_EVENTS + 3
        events = [make_event(float(second), target_id=second + 1) for second in range(total_count)]
        _, user_text = build_sitrep_prompt(events, {}, now=100.0)
        self.assertIn("이전 3건 생략", user_text)
        # 가장 오래된 3건(표적 1~3번)은 빠지고 최신 건(표적 4번~)은 들어간다.
        for omitted_id in (1, 2, 3):
            self.assertNotIn(f"표적 {omitted_id}번", user_text)
        for kept_id in range(4, total_count + 1):
            self.assertIn(f"표적 {kept_id}번", user_text)

    def test_at_limit_has_no_omitted_mark(self):
        events = [make_event(float(second), target_id=second + 1) for second in range(MAX_BATCH_EVENTS)]
        _, user_text = build_sitrep_prompt(events, {}, now=100.0)
        self.assertNotIn("생략", user_text)


class RelativeTimeTest(unittest.TestCase):
    def test_relative_seconds_are_truncated(self):
        _, user_text = build_sitrep_prompt([make_event(10.0)], {}, now=22.7)
        self.assertIn("12초 전", user_text)
        self.assertNotIn("13초 전", user_text)


class QuestionPromptTest(unittest.TestCase):
    def test_window_boundary_and_question_text(self):
        now = 200.0
        outside_event = make_event(now - QUESTION_WINDOW_SECONDS - 0.1, target_id=7)
        inside_event = make_event(now - QUESTION_WINDOW_SECONDS + 0.1, target_id=8)
        _, user_text = build_question_prompt("칼 든 사람 있어?", [outside_event, inside_event], {}, now=now)
        self.assertNotIn("표적 7번", user_text)
        self.assertIn("표적 8번", user_text)
        self.assertIn("칼 든 사람 있어?", user_text)

    def test_zero_events_still_builds_prompt_with_snapshot(self):
        system_text, user_text = build_question_prompt("지난 1분 동안 무엇이 바뀌었나?", [], TWO_TARGETS, now=30.0)
        self.assertTrue(system_text)
        self.assertIn("기록 없음", user_text)
        self.assertIn("현재: 표적 2명.", user_text)

    def test_events_after_question_time_are_excluded(self):
        # now = 질문 시각(키를 누른 순간). 그 뒤에 생긴 Event는 질문 시점의 상황이 아니다
        question_time = 100.0
        later_event = make_event(question_time + 0.5, target_id=9)
        _, user_text = build_question_prompt("지금 상황은?", [later_event], {}, now=question_time)
        self.assertNotIn("표적 9번", user_text)

    def test_event_at_question_time_is_included(self):
        question_time = 100.0
        same_time_event = make_event(question_time, target_id=4)
        _, user_text = build_question_prompt("지금 상황은?", [same_time_event], {}, now=question_time)
        self.assertIn("표적 4번", user_text)


class SnapshotTest(unittest.TestCase):
    def test_two_targets_match_spec_shape(self):
        self.assertEqual(
            describe_snapshot(TWO_TARGETS),
            "현재: 표적 2명. 표적 1번 가까운 구역, 칼 보유. 표적 2번 먼 구역, 물체 없음.",
        )

    def test_no_targets(self):
        self.assertEqual(describe_snapshot({}), "현재: 표적 없음.")

    def test_both_prompts_include_snapshot_line(self):
        snapshot_line = describe_snapshot(TWO_TARGETS)
        _, sitrep_user = build_sitrep_prompt([make_event(1.0)], TWO_TARGETS, now=5.0)
        _, question_user = build_question_prompt("지금 상황은?", [make_event(1.0)], TWO_TARGETS, now=5.0)
        self.assertIn(snapshot_line, sitrep_user)
        self.assertIn(snapshot_line, question_user)


class ImportTest(unittest.TestCase):
    def test_import_reporter_does_not_load_llama_cpp(self):
        # 새 프로세스에서 import해 이 테스트 프로세스의 sys.modules 상태와 무관하게 확인한다.
        check_code = "import sys, reporter; print('llama_cpp' in sys.modules)"
        completed = subprocess.run(
            [sys.executable, "-c", check_code], cwd=CODE_DIRECTORY, capture_output=True, text=True, check=True
        )
        self.assertEqual(completed.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
