"""announcer 안내방송 테스트. 스레드·Piper·스피커 없이 방송 선택·대기열·문구표만 검사한다.

실행: 코드 폴더에서 python -m unittest tests.test_announcer
"""

import subprocess
import sys
import unittest
from pathlib import Path

from announcer import (
    ANNOUNCEMENT_TEXTS,
    MANUAL_ANNOUNCEMENT_KEYS,
    Announcer,
    build_wav_path,
    select_auto_announcement,
)

CODE_DIRECTORY = Path(__file__).resolve().parent.parent


def make_event(kind, before, after, target_id=1, timestamp=1.0):
    """테스트용 Event dict 하나를 만든다."""
    return {"timestamp": timestamp, "target_id": target_id, "kind": kind, "before": before, "after": after}


HOLD_KNIFE_EVENT = make_event("object_changed", [], ["knife"])


class SelectAutoAnnouncementTest(unittest.TestCase):
    def test_object_changed_to_held_object(self):
        self.assertEqual(select_auto_announcement([HOLD_KNIFE_EVENT]), "put_down")

    def test_object_changed_to_empty(self):
        self.assertIsNone(select_auto_announcement([make_event("object_changed", ["knife"], [])]))

    def test_appeared_with_held_object(self):
        event = make_event("appeared", None, {"zone": "far", "held_objects": ["cell phone"]})
        self.assertEqual(select_auto_announcement([event]), "put_down")

    def test_appeared_without_held_object(self):
        event = make_event("appeared", None, {"zone": "far", "held_objects": []})
        self.assertIsNone(select_auto_announcement([event]))

    def test_zone_changed_and_gone(self):
        events = [
            make_event("zone_changed", "far", "near"),
            make_event("gone", {"zone": "near", "held_objects": ["knife"]}, None),
        ]
        self.assertIsNone(select_auto_announcement(events))

    def test_no_events(self):
        self.assertIsNone(select_auto_announcement([]))


# start()를 부르지 않으므로 작업 스레드가 없다. 대기열이 그대로 남아 있어 내부 목록 _pending을 직접 읽어 확인한다
class HandleEventsTest(unittest.TestCase):
    def setUp(self):
        self.announcer = Announcer(is_voice_enabled=False)

    def test_first_event_is_queued(self):
        self.assertTrue(self.announcer.handle_events([HOLD_KNIFE_EVENT], now=5.0))
        self.assertEqual(self.announcer._pending, ["put_down"])

    def test_rebroadcast_waits_ten_seconds(self):
        self.announcer.handle_events([HOLD_KNIFE_EVENT], now=5.0)
        self.assertFalse(self.announcer.handle_events([HOLD_KNIFE_EVENT], now=12.0))
        self.assertTrue(self.announcer.handle_events([HOLD_KNIFE_EVENT], now=15.0))

    def test_other_target_is_gated_by_same_announcement(self):
        self.announcer.handle_events([HOLD_KNIFE_EVENT], now=5.0)
        other_target_event = make_event("object_changed", [], ["bottle"], target_id=2)
        self.assertFalse(self.announcer.handle_events([other_target_event], now=8.0))

    def test_no_announcement_returns_false(self):
        self.assertFalse(self.announcer.handle_events([make_event("zone_changed", "far", "near")], now=5.0))
        self.assertEqual(self.announcer._pending, [])


class AnnounceManualTest(unittest.TestCase):
    def test_manual_goes_first_without_duplicate(self):
        announcer = Announcer(is_voice_enabled=False)
        announcer.handle_events([HOLD_KNIFE_EVENT], now=5.0)
        announcer.announce_manual("hands_up")
        self.assertEqual(announcer._pending, ["hands_up", "put_down"])
        announcer.announce_manual("hands_up")
        self.assertEqual(announcer._pending, ["hands_up", "put_down"])


class AnnouncementTableTest(unittest.TestCase):
    def test_texts_are_three_fixed_phrases(self):
        self.assertEqual(ANNOUNCEMENT_TEXTS, {
            "put_down": "들고 있는 물건을 바닥에 내려놓으세요.",
            "hands_up": "양손을 펼쳐 어깨 위로 드세요.",
            "hold_still": "그 자세를 유지하세요.",
        })

    def test_manual_keys_point_to_existing_texts(self):
        self.assertEqual(sorted(MANUAL_ANNOUNCEMENT_KEYS), ["1", "2", "3"])
        for announcement_id in MANUAL_ANNOUNCEMENT_KEYS.values():
            self.assertIn(announcement_id, ANNOUNCEMENT_TEXTS)

    def test_wav_path_uses_announcement_id(self):
        self.assertTrue(str(build_wav_path("put_down", "/tmp/x")).endswith("put_down.wav"))


class ImportTest(unittest.TestCase):
    def test_import_announcer_does_not_load_models(self):
        # 새 프로세스에서 import해 이 테스트 프로세스의 sys.modules 상태와 무관하게 확인한다.
        check_code = "import sys, announcer; print('llama_cpp' in sys.modules or 'ultralytics' in sys.modules)"
        completed = subprocess.run(
            [sys.executable, "-c", check_code], cwd=CODE_DIRECTORY, capture_output=True, text=True, check=True
        )
        self.assertEqual(completed.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
