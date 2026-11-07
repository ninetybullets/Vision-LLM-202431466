"""Event Log: Event를 시간순으로 메모리 목록과 실행별 JSONL 파일에 쌓는다.

실행 중에는 메모리 목록을 읽고, 실행 뒤에는 logs/events-YYYYMMDD-HHMMSS.jsonl 파일로 다시 본다.
수동 확인: .venv/bin/python event_log.py (가짜 Event 3건을 기록하고 결과를 출력)
"""

import json
import time
from pathlib import Path

# 로그 폴더는 코드 폴더 기준 logs/ (스펙 「임시 결정」, .gitignore에 등록됨)
LOG_DIRECTORY = Path(__file__).resolve().parent / "logs"
# 파일 이름의 벽시계 시각 형식 → events-20261106-143015.jsonl
LOG_FILE_TIME_FORMAT = "%Y%m%d-%H%M%S"


def format_event(event):
    """Event 한 건을 한 줄 글로 바꾼다(Evidence·터미널 출력·JSONL 줄이 같은 모양).

    입력: Event dict {"timestamp", "target_id", "kind", "before", "after"}
    출력: 영어 원문 그대로의 JSON 한 줄 문자열(한글이 있으면 그대로 보이게 ensure_ascii=False)
    """
    return json.dumps(event, ensure_ascii=False)


class EventLog:
    """실행 하나의 Event Log. 메모리 목록과 열린 JSONL 파일을 상태로 들고 있다."""

    def __init__(self, log_directory=LOG_DIRECTORY):
        """로그 폴더를 만들고 이번 실행의 JSONL 파일을 연다.

        입력: log_directory — JSONL을 둘 폴더(기본: 코드 폴더의 logs/)
        출력: 없음. self.path에 만든 파일 경로가 남는다.
        """
        log_directory = Path(log_directory)
        log_directory.mkdir(parents=True, exist_ok=True)
        file_name = f"events-{time.strftime(LOG_FILE_TIME_FORMAT)}.jsonl"
        self.path = log_directory / file_name
        self._events = []
        # 같은 초에 두 번 실행해도 앞 기록을 지우지 않도록 이어 쓰기("a")로 연다
        self._file = open(self.path, "a", encoding="utf-8")

    def append(self, event):
        """Event 한 건을 메모리 목록 끝과 파일 한 줄에 추가한다.

        입력: Event dict
        출력: 없음
        """
        self._events.append(event)
        self._file.write(format_event(event) + "\n")
        # 프로그램이 중간에 죽어도 이미 쓴 줄은 디스크에 남게 매번 비운다
        self._file.flush()

    def read_events(self):
        """지금까지 쌓인 Event 목록의 복사본을 준다.

        입력: 없음
        출력: Event dict 목록(새 list). reporter 작업 스레드가 읽는 동안
              메인 스레드가 append해도 받은 목록은 바뀌지 않는다.
        """
        return list(self._events)

    def close(self):
        """JSONL 파일을 닫는다. 실행 끝에 한 번 부른다."""
        self._file.close()


def make_sample_events():
    """수동 확인용 가짜 Event 3건(appeared → zone_changed → gone)을 만든다."""
    return [
        {"timestamp": 0.5, "target_id": 1, "kind": "appeared",
         "before": None, "after": {"zone": "far", "held_objects": []}},
        {"timestamp": 3.2, "target_id": 1, "kind": "zone_changed",
         "before": "far", "after": "near"},
        {"timestamp": 7.8, "target_id": 1, "kind": "gone",
         "before": {"zone": "near", "held_objects": []}, "after": None},
    ]


def main():
    """가짜 Event 3건을 기록하고 목록 길이·한 줄 글·파일 경로를 출력한다."""
    event_log = EventLog()
    for event in make_sample_events():
        event_log.append(event)
    event_log.close()

    events = event_log.read_events()
    print(f"메모리 목록 길이: {len(events)}")
    for event in events:
        print(format_event(event))
    print(f"JSONL 파일: {event_log.path}")


if __name__ == "__main__":
    main()
