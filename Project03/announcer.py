"""announcer: 안내방송 — 화면 속 사람에게 스피커로 고정 문구를 내보낸다.

자동 방송(물체를 들었다고 인정된 Event) + 수동 방송(키 1·2·3). 작업 스레드 하나가 방송을 차례로 낸다.
문구 wav는 시작 때 Piper로 한 번 합성해 캐시한다(Jetson Piper 한 번 약 3.3초 → 방송 때는 재생만 해서 지연이 거의 없다).
테스트: 코드 폴더에서 python -m unittest tests.test_announcer
수동 확인(세 문구를 합성해 차례로 재생): .venv/bin/python announcer.py
"""

import threading
import time
from pathlib import Path

import voice
from platform_config import ANNOUNCEMENT_WAV_DIRECTORY

# 안내방송 문구표(2026-11-08 사용자 결정). 키 1·2·3과 자동 방송이 같은 표를 쓴다. 문구를 바꾸면 시작 때 다시 합성된다
ANNOUNCEMENT_TEXTS = {
    "put_down": "들고 있는 물건을 바닥에 내려놓으세요.",
    "hands_up": "양손을 펼쳐 어깨 위로 드세요.",
    "hold_still": "그 자세를 유지하세요.",
}
# 수동 방송 키(Operator가 화면을 보고 고른다). 옛 Preset Question 키 1·2를 대체한다
MANUAL_ANNOUNCEMENT_KEYS = {"1": "put_down", "2": "hands_up", "3": "hold_still"}
# 같은 자동 방송을 다시 내보내기까지 최소 간격(초). 임시값(근거 없음), 실측 후 교체
REBROADCAST_SECONDS = 10.0
# 자동 방송 근거: 물체를 들었다고 인정된 Event → put_down. 손 숨김·항복 판정은 이번 범위 밖(README 「확장성」)
AUTO_ANNOUNCEMENT_ID = "put_down"


def select_auto_announcement(events):
    """이번 프레임 Event 중 Target이 물체를 들게 된 것이 있으면 자동 방송 id를 고른다.

    입력: events — 상태기계 step()이 낸 Event 목록. 물체 보유를 뜻하는 모양은 둘이다.
        object_changed: after = 새 Held Object 목록(예: ["knife"]). 빈 목록이면 물체를 내려놓은 것이다.
        appeared: after = Target State dict(예: {"zone": "far", "held_objects": ["cell phone"]}).
            물체를 든 채 처음 나타나면 object_changed 없이 appeared만 생긴다(event_log.make_sample_events 모양).
    출력: AUTO_ANNOUNCEMENT_ID("put_down") 또는 None(해당 Event 없음)
    """
    for event in events:
        if event["kind"] == "object_changed" and event["after"]:
            return AUTO_ANNOUNCEMENT_ID
        if event["kind"] == "appeared" and event["after"]["held_objects"]:
            return AUTO_ANNOUNCEMENT_ID
    return None


def build_wav_path(announcement_id, directory=ANNOUNCEMENT_WAV_DIRECTORY):
    """방송 id의 캐시 wav 경로를 만든다. 입력: 방송 id, 캐시 폴더. 출력: Path(<폴더>/<id>.wav)"""
    return Path(directory) / f"{announcement_id}.wav"


class Announcer:
    """안내방송을 작업 스레드 하나에서 차례로 재생한다(Reporter와 같은 threading.Condition 대기열 방식).

    reporter와 따로 둔 이유: Gemma 생성(2~8초)이 방송을 늦추지 않게 한다.
    메인 스레드(YOLO 루프)는 handle_events·announce_manual로 방송을 넣기만 하고 바로 돌아온다.
    알려진 한계: 재생 프로세스 손잡이 voice._playback_process는 reporter와 같이 쓰는 한 칸이다.
    그래서 수동 방송의 stop_playback()은 재생 중인 reporter 답도 끊는다(의도한 동작). 두 쪽이 동시에 재생하면
    늦게 시작한 쪽이 그 칸을 덮어 먼저 시작한 소리는 끊을 수 없게 된다.
    """

    def __init__(self, is_voice_enabled=True):
        """입력: is_voice_enabled — False면 합성·재생을 전혀 부르지 않고 글자로만 낸다(테스트용)."""
        self.is_voice_enabled = is_voice_enabled
        # 아래 대기열 값은 메인 스레드와 작업 스레드가 같이 보므로 항상 _condition 잠금 안에서만 읽고 쓴다
        self._condition = threading.Condition()
        self._pending = []  # 대기 중인 방송 id 목록(맨 앞이 다음 방송)
        self._last_auto_times = {}  # 방송 id → 마지막으로 자동 방송을 넣은 시각(now)
        self._is_working = False  # 작업 스레드가 방송 하나(출력·재생)를 처리하는 중이면 True
        self._should_stop = False
        self._worker = None

    def start(self):
        """문구 wav를 모두 합성해 캐시한 뒤 작업 스레드를 시작한다. 합성 동안 부른 스레드가 기다린다.

        문구가 바뀌어도 옛 wav가 나오지 않도록 있던 wav도 늘 다시 합성한다. 합성이 실패해도 경고 한 줄만 내고 계속한다
        (그 방송은 작업 스레드가 꺼낼 때 다시 합성을 시도한다).
        출력: 합성에 걸린 초(float). 음성이 꺼져 있으면 거의 0.
        """
        synthesis_start = time.perf_counter()
        if self.is_voice_enabled:
            for announcement_id, text in ANNOUNCEMENT_TEXTS.items():
                wav_path = build_wav_path(announcement_id)
                if not voice.synthesize_speech(text, wav_path):
                    # 이전 실행의 옛 wav(다른 문구·덜 쓰인 파일)가 남아 있으면 방송 때 그것이 나가므로 지운다
                    wav_path.unlink(missing_ok=True)
                    print(f"[announcer 경고] {announcement_id} 문구 합성 실패, 방송 때 다시 시도합니다.", flush=True)
        synthesis_seconds = time.perf_counter() - synthesis_start
        # daemon=True: stop()을 못 부르고 예외로 끝나도 이 스레드 때문에 프로그램이 안 끝나는 일이 없게 한다
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        return synthesis_seconds

    def stop(self):
        """작업 스레드를 멈춘다. 재생 중인 방송은 바로 끊고, 대기 중인 방송은 버린다. 프로그램 끝에 한 번 부른다."""
        with self._condition:
            self._should_stop = True
            self._condition.notify_all()
        voice.stop_playback()
        if self._worker is not None:
            self._worker.join()

    def announce_manual(self, announcement_id):
        """수동 방송(키 1·2·3)을 대기열 맨 앞에 넣고, 재생 중인 소리를 끊는다. 바로 돌아온다.

        입력: announcement_id — ANNOUNCEMENT_TEXTS의 id. 출력: 없음.
        같은 id가 이미 대기 중이면 빼고 맨 앞에 다시 넣는다(같은 방송이 두 번 나오지 않게).
        stop_playback()은 voice._playback_process 한 칸을 끊으므로 재생 중인 reporter 답도 끊는다 —
        Operator가 고른 방송이 먼저라서 의도한 동작이다.
        """
        with self._condition:
            if announcement_id in self._pending:
                self._pending.remove(announcement_id)
            self._pending.insert(0, announcement_id)
            self._condition.notify()
        voice.stop_playback()

    def handle_events(self, events, now):
        """이번 프레임 Event로 자동 방송을 고르고, 재방송 간격이 지났으면 대기열 뒤에 넣는다. 바로 돌아온다.

        입력: events(이번 프레임 Event 목록), now(이 프레임 시각, 실행 시작 기준 초)
        출력: 방송을 넣었으면 True, 고를 방송이 없거나 간격 안이면 False.
        간격은 Target마다가 아니라 방송 id마다 잰다: 다른 Target이 물체를 들어도 10초 안이면 다시 내보내지 않는다.
        """
        announcement_id = select_auto_announcement(events)
        if announcement_id is None:
            return False
        with self._condition:
            last_time = self._last_auto_times.get(announcement_id)
            if last_time is not None and now - last_time < REBROADCAST_SECONDS:
                return False
            self._last_auto_times[announcement_id] = now
            if announcement_id not in self._pending:
                self._pending.append(announcement_id)
            self._condition.notify()
            return True

    def wait_until_idle(self):
        """대기열이 비고 작업 스레드가 하던 방송이 끝날 때까지 부른 스레드를 기다리게 한다(수동 확인용, 메인 루프에서는 부르지 않는다)."""
        with self._condition:
            while self._pending or self._is_working:
                self._condition.wait()

    def _run_worker(self):
        """작업 스레드 본체: 방송이 들어올 때까지 기다렸다가 하나씩 출력·재생한다."""
        while True:
            with self._condition:
                while not self._pending and not self._should_stop:
                    self._condition.wait()
                if self._should_stop:
                    return
                announcement_id = self._pending.pop(0)
                self._is_working = True

            # 재생이 실패해도 스레드를 살려 두고 그 방송은 버린다(reporter와 같은 방식: 시연 중 멈추지 않는 게 먼저다)
            try:
                text = ANNOUNCEMENT_TEXTS[announcement_id]
                print(f"[방송] {text}", flush=True)
                if self.is_voice_enabled and not self._should_stop:
                    wav_path = build_wav_path(announcement_id)
                    # 시작 때 합성이 실패했으면 여기서 한 번 더 합성한다. 또 실패하면 재생하지 않는다
                    if wav_path.exists() or voice.synthesize_speech(text, wav_path):
                        voice.play_wav(wav_path)
            except Exception as error:
                print(f"[경고] 방송 실패, 이 방송은 건너뜀: {error}", flush=True)
            finally:
                # 실패해도 표시를 되돌려야 wait_until_idle()이 영원히 막히지 않는다
                with self._condition:
                    self._is_working = False
                    self._condition.notify_all()


def main():
    """세 문구를 합성한 뒤 수동 방송으로 하나씩 차례로 재생해 확인한다(실제 Piper·스피커)."""
    announcer = Announcer()
    synthesis_seconds = announcer.start()
    print(f"[합성] {synthesis_seconds:.2f}초")
    for announcement_id in ANNOUNCEMENT_TEXTS:
        announcer.announce_manual(announcement_id)
        announcer.wait_until_idle()
    announcer.stop()


if __name__ == "__main__":
    main()
