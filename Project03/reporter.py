"""reporter: Event Batch·질문과 Snapshot을 프롬프트로 바꾸고, 작업 스레드에서 Gemma로 SITREP·답변을 쓰고, 답변만 소리로 읽는다(SITREP은 터미널만).

앞부분은 프롬프트 생성 순수 함수(모델 없음), 뒷부분은 Gemma 호출 + 작업 스레드 하나(녹음 질문·음성 출력 포함) + Evidence 출력이다.
테스트: 코드 폴더에서 python -m unittest tests.test_prompt
수동 확인(실제 Gemma, SITREP·답을 스피커로 읽음): .venv/bin/python reporter.py
"""

import os
import threading
import time

import voice
from event_log import EventLog, format_event
from platform_config import GEMMA_MODEL_PATH, RECORD_SECONDS

# Event Batch에 넣는 최대 Event 수 N. 넘치면 최신 N건만 넣고 "이전 M건 생략"을 붙인다.
MAX_BATCH_EVENTS = 10  # 임시값(근거 없음), 실측 후 교체

# 질문 프롬프트에 넣는 Event의 시간 범위(초). 스펙 「질문 입력」의 "최근 60초".
QUESTION_WINDOW_SECONDS = 60

# 한국어 이름표: Piper가 영어 낱말을 못 읽으므로 Event·Snapshot의 영어 값을 한국어로 바꾼다.
# 스펙 「임시 결정」의 Zone 두 값 + 임시 Held Object 클래스 4종. 없는 값은 영어 원문 그대로 둔다.
# 물체 목록은 detector.py HELD_OBJECT_CLASS_NAMES와 맞춘다.
KOREAN_LABELS = {
    "near": "가까운 구역",
    "far": "먼 구역",
    "knife": "칼",
    "bottle": "병",
    "cell phone": "휴대폰",
    "backpack": "배낭",
}

NO_OBJECT_TEXT = "물체 없음"

# system 공통 규칙: 환각 방지 1차(로그 사실만), Piper용 한국어 낱말, ADR 0001(위험 판정 금지).
COMMON_RULES = (
    "규칙:\n"
    "1. 로그에 있는 사실만 써라. 로그에 없는 것은 '관측되지 않음'이라고 답하라.\n"
    "2. 한국어 낱말로만 써라. 영어 낱말을 쓰지 마라.\n"
    "3. 위험 여부를 판정하지 마라. 관측된 변화만 설명하라.\n"
    "4. 2~3문장으로 써라."
)

SITREP_SYSTEM_PROMPT = "너는 정찰 상황 보고를 쓰는 보조원이다. 변화 기록과 현재 상태를 읽고 상황 보고를 쓴다.\n" + COMMON_RULES

QUESTION_SYSTEM_PROMPT = "너는 정찰 상황 질문에 답하는 보조원이다. 변화 기록과 현재 상태만 근거로 답한다.\n" + COMMON_RULES


def translate_label(value):
    """영어 값(Zone·클래스 이름)을 한국어 이름표로 바꾼다. 이름표에 없으면 원문 그대로 돌려준다."""
    return KOREAN_LABELS.get(value, value)


def describe_target_id(target_id):
    """Target id를 "표적 N번" 글로 바꾼다."""
    return f"표적 {target_id}번"


def describe_held_objects(held_objects):
    """Held Object 목록을 "칼 보유" / "칼, 병 보유" / "물체 없음" 글로 바꾼다."""
    if not held_objects:
        return NO_OBJECT_TEXT
    korean_names = [translate_label(class_name) for class_name in held_objects]
    return ", ".join(korean_names) + " 보유"


def describe_target_state(target_state):
    """Target State dict를 "가까운 구역, 칼 보유" 글로 바꾼다."""
    zone_text = translate_label(target_state["zone"])
    return f"{zone_text}, {describe_held_objects(target_state['held_objects'])}"


def describe_snapshot(targets):
    """Snapshot 재료 {target_id: Target State}를 한 줄 글로 바꾼다.

    입력: 지금 확정된 Target 전부. 출력: "현재: 표적 2명. 표적 1번 가까운 구역, 칼 보유. ..." 또는 "현재: 표적 없음."
    """
    if not targets:
        return "현재: 표적 없음."
    parts = [f"현재: 표적 {len(targets)}명."]
    for target_id in sorted(targets):
        parts.append(f"{describe_target_id(target_id)} {describe_target_state(targets[target_id])}.")
    return " ".join(parts)


def describe_event(event, now):
    """Event 한 건을 "12초 전: 표적 1번 구역 변경, 이전 먼 구역, 이후 가까운 구역." 한 줄로 바꾼다.

    입력: Event dict, 프롬프트를 만드는 시각 now(실행 시작 기준 초). 상대 시간은 소수점 아래를 버린 정수 초.
    """
    seconds_ago = int(now - event["timestamp"])
    target_text = describe_target_id(event["target_id"])
    kind = event["kind"]
    if kind == "appeared":
        body = f"출현, {describe_target_state(event['after'])}"
    elif kind == "gone":
        body = f"이탈, 마지막 상태 {describe_target_state(event['before'])}"
    elif kind == "zone_changed":
        # "~에서 ~으로" 대신 이전/이후로 써서 받침에 따른 조사(으로/로) 처리를 피한다.
        body = f"구역 변경, 이전 {translate_label(event['before'])}, 이후 {translate_label(event['after'])}"
    elif kind == "object_changed":
        body = f"물체 변경, 이전 {describe_held_objects(event['before'])}, 이후 {describe_held_objects(event['after'])}"
    else:
        # 스펙의 Event 종류는 넷뿐이다. 다른 값이 오면 멈추지 않고 원문을 그대로 넣는다.
        body = str(kind)
    return f"{seconds_ago}초 전: {target_text} {body}."


def describe_events(events, now):
    """Event 목록을 줄 목록으로 바꾼다. 0건이면 "기록 없음" 한 줄."""
    if not events:
        return ["- 기록 없음."]
    return [f"- {describe_event(event, now)}" for event in events]


def build_sitrep_prompt(events, targets, now):
    """Event Batch + Snapshot으로 SITREP 프롬프트를 만든다.

    입력: events(시간순 Event 목록), targets({target_id: Target State}), now(지금 시각, 초).
    출력: (system 글, user 글). 최신 MAX_BATCH_EVENTS건만 넣고 넘친 수는 "이전 M건 생략"으로 적는다.
    """
    omitted_count = max(0, len(events) - MAX_BATCH_EVENTS)
    latest_events = events[omitted_count:]
    lines = ["변화 기록:"]
    if omitted_count > 0:
        lines.append(f"- 이전 {omitted_count}건 생략.")
    lines.extend(describe_events(latest_events, now))
    lines.append(describe_snapshot(targets))
    lines.append("위 기록으로 상황 보고를 써라.")
    return SITREP_SYSTEM_PROMPT, "\n".join(lines)


def build_question_prompt(question, events, targets, now):
    """질문 글 + 최근 QUESTION_WINDOW_SECONDS초 Event + Snapshot으로 질문 프롬프트를 만든다.

    Preset Question·Spoken Question 모두 이 함수를 쓴다.
    입력: question(질문 글), events(Event Log 전체 또는 일부), targets, now(질문 시각 = 키를 누른 순간).
    출력: (system 글, user 글). now보다 뒤의 Event는 넣지 않는다(질문 시점의 Snapshot과 어긋나지 않게).
    """
    recent_events = [event for event in events if 0 <= now - event["timestamp"] <= QUESTION_WINDOW_SECONDS]
    lines = [f"질문: {question}", f"최근 {QUESTION_WINDOW_SECONDS}초 변화 기록:"]
    lines.extend(describe_events(recent_events, now))
    lines.append(describe_snapshot(targets))
    lines.append("위 기록으로 질문에 답하라.")
    return QUESTION_SYSTEM_PROMPT, "\n".join(lines)


# ---- Gemma 호출 ----

# Gemma 설정은 board-check.md 4번 gemma_check.py·강의 05 노트북과 같다.
# 메모리가 모자랄 때 내리는 순서(스펙 「GPU·메모리」, 코드로 자동 전환하지 않고 사람이 상수를 바꾼다):
#   ① Whisper를 CPU 빌드로 → ② GEMMA_CONTEXT_TOKENS를 1024로 → ③ GEMMA_GPU_LAYERS를 줄이기. YOLO는 끝까지 GPU.
GEMMA_CONTEXT_TOKENS = 2048  # n_ctx: 프롬프트 + 생성 글이 들어갈 토큰 수(05 노트북 설정)
GEMMA_GPU_LAYERS = -1  # n_gpu_layers: -1은 모든 레이어를 GPU(Mac은 Metal, Jetson은 CUDA)에 올린다는 뜻 — Jetson(CUDA 빌드) 경로: 보드 미검증
GEMMA_BATCH_TOKENS = 32  # n_batch·n_ubatch: 프롬프트를 한 번에 처리하는 토큰 수(gemma_check.py와 같음)
MAX_GENERATED_TOKENS = 120  # 생성 토큰 상한(gemma_check.py 값). 임시값, 실측 후 교체
# 워밍업: start()에서 짧은 생성 1회를 미리 돌려 첫 SITREP이 느려지지 않게 한다(보드 2차 B 실측 근거는 start() docstring)
WARMUP_USER_TEXT = "준비 확인. '준비됨'이라고만 답하라."
WARMUP_MAX_TOKENS = 8  # 답은 버리므로 몇 토큰이면 충분하다
# ggml-cuda의 CUDA 그래프 캡처를 끄는 환경변수(값은 무엇이든 있으면 꺼짐 — ggml common.cuh의 getenv != nullptr 검사).
# 2026-11-08 Jetson 실측: YOLO(TensorRT)와 같은 프로세스에서 Gemma가 생성을 시작하면 bench fps·main 둘 다
# `ggml-cuda.cu:109: CUDA error` → Aborted. 유력 원인은 생성 중 그래프 캡처에 다른 스레드(TensorRT)의 CUDA 호출이 섞이는 것이라
# 캡처를 끈다(오류 이름은 로그에 안 남아 미확정). 그래프 없이는 tok/s가 떨어질 수 있으므로 bench로 다시 잰다.
DISABLE_CUDA_GRAPHS_ENVIRONMENT_NAME = "GGML_CUDA_DISABLE_GRAPHS"


def load_gemma(model_path=GEMMA_MODEL_PATH):
    """Gemma GGUF를 불러와 llama_cpp.Llama 객체를 돌려준다.

    입력: model_path — GGUF 경로(기본: platform_config의 GEMMA_MODEL_PATH)
    출력: Llama 객체. llama_cpp는 여기서 불러온다 — `import reporter`만으로 모델·Metal이 열리지 않게(tests/test_prompt.py의 ImportTest가 확인).
    불러오기 전에 CUDA 그래프를 끄는 환경변수를 넣는다(DISABLE_CUDA_GRAPHS_ENVIRONMENT_NAME 주석). 셸에서 이미 정했으면 그 값을 둔다.
    """
    os.environ.setdefault(DISABLE_CUDA_GRAPHS_ENVIRONMENT_NAME, "1")
    from llama_cpp import Llama

    # verbose=False: 로드할 때 stderr로 나오는 ggml_metal_... 로그를 줄인다
    return Llama(
        model_path=str(model_path),
        n_gpu_layers=GEMMA_GPU_LAYERS,
        n_ctx=GEMMA_CONTEXT_TOKENS,
        n_batch=GEMMA_BATCH_TOKENS,
        n_ubatch=GEMMA_BATCH_TOKENS,
        verbose=False,
    )


def generate_text(model, system_text, user_text, max_tokens=MAX_GENERATED_TOKENS):
    """(system, user) 프롬프트로 Gemma 글 하나를 생성한다. 스레드 없이 바로 부르는 함수(bench.py도 쓴다).

    입력: model(load_gemma 결과), system_text, user_text(build_sitrep_prompt·build_question_prompt 결과),
    max_tokens — 생성 토큰 상한(기본 MAX_GENERATED_TOKENS, 워밍업은 WARMUP_MAX_TOKENS)
    출력: 생성된 글(앞뒤 공백 제거). 생성이 끝날 때까지 이 함수를 부른 스레드가 기다린다.
    """
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    response = model.create_chat_completion(messages=messages, max_tokens=max_tokens)
    return response["choices"][0]["message"]["content"].strip()


# ---- 작업 스레드 ----

# 녹음 질문(v)이 빈 인식 결과일 때 Gemma 대신 내는 글(스펙 「음성」)
NOT_HEARD_TEXT = "질문을 듣지 못했습니다."


class Reporter:
    """Gemma를 작업 스레드 하나에서 돌려 SITREP·질문 답을 터미널에 출력하고 Piper로 읽는다.

    메인 스레드(YOLO 루프)는 submit_events·submit_question·submit_spoken_question으로 일을 넣기만 하고 바로 돌아온다.
    녹음·Whisper·Piper·재생도 이 작업 스레드 안에서 기다린다(스펙 「임시 결정」 동시 실행 방식).
    대기열 규칙(스펙 「reporter 트리거와 순서」): 생성은 한 번에 하나, 대기 중인 질문이 Event Batch보다 먼저,
    생성 중 들어온 Event는 다음 SITREP 하나로 묶는다(Snapshot은 가장 최근 것).
    """

    def __init__(self, event_log, get_now, is_voice_enabled=True):
        """입력: event_log — event_log.EventLog(질문 때 전체를 읽는다), get_now — main과 같은 실행 기준 상대 초 함수,
        is_voice_enabled — False면 합성·재생·녹음을 전혀 부르지 않고 글자로만 낸다(bench가 걸린 초에 재생 시간이 섞이지 않게).
        """
        self.event_log = event_log
        self.get_now = get_now
        self.is_voice_enabled = is_voice_enabled
        self.model = None
        # SITREP마다 "Event를 넣은 시각 → SITREP 출력" 걸린 초를 쌓는다(bench.py fps가 읽는다)
        self.sitrep_seconds = []
        # 아래 대기열 값은 메인 스레드와 작업 스레드가 같이 보므로 항상 _condition 잠금 안에서만 읽고 쓴다
        self._condition = threading.Condition()
        # 질문 한 건 = (질문 글 — 녹음 질문이면 None, Snapshot 재료, 넣은 시각(perf_counter), 질문 시각(get_now))
        self._pending_questions = []
        self._pending_events = []  # 다음 Event Batch가 될 Event 목록
        self._pending_targets = {}  # 가장 최근에 받은 Snapshot 재료
        self._first_event_time = None  # 대기 중인 Event Batch에서 가장 먼저 넣은 시각
        self._is_working = False  # 작업 스레드가 일 하나(녹음·인식·생성·합성·재생)를 처리하는 중이면 True
        self._is_question_active = False
        self._should_stop = False
        self._worker = None

    def start(self):
        """Gemma를 불러오고 워밍업 생성 1회 뒤 작업 스레드를 시작한다.
        모델 로드(Mac 약 3~4초, Jetson 74초(1차 점검 4번))와 워밍업(약 8초) 동안 부른 스레드가 기다린다.

        워밍업 이유: 보드 2차 점검 B에서 Gemma 첫 생성은 첫 토큰 5.7초·총 8.5초, 두 번째부터 0.09초·2.4초였다.
        그 첫 생성 비용을 시연 첫 SITREP 대신 시작할 때 치른다. 워밍업이 실패해도 경고 한 줄만 내고 계속한다.
        출력: 모델 로드에 걸린 초(워밍업 초는 넣지 않는다 — bench·main이 "로드 초"로 쓴다).
        """
        load_start = time.perf_counter()
        self.model = load_gemma()
        load_seconds = time.perf_counter() - load_start
        warmup_start = time.perf_counter()
        try:
            # 출력 글은 버린다. 실제 SITREP과 같은 system 프롬프트를 써서 같은 경로를 한 번 지나게 한다
            generate_text(self.model, SITREP_SYSTEM_PROMPT, WARMUP_USER_TEXT, max_tokens=WARMUP_MAX_TOKENS)
            print(f"[reporter] 워밍업 {time.perf_counter() - warmup_start:.1f}초", flush=True)
        except Exception as error:
            # 생성 실패와 같은 취급: 멈추지 않고 넘어간다(_run_worker와 같은 방식)
            print(f"[경고] 워밍업 생성 실패, 건너뜀: {error}", flush=True)
        # daemon=True: stop()을 못 부르고 예외로 끝나도 이 스레드 때문에 프로그램이 안 끝나는 일이 없게 한다
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        return load_seconds

    def stop(self):
        """작업 스레드를 멈추고 모델을 닫는다. 프로그램 끝에 한 번 부른다.

        재생 중인 소리는 바로 끊고, 대기 중인 일은 버린다. 작업 스레드는 단계 사이(녹음 전·생성 전·재생 전)에
        멈춤 표시를 보고 남은 단계를 건너뛴다. 단 이미 시작된 녹음(5초)·Whisper·생성·합성은 끝까지 기다리므로
        그만큼(최대 수 초) q 뒤 종료가 늦어질 수 있다.
        이유: 모델을 쥔 스레드가 살아 있는 채로 프로세스가 끝나면 Mac에서 ggml-metal GGML_ASSERT로
        비정상 종료(exit 134)되는 것을 확인했다. 그래서 스레드를 끝내고 model.close()로 먼저 놓는다.
        """
        with self._condition:
            self._should_stop = True
            self._condition.notify_all()
        # 재생 시작 전이면 효과가 없지만, 그때는 작업 스레드가 재생 직전에 멈춤 표시를 보고 재생을 건너뛴다
        voice.stop_playback()
        self._worker.join()
        self.model.close()

    def submit_events(self, events, targets):
        """Event 목록과 그 시점 Snapshot 재료를 대기열에 넣고 바로 돌아온다.

        입력: events(이번 프레임의 Event 목록), targets({target_id: Target State}). 출력: 없음.
        Event dict는 고치지 않고 목록만 복사해 둔다.
        """
        if not events:
            return
        with self._condition:
            if not self._pending_events:
                self._first_event_time = time.perf_counter()
            self._pending_events.extend(events)
            self._pending_targets = dict(targets)
            self._condition.notify()

    def submit_question(self, question, targets):
        """질문 글(Preset Question)과 Snapshot 재료를 대기열에 넣고 바로 돌아온다.

        입력: question(질문 글, None이면 녹음 질문), targets({target_id: Target State}). 출력: 없음.
        질문 시각(get_now)을 같이 저장한다: 답의 기준 시각은 키를 누른 순간이다(Snapshot도 그 프레임 것).
        """
        with self._condition:
            self._pending_questions.append((question, dict(targets), time.perf_counter(), self.get_now()))
            self._is_question_active = True
            self._condition.notify()

    def submit_spoken_question(self, targets):
        """녹음 질문(v) 요청을 대기열에 넣고 바로 돌아온다. 녹음·Whisper는 작업 스레드가 꺼낼 때 한다.

        입력: targets(v를 누른 프레임의 Snapshot 재료). 출력: 없음.
        """
        self.submit_question(None, targets)

    def is_question_busy(self):
        """질문이 대기 중이거나 처리 중(녹음·STT·생성, 답 글 출력 전)이면 True. main이 v 무시에 쓴다.

        답 재생이 시작되기 전에 False로 돌아간다: 그래야 답 재생 중의 v가 "재생을 끊고 녹음"이 된다.
        """
        with self._condition:
            return self._is_question_active

    def wait_until_idle(self):
        """대기열이 비고 작업 스레드가 하던 일(녹음·인식·생성·합성·재생)이 끝날 때까지 부른 스레드를 기다리게 한다(수동 확인·bench용, 메인 루프에서는 부르지 않는다)."""
        with self._condition:
            while self._pending_questions or self._pending_events or self._is_working:
                self._condition.wait()

    def _take_next_job(self):
        """대기열에서 다음 일을 꺼낸다. 질문이 먼저, 없으면 쌓인 Event 전부를 Event Batch 하나로.

        출력: ("question", 질문 글, Snapshot 재료, 넣은 시각, 질문 시각) 또는
            ("sitrep", Event Batch, Snapshot 재료, 넣은 시각, None). _condition 잠금 안에서만 부른다.
        """
        if self._pending_questions:
            question, targets, submit_time, question_time = self._pending_questions.pop(0)
            return "question", question, targets, submit_time, question_time
        batch = self._pending_events
        self._pending_events = []
        return "sitrep", batch, self._pending_targets, self._first_event_time, None

    def _run_worker(self):
        """작업 스레드 본체: 일이 들어올 때까지 기다렸다가 하나씩 생성·출력·재생한다."""
        while True:
            with self._condition:
                while not self._pending_questions and not self._pending_events and not self._should_stop:
                    self._condition.wait()
                # 일을 꺼내기 직전의 멈춤 확인: 녹음 전·생성 전 확인이 여기서 된다
                if self._should_stop:
                    return
                job_kind, content, targets, submit_time, question_time = self._take_next_job()
                self._is_working = True

            # 생성이 실패해도(예: n_ctx 초과, GPU 메모리 부족) 스레드를 살려 두고 그 일은 버린다.
            # 다시 시도하지 않는 이유: 같은 입력이면 같은 이유로 또 실패해 대기열이 막힌다. 시연 중 멈추지 않는 게 먼저다.
            try:
                if job_kind == "question":
                    self._answer_question(content, targets, submit_time, question_time)
                else:
                    self._write_sitrep(content, targets, submit_time)
            except Exception as error:
                print(f"[경고] {job_kind} 생성 실패, 이 일은 건너뜀: {error}", flush=True)
            finally:
                # 실패해도 표시를 되돌려야 is_question_busy()·wait_until_idle()이 영원히 막히지 않는다
                with self._condition:
                    self._is_working = False
                    if job_kind == "question" and not self._pending_questions:
                        self._is_question_active = False
                    # wait_until_idle로 기다리는 스레드를 깨운다
                    self._condition.notify_all()

    def _write_sitrep(self, batch, targets, submit_time):
        """Event Batch로 SITREP을 생성하고 SITREP → Evidence(+ Snapshot) → 걸린 초 순으로 출력한다. 소리로는 읽지 않는다."""
        system_text, user_text = build_sitrep_prompt(batch, targets, self.get_now())
        sitrep = generate_text(self.model, system_text, user_text)
        # 걸린 초는 글자 출력 전·재생 전에 잰다(정의: Event를 넣은 시각 → SITREP 글)
        elapsed_seconds = time.perf_counter() - submit_time
        self.sitrep_seconds.append(elapsed_seconds)

        print(f"[SITREP] {sitrep}", flush=True)
        # Evidence는 Event Batch 전부를 영어 원문으로 낸다. 프롬프트에서 생략된 앞부분도 근거 확인용으로 함께 낸다.
        omitted_count = max(0, len(batch) - MAX_BATCH_EVENTS)
        print(f"[Evidence] {len(batch)}건" + (f" (앞 {omitted_count}건은 프롬프트에서 생략)" if omitted_count else ""),
              flush=True)
        for event in batch:
            print(f"  {format_event(event)}", flush=True)
        # 프롬프트에 넣은 것과 같은 Snapshot 글(Event 줄과 구분되게 줄 머리를 단다)
        print(f"  [Snapshot] {describe_snapshot(targets)}", flush=True)
        print(f"[걸린 초] {elapsed_seconds:.2f}초 (Event를 넣은 시각부터 SITREP 출력까지)", flush=True)
        # SITREP은 터미널로만 낸다(2026-11-08 결정: 스피커는 안내방송 전용, announcer.py). 질문 답(_answer_question)은 계속 읽는다

    def _answer_question(self, question, targets, submit_time, question_time):
        """질문에 답을 생성해 질문·답을 출력한 뒤 답을 읽는다. 녹음 질문(question이 None)이면 먼저 녹음·인식한다.

        입력: question(질문 글, 녹음 질문이면 None), targets(키를 누른 프레임의 Snapshot 재료),
            submit_time(넣은 시각, 걸린 초용), question_time(키를 누른 순간의 get_now 값). 출력: 없음.
        Event Log 전체를 build_question_prompt에 넘기면 질문 시각 기준 최근 60초로 걸러진다.
        """
        if question is None:
            question = self._record_question()
            if not question:
                # 빈 인식 결과: Gemma를 부르지 않고 정해진 글을 낸다(스펙 「음성」)
                print(f"[답] {NOT_HEARD_TEXT}", flush=True)
                self._clear_question_busy()
                self._speak(NOT_HEARD_TEXT)
                return
            print(f"[인식] {question}", flush=True)
            # 녹음·인식(약 5초) 사이에 stop()이 불렸으면 생성 전에 멈춘다
            if self._is_stopping():
                return
        # 기준 시각은 키를 누른 순간(question_time). 녹음 질문은 그래서 5초 이상 전의 상황을 답한다
        system_text, user_text = build_question_prompt(question, self.event_log.read_events(), targets, question_time)
        answer = generate_text(self.model, system_text, user_text)
        elapsed_seconds = time.perf_counter() - submit_time
        print(f"[질문] {question}", flush=True)
        print(f"[답] {answer}", flush=True)
        print(f"[걸린 초] {elapsed_seconds:.2f}초 (질문을 넣은 시각부터 답 출력까지)", flush=True)
        self._clear_question_busy()
        self._speak(answer)

    def _record_question(self):
        """녹음 질문(v)의 질문 글을 얻는다: 안내 출력 → 5초 녹음 → (무음 검사) → Whisper.

        출력: 인식된 글. 무음·녹음·인식 실패이거나 음성이 꺼져 있으면 빈 문자열.
        """
        if not self.is_voice_enabled:
            return ""
        # SITREP 생성 중에 v를 누르면 이 일은 그 SITREP이 끝난 뒤에 꺼내지므로, 녹음과 이 안내도 그때 나온다
        print(f"[녹음 시작] {RECORD_SECONDS}초 동안 질문을 말하세요.", flush=True)
        return voice.speech_to_text()

    def _clear_question_busy(self):
        """답 글을 낸 뒤·재생 전에 "질문 처리 중" 표시를 내린다(답 재생 중의 v는 받아서 재생을 끊게). 출력: 없음."""
        with self._condition:
            if not self._pending_questions:
                self._is_question_active = False

    def _speak(self, text):
        """글을 Piper로 합성한 뒤 재생한다. 음성이 꺼져 있거나 건너뛸 조건이면 합성·재생을 하지 않는다.

        voice.text_to_speech() 대신 합성과 재생을 나눠 부른다: stop_playback()은 재생 시작 전이면 효과가 없어서,
        합성(약 1초) 중에 v·q가 눌렸는지를 재생 직전에 한 번 더 확인해야 한다.
        입력: text — 읽을 글(SITREP 또는 답. Evidence는 넘기지 않는다). 출력: 없음. 실패는 voice가 경고 한 줄로 알린다.
        """
        if not self.is_voice_enabled or self._should_skip_speech():
            return
        if not voice.synthesize_speech(text):
            return
        # 이 확인과 재생 시작 사이의 아주 짧은 틈에 v가 눌리면 그 재생은 끊기지 않고 끝까지 나온다(구조를 키우지 않으려고 남겨 둠)
        if self._should_skip_speech():
            return
        voice.play_wav()

    def _should_skip_speech(self):
        """재생을 건너뛸지 판단한다. 녹음 질문(v)이 대기 중이거나 stop()이 불렸으면 True.

        Preset Question(1·2)만 대기 중이면 False: 스펙은 v만 소리를 끊는다고 했으므로 재생 뒤에 처리한다.
        """
        with self._condition:
            has_spoken_question = any(question is None for question, _, _, _ in self._pending_questions)
            return self._should_stop or has_spoken_question

    def _is_stopping(self):
        """stop()이 불렸으면 True(작업 스레드가 단계 사이에 확인한다)."""
        with self._condition:
            return self._should_stop


# ---- 수동 확인 ----

# reporter 수동 확인 전용 질문 두 개(main.py는 쓰지 않는다. 키 1·2는 이제 announcer의 수동 방송)
DEMO_QUESTION_RECENT = "지난 1분 동안 무엇이 바뀌었나?"
DEMO_QUESTION_CURRENT = "지금 상황은 어떤가?"
# Event 묶음 확인용: 첫 SITREP 생성이 시작되도록 잠깐 기다리는 초(생성은 약 2초 걸림)
DEMO_WAIT_SECONDS = 0.5


def submit_and_measure(reporter, event_log, events, targets):
    """Event를 Event Log에 기록하고 reporter에 넣은 뒤, 넣기 함수가 걸린 초를 출력한다(main이 할 일을 흉내)."""
    for event in events:
        event_log.append(event)
        print(f"[Event] {format_event(event)}")
    submit_start = time.perf_counter()
    reporter.submit_events(events, targets)
    print(f"[넣기] submit_events 걸린 초: {time.perf_counter() - submit_start:.6f}")


def main():
    """가짜 Event·질문으로 Reporter 흐름을 실제 Gemma로 확인한다.

    순서: ① Event Log가 빈 상태의 질문 → ② Event 묶음 A → A 생성 중 Event 두 번 + 질문 하나
    (기대: A SITREP → 질문 답 → 묶인 SITREP 한 번) → ③ 현재 상황 질문.
    """
    run_start = time.perf_counter()

    def get_now():
        return time.perf_counter() - run_start

    event_log = EventLog()
    reporter = Reporter(event_log, get_now)
    load_seconds = reporter.start()
    print(f"[모델 로드] {load_seconds:.2f}초")

    print("\n=== ① Event Log가 빈 상태의 질문 ===")
    reporter.submit_question(DEMO_QUESTION_RECENT, {})
    reporter.wait_until_idle()

    print("\n=== ② Event 묶음 A, 생성 중 Event 두 번 + 질문 ===")
    targets = {1: {"zone": "far", "held_objects": []}, 2: {"zone": "far", "held_objects": []}}
    submit_and_measure(reporter, event_log, [
        {"timestamp": get_now(), "target_id": 1, "kind": "appeared", "before": None, "after": targets[1]},
        {"timestamp": get_now(), "target_id": 2, "kind": "appeared", "before": None, "after": targets[2]},
    ], targets)
    time.sleep(DEMO_WAIT_SECONDS)

    targets = {1: {"zone": "near", "held_objects": []}, 2: {"zone": "far", "held_objects": []}}
    submit_and_measure(reporter, event_log, [
        {"timestamp": get_now(), "target_id": 1, "kind": "zone_changed", "before": "far", "after": "near"},
    ], targets)
    targets = {1: {"zone": "near", "held_objects": ["knife"]}, 2: {"zone": "far", "held_objects": []}}
    submit_and_measure(reporter, event_log, [
        {"timestamp": get_now(), "target_id": 1, "kind": "object_changed", "before": [], "after": ["knife"]},
    ], targets)
    submit_start = time.perf_counter()
    reporter.submit_question(DEMO_QUESTION_RECENT, targets)
    print(f"[넣기] submit_question 걸린 초: {time.perf_counter() - submit_start:.6f}")
    print(f"[질문 처리 중?] {reporter.is_question_busy()}")
    reporter.wait_until_idle()
    print(f"[질문 처리 중?] {reporter.is_question_busy()}")

    print("\n=== ③ 현재 상황 질문 ===")
    reporter.submit_question(DEMO_QUESTION_CURRENT, targets)
    reporter.wait_until_idle()

    print(f"\n[SITREP 걸린 초 목록] {[round(seconds, 2) for seconds in reporter.sitrep_seconds]}")
    reporter.stop()
    event_log.close()
    print(f"[JSONL] {event_log.path}")


if __name__ == "__main__":
    main()
