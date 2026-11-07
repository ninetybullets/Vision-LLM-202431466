# Project03 발표 — 정찰 SITREP 보조관 (5분, 8장)

발표: 2026-11-08 · 구영탄 202431466 · 슬라이드: `presentation.html`(16:9, ←/→ 이동, N 키 = 발표자 노트)

시연 준비: Jetson Orin Nano + USB 카메라(`/dev/video0`)·마이크·스피커, Jetson 모니터의 터미널에서 `cd ~/vision-llm/Project03 && source ../.venv/bin/activate && python main.py`. 카메라는 R36.5.2 업그레이드 뒤 프레임 정지 문제로 진단 중(11-08 갱신).

---

## 1. 표지

**Project03 — 정찰 SITREP 보조관**
YOLO 검출·추적 → 규칙 상태기계 → Event Log → Gemma 4 SITREP·음성 질의응답

| 과제 요구 | 이 프로젝트 |
|---|---|
| 객체 탐지 | YOLO11n(TensorRT) 사람·물체 검출 + ultralytics 내장 추적 |
| 이벤트 감지 | 규칙 상태기계 → Event 4종 |
| 유저 음성 입력 | 키 `v` → 5초 녹음 → Whisper |
| 음성 출력 | Gemma 4 E2B SITREP·답변 → Piper 한국어 음성 |

구영탄 202431466 · 2026-11-08

> 노트 (15초): "정찰 자산 카메라 앞에서 무엇이 바뀌었는지를 규칙으로 기록하고, 로컬 LLM이 그 기록만 읽어 한국어로 보고하고 말로 한 질문에 답하는 앱입니다. 과제 요구 네 가지가 표처럼 하나씩 대응됩니다."

**예상 질문**
- **Project01과 무엇이 다른가?** → 검출기를 YOLO11n TensorRT로 교체, 판정을 Zone·Held Object로 단순화, 그 위에 보고 층(Event Log → LLM → 음성)을 추가하고 Jetson에서 측정.
- **왜 "정찰" 상황인가?** → Project01(항복 의사 HUD)의 정찰 자산 상황을 잇는 설계. Operator가 화면을 계속 보지 않아도 변화와 근거를 알게 하는 것이 목표.

---

## 2. 역할 분리 — 판정은 규칙, 설명은 LLM

| 층 | 하는 일 | 성질 |
|---|---|---|
| 규칙 상태기계(`state_machine.py`) | Target State(Zone, Held Object) 판정, 바뀐 것만 Event로 | 결정론: 같은 입력 → 같은 Event 열 |
| LLM(Gemma 4 E2B, `reporter.py`) | Event Log 텍스트로 SITREP·질의응답 | 이미지 입력 없음, 위험 판정 없음(ADR 0001) |

**왜 이렇게 나눴나**
- 환각: SITREP 옆에 근거 Event 원문(**Evidence**)을 같이 출력 → 문장을 한 줄씩 대조 가능
- 재현성: 위험 여부를 LLM이 정하면 같은 장면에 다른 답이 나와 검증 불가
- 비용: Jetson 8GB 한 대에서 YOLO와 같이 돌려야 해서 매 프레임 LLM 호출·이미지 입력 불가

system 규칙(`COMMON_RULES`): 로그에 있는 사실만, 없으면 '관측되지 않음' / 한국어 낱말로만 / 위험 여부를 판정하지 마라 / 2~3문장

> 노트 (40초): "핵심 결정은 역할 분리입니다. 판정은 규칙 코드가 하고, LLM은 그 결과를 설명만 합니다. LLM은 카메라 영상을 받지 않고 Event Log와 지금 장면 한 줄만 텍스트로 읽습니다. 그래서 SITREP 옆에 근거 Event를 같이 찍어 사람이 환각을 직접 확인할 수 있고, 판정은 규칙이라 카메라 없이 테스트로 재현됩니다."

**예상 질문**
- **왜 Gemma에 이미지를 안 보여 주나?** → 8GB에서 YOLO와 동시 실행이라 비용이 안 맞고, 이미지 입력은 출력의 근거를 추적할 수 없음. 텍스트 Event는 Evidence로 대조 가능. 크롭 이미지 비교는 스트레치(미구현).
- **왜 LLM이 위험 판정을 안 하나?** → 같은 장면에 다른 답이 나올 수 있고 검증할 방법이 없음. 판정은 규칙에 두어 결정론·테스트 가능, LLM은 설명 층.

---

## 3. 파이프라인 — 파일 11개, 두 줄의 흐름

![파이프라인](images/pipeline.svg)

**메인 스레드(프레임마다)**: `camera` → `detector`(YOLO11n + 추적 id) → `state_machine`(`build_observations` → `step`) → `event_log`(메모리 목록 + `logs/events-YYYYMMDD-HHMMSS.jsonl`) + `hud`(박스·Target id·Zone·Held Object·FPS·Motion Level, 영어만) → 조립은 `main.process_frame`

**작업 스레드(Event·질문이 있을 때만)**: `reporter`(Gemma 4 E2B, SITREP + Evidence 터미널 출력) → `voice`(Whisper STT / Piper TTS)

- 그 밖: `motion`(Motion Level), `platform_config`(트랙별 경로·명령), `bench`(측정표)
- 한 코드 두 트랙: 개발·테스트는 Mac 사본, 제출본은 Jetson 전용 카메라 파이프라인(`camera.JETSON_PIPELINE`, 읽기 제한 시간 `JETSON_READ_TIMEOUT_MILLISECONDS` 3초)

> 노트 (45초): "메인 스레드는 매 프레임 검출, 상태 판정, 기록, HUD 그리기만 하고, Gemma·녹음·음성은 작업 스레드 하나가 맡습니다. 메인 스레드는 일을 큐에 넣기만 하고 바로 돌아오니까 생성이나 녹음 중에도 영상이 멈추지 않게 했습니다. 개발은 Mac 사본에서 했고, 제출본 카메라는 Jetson 파이프라인 하나만 남겼습니다."

**예상 질문**
- **왜 별도 프로세스가 아니라 스레드인가?** → 가장 단순한 구조로 시작(단일 프로세스 + 작업 스레드 하나 + 큐). Jetson에서 이 구조로 YOLO FPS가 떨어지면 별도 프로세스로 바꾸기로 함 — 동시 FPS는 3차 측정.
- **Mac 버전이 왜 있나?** → 보드 없이 흐름·로직을 먼저 확인하려고. 발표 숫자는 Jetson 것만 쓴다.

---

## 4. Event 4종과 LLM 트리거 3경로

| Event | 조건 | 임시값 |
|---|---|---|
| `appeared` | 보이는 상태가 T 동안 이어짐 → 확정 프레임의 Zone·물체를 담음 | `DEBOUNCE_SECONDS` = 1.0 |
| `gone` | Target이 T 동안 안 보임 → 마지막 Target State를 담고 삭제 | 같은 T |
| `zone_changed` | 사람 박스 높이 / 프레임 높이 ≥ 기준이면 `near`, 아니면 `far`. 새 값이 T 유지 | `NEAR_HEIGHT_RATIO` = 0.5 |
| `object_changed` | 물체 박스 중심이 사람 박스 안이면 보유. **물체 종류별로 따로** 디바운스 | `HELD_OBJECT_CLASS_NAMES` 4종(knife·bottle·cell phone·backpack) |

Event 필드는 `timestamp`, `target_id`, `kind`, `before`, `after` 다섯 개로 고정.

**LLM이 불리는 경로는 셋뿐** (매 프레임·주기 호출 없음)
1. Event 발생 → SITREP(Event Batch + Snapshot)
2. 키 `1`(지난 1분 변화)·`2`(현재 상황) → Preset Question
3. 키 `v` → 5초 녹음 → Whisper → Spoken Question. 음량 RMS < 500(`SILENCE_RMS_THRESHOLD`)이면 Gemma를 부르지 않고 "질문을 듣지 못했습니다."

- 질문은 최근 60초(`QUESTION_WINDOW_SECONDS`) Event + Snapshot을 근거로 답
- 우선순위: 질문 > 대기 중인 Event Batch. 생성은 한 번에 하나, 생성 중 쌓인 Event는 다음 SITREP 한 번에

> 노트 (45초): "Event는 네 종류뿐입니다. 디바운스는 프레임 수가 아니라 초 단위 T 하나라서 Gemma 생성 중 FPS가 떨어져도 확정까지 걸리는 시간이 같습니다. 물체는 종류별로 따로 걸러서 칼 검출이 깜빡여도 휴대폰 변화는 제때 기록됩니다. LLM은 Event가 생기거나 Operator가 물을 때만 불립니다."

**예상 질문**
- **디바운스를 왜 초 단위로 했나?** → 프레임 수로 세면 FPS가 떨어질 때 확정이 늦어짐. 상태기계는 시계를 직접 읽지 않고 시각을 인자로 받아서, 같은 입력 시퀀스로 테스트가 재현됨.
- **왜 상시 듣기가 아니라 키 + 5초 녹음인가?** → 상시 듣기는 스피커 소리·잡음에 오작동. 무음이면 Whisper가 글을 지어내서(조용한 방 녹음에서 확인) 음량 검사로 막음.

---

## 5. 측정표 — Jetson Orin Nano 8GB

보드: L4T R36.5.2(11-07 업그레이드), torch 2.8.0 CUDA, TensorRT 10.3, cv2 4.8.0

| 항목 | 값 |
|---|---|
| YOLO11n fp16 엔진 단독 | 54.2 ms/프레임(18.4 FPS)* |
| YOLO 동시 FPS·단독/동시 비율 | (3차 측정) |
| Gemma 4 E2B 로드 | 74초 |
| Gemma 생성 | 16.3 tok/s(첫 토큰 0.09s, 총 2.42s, 38토큰), YOLO 동시 실행 때도 16.3 tok/s |
| Gemma 첫 생성 | 5.7~8.5s → 시작할 때 워밍업 1회 |
| RAM 최대(Gemma + YOLO 동시) | 5377 / 7607 MB |
| Whisper base(CPU 빌드) | 5초 wav에 7.3초 |
| Piper 한국어 TTS | 2회째 3.3초(그중 import 1.3초) |
| Event → SITREP 초 | (3차 측정) |
| STT/TTS 동시 초 | (3차 측정) |
| Motion Level NumPy/CuPy ms | (3차 측정) |
| 앱 RAM | (3차 측정) |
| 자동 테스트 | unittest 49개(맥) |

\* 54.2 ms는 fp16 엔진, R36.4.7·100프레임 측정. 현재 기본 엔진은 int8(`platform_config.py` `YOLO_MODEL_PATH`)이며 int8 단독·동시 수치와 업그레이드 뒤 재측정은 (3차 측정).

> 노트 (35초): "숫자는 Jetson 실측만 적었습니다. Gemma는 YOLO와 같이 돌아도 초당 토큰이 그대로였고, RAM은 둘을 같이 올려도 8GB 안에 들어왔습니다. 첫 생성만 느려서 시작할 때 워밍업을 한 번 넣었습니다. 빈칸은 카메라 문제로 아직 못 잰 항목이고, 3차 측정에서 채웁니다."

**예상 질문**
- **무엇이 병목인가?** → 지금 숫자로는 음성 쪽: Whisper base CPU 빌드가 5초 녹음에 7.3초, Piper가 호출마다 3.3초. 동시 실행 FPS는 3차 측정 뒤 답.
- **왜 Whisper를 CPU 빌드로 돌리나?** → 11-07 사용자 결정(CUDA 컴파일 안 함). 메모리가 모자랄 때 내리는 순서도 ① Whisper CPU → ② Gemma `n_ctx` → ③ Gemma GPU 레이어이고 YOLO는 끝까지 GPU.

---

## 6. 시연

캡처: Jetson 실행 후 삽입

사용자 동작 순서(시연 리허설 순서 그대로)
1. 화면 밖 → 멀리서 들어온다 → `appeared`, SITREP 소리
2. 가까이 다가온다 → `zone_changed` far→near
3. 병(bottle)이나 휴대폰을 든다 → `object_changed`
4. 키 `1` → Preset Question(지난 1분)
5. 키 `v` → "지금 몇 명 있어?" 말하기 → Spoken Question, 답 음성
6. 소리 나는 중에 `v` 한 번 더 → 재생이 끊기고 녹음 시작
7. 물건 내려놓고 화면 밖으로 → `gone`
8. `q` → 종료

볼 것: 터미널의 Event 한 줄 → SITREP + Evidence + Snapshot, HUD의 `#id near bottle` 라벨, FPS·MOTION

> 노트 (60초): "직접 카메라 앞에서 보여 드리겠습니다. 들어오면 잠깐 뒤에 appeared가 찍히고 SITREP이 소리로 나옵니다. 터미널에서 SITREP 바로 아래 근거 Event 원문을 보시면 문장과 한 줄씩 맞는지 확인할 수 있습니다. 마지막으로 말로 질문하고, 답이 나오는 중에 v를 다시 눌러 재생이 끊기는 것까지 보여 드리겠습니다."

**예상 질문**
- **SITREP이 로그에 없는 말을 하면?** → system 규칙(로그 사실만, 없으면 '관측되지 않음')이 1차, 터미널 Evidence 대조가 2차. Mac 소수 실행에서 로그에 없는 것을 직접 물었을 때 "관측되지 않음"으로 답함(표본이 작아 일반화 안 함).
- **재생 중에 `v`를 누르면 왜 끊나?** → 끊지 않으면 스피커 소리가 마이크에 섞임. 녹음 질문을 먼저 넣고 그다음 재생을 끊는다(`submit_spoken_question` → `stop_playback`).

---

## 7. 확장 — 판정 입력·보고 문구만 바꾸면 되는 후보

상태기계·Event Log·보고 층은 그대로 두고, 클래스 목록(`HELD_OBJECT_CLASS_NAMES`)·Zone 판정 함수(`judge_zone`)·이름표(`KOREAN_LABELS`)·프롬프트(`SITREP_SYSTEM_PROMPT`)만 바꾸면 되는 후보.

영역 기반 Zone(운전석·출입구)은 상수 조정으로는 안 되고, 지금의 높이 비율 `judge_zone`(`NEAR_HEIGHT_RATIO`)을 영역 판정으로 교체해야 한다.

<!-- 사용자 검토: S2·S6 설명은 이름에서 추정한 예시 -->
| 후보 | 같은 구조를 다시 쓰는 방식 | 상태 |
|---|---|---|
| S2 운전자 휴대폰 | 예: 클래스 목록을 `cell phone`(지금 `HELD_OBJECT_CLASS_NAMES`에 있음)만 남기고 `judge_zone`을 운전석 영역 판정으로 교체 → 기존 `zone_changed`(운전석 진입) + `object_changed`(휴대폰 듦) 조합으로 '운전 중 휴대폰 사용'을 보고 | **후보로 검토, 미구현** |
| S6 자산 반출 | 예: 클래스 목록을 `laptop` 등 자산 클래스로 바꾸고(지금 목록에는 없음) `judge_zone`을 출입구 영역 판정으로 교체 → 자산을 든 Target의 기존 `zone_changed`(출입구 쪽 이동)로 '자산이 출입구로 이동'을 보고 | **후보로 검토, 미구현** |

스트레치: Event가 날 때 크롭 1장을 mmproj로 Gemma에 넣어 텍스트 SITREP과 비교(미구현)

> 노트 (30초): "판정 규칙과 보고 층이 나뉘어 있어서, 상태기계·Event Log·보고 층은 그대로 두고 클래스 목록, Zone 판정 함수, 이름표, 프롬프트만 바꾸면 다른 장면에 옮길 수 있습니다. 다만 운전석이나 출입구 같은 영역 Zone은 지금의 높이 비율 판정을 교체해야 합니다. 운전자 휴대폰, 자산 반출 두 가지를 후보로 검토했지만 구현은 하지 않았습니다."

**예상 질문**
- **물체 클래스를 바꾸면 어디를 같이 고치나?** → `detector.py`의 `HELD_OBJECT_CLASS_NAMES`와 `reporter.py`의 `KOREAN_LABELS`를 같이. 이름표가 없으면 영어 낱말이 프롬프트·Piper 음성에 그대로 들어감.
- **이미지 입력을 왜 스트레치로만 두었나?** → 추론 비용이 미확인이고 출력 근거를 추적할 수 없어서(ADR 0001). 비용 측정 자체를 결과로 쓰는 비교 실험으로만 남김.

---

## 8. 한계

- **카메라**: R36.5.2 업그레이드 뒤 1~2프레임 뒤 프레임 정지 — **진단 중, 11-08 갱신**. 그래서 YOLO 수치는 업그레이드 전 측정
- STT: Whisper `base` 체감 품질 "부족", `--prompt` 힌트(`WHISPER_PROMPT_HINT`) 효과 미측정
- TTS: Piper 호출마다 3.3초(그중 import 1.3초) — 문장마다 프로세스를 새로 띄움
- Gemma 첫 생성이 느려 워밍업 필요(시작 시간이 그만큼 늘어남)
- T보다 짧게 가려진 시간도 Target이 있던 시간으로 침 — 안 보이는 동안 Zone·Held Object는 그대로
- 임시값 미실측: `DEBOUNCE_SECONDS`, `NEAR_HEIGHT_RATIO`, `HELD_OBJECT_CLASS_NAMES`, `SILENCE_RMS_THRESHOLD`(말소리 RMS는 아직 안 잼)
- 질문 프롬프트는 최근 60초 Event를 상한 없이 넣음(SITREP만 `MAX_BATCH_EVENTS`로 자름)
- 같은 추적 id로 다시 들어오면 `#1 gone` 다음 `#1 appeared`(번호 재부여 안 함)

> 노트 (30초): "한계는 숨기지 않겠습니다. 가장 큰 문제는 보드 업그레이드 뒤 카메라가 몇 프레임 만에 멈추는 현상이고 지금 진단 중입니다. 음성 인식 품질이 부족하고, 판정 기준값들은 아직 카메라 앞에서 실측하지 못한 임시값입니다."

**예상 질문**
- **임시값은 어떻게 정할 계획인가?** → Project01처럼 카메라 캡처 실측으로. 테스트는 상수를 이름으로 불러 경계 바로 아래·위를 검사해서 값이 바뀌어도 따라옴.
- **질문 프롬프트에 상한이 없으면 문제는?** → 60초 안에 Event가 많이 쌓이면 프롬프트가 길어져 `n_ctx`를 넘을 수 있음. 생성 실패는 경고 한 줄 후 그 일만 건너뛰고 앱은 계속 돈다.
