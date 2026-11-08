# Project03 — 정찰 SITREP 보조관: YOLO 검출·추적 → 규칙 상태기계 → Event Log → Gemma 4 SITREP(터미널) + 현장 안내방송(스피커)

최종 프로젝트(Vision + LLM, 음성 필수). 정찰 자산(드론·로봇)의 카메라로 사람을 지켜보면서 **변화가 생긴 순간만** 기록하고, 그 기록을 근거로 보드 위 로컬 LLM이 상황 보고(SITREP)를 쓰는 상황을 가정했다.
**YOLO11n(TensorRT)으로 사람과 물체를 검출·추적하고 → 규칙 코드가 사람마다 Zone(near/far)·Held Object를 정해 1초 디바운스 뒤 바뀐 것만 Event로 만들고 → Event Log(JSONL)에 쌓는다.
Event가 생기면 Gemma 4 E2B가 Event 원문(Evidence)과 함께 SITREP을 터미널에 쓰고(Operator용), 스피커는 카메라 앞 사람에게 미리 합성한 안내방송("들고 있는 물건을 바닥에 내려놓으세요" 등)을 낸다.**
위험·방송 판단은 전부 규칙이 하고, LLM은 Event Log를 읽고 설명만 한다(ADR `docs/0001-llm-reads-event-log-only.md`).

| Event (`kind`) | 조건 | `before` → `after` |
|---|---|---|
| `appeared` | 새 추적 id가 **1.0초**(`DEBOUNCE_SECONDS`) 계속 보이면 Target으로 확정 | `None` → 첫 Target State |
| `gone` | Target이 1.0초 계속 안 보이면 이탈 확정 | 마지막 Target State → `None` |
| `zone_changed` | 사람 박스 높이 / 프레임 높이 ≥ **0.5**(`NEAR_HEIGHT_RATIO`)면 `near`, 미만이면 `far`. 바뀐 값이 1.0초 유지 | `"far"` → `"near"` 등 |
| `object_changed` | Held Object 클래스(**knife·bottle·cell phone·backpack**, `HELD_OBJECT_CLASS_NAMES`) 박스 중심이 사람 박스 안이면 그 사람이 보유. 종류별로 1.0초 디바운스 | 보유 목록 → 보유 목록 |

위 숫자·클래스 목록은 전부 **임시값**(근거 없음 또는 프로토타입 기본값, 실측 후 교체 예정)이다. Event는 `{"timestamp", "target_id", "kind", "before", "after"}` 다섯 필드이고, `timestamp`는 실행 시작 기준 상대 초다.

| 안내방송 id | 문구 (카메라 앞 사람에게) | 자동 조건 | 수동 키 |
|---|---|---|---|
| `put_down` | 들고 있는 물건을 바닥에 내려놓으세요. | Event가 "Target이 물건을 들고 있음"을 보일 때(`object_changed`의 `after`가 비어 있지 않음, 또는 `appeared`의 `held_objects`가 있음). 같은 문구는 **10초**(`REBROADCAST_SECONDS`, 임시값) 안에 다시 내지 않음 | `1` |
| `hands_up` | 양손을 펼쳐 어깨 위로 드세요. | 없음(수동만, 자동은 미구현 — 창의성 및 확장성 참고) | `2` |
| `hold_still` | 그 자세를 유지하세요. | 없음(수동만) | `3` |

깜빡임 거르기는 상태기계의 1.0초 디바운스를 그대로 쓴다. 1초 안에 나타났다 사라진 물체는 Event가 되지 않으므로 방송도 나가지 않는다.

**LLM 트리거.** Gemma는 두 경우에만 부른다. ① Event 발생 → SITREP(터미널 글자만), ② `v` 키 → 5초 녹음 → Whisper → 질문 답(터미널 + Piper로 읽음). 생성은 한 번에 하나이고, 생성 중 쌓인 Event는 다음 SITREP 하나(Event Batch, 최대 10건 `MAX_BATCH_EVENTS`, 임시값)로 묶는다. 대기 중인 질문이 Event Batch보다 먼저다. 매 프레임·주기적 LLM 호출은 없다. 녹음 음량(RMS)이 500(`SILENCE_RMS_THRESHOLD`, 임시값) 미만이면 Whisper·Gemma를 부르지 않고 "질문을 듣지 못했습니다."를 낸다.

## 파일 구조

```
Project03/
├── main.py               # 메인 루프: 카메라 → 검출 → Target State 판정 → 상태기계 → Event Log + 터미널 한 줄 → reporter(SITREP)·announcer(안내방송) → HUD, 키 입력
├── camera.py             # Jetson USB 카메라 /dev/video0 열기(GStreamer MJPG 1280x720 30fps, 파이프라인은 JETSON_PIPELINE 한 곳)
├── detector.py           # YOLO11n 검출 + ultralytics 내장 추적 → 사람 [(track_id, 박스)], Held Object 후보 물체 [(클래스, 박스)]
├── state_machine.py      # Target State 판정(Zone·Held Object) + 초 단위 디바운스 → 바뀐 것만 Event (순수 로직)
├── event_log.py          # Event를 메모리 목록과 실행별 logs/events-YYYYMMDD-HHMMSS.jsonl에 시간순으로 쌓음
├── reporter.py           # Event Batch·질문 + Snapshot → 프롬프트(순수 함수) → 작업 스레드에서 Gemma 생성 → SITREP + Evidence 출력, 질문 답은 Piper로 읽음
├── announcer.py          # 안내방송: 고정 문구 3개를 시작 때 Piper로 wav 합성, Event·수동 키 → 대기열 → 작업 스레드에서 aplay 재생
├── voice.py              # 녹음 → Whisper(STT), 글 → Piper(TTS) → 재생, 재생 끊기 (외부 프로그램을 subprocess로 부르는 얇은 층)
├── motion.py             # Motion Level: 연속 두 프레임 차이의 평균 하나(0.0~255.0), Jetson은 CuPy
├── hud.py                # HUD 오버레이: 사람 박스·#id·Zone·Held Object, 좌상단 FPS·Motion Level (영어만)
├── platform_config.py    # 트랙(Jetson/Mac)별 값 한 곳: 모델·whisper·Piper 경로, 녹음/재생 명령, 배열 모듈
├── bench.py              # 측정표 생성: 구성 요소별 초, YOLO 루프 FPS(단독/Gemma 동시), Event당 SITREP 초, RAM 피크
├── tests/                # unittest (카메라·모델 없이 실행): test_state_machine, test_target_state, test_prompt, test_motion, test_announcer
├── docs/
│   ├── spec.md                          # 스펙(User Story, 구현 결정, 임시 결정, 미확인 사실)
│   ├── GLOSSARY.md                      # 용어집(Target, Event, Event Log, SITREP, Evidence ...)
│   ├── 0001-llm-reads-event-log-only.md # ADR: LLM은 Event Log만 읽고 위험 판정을 하지 않는다
│   ├── design-draft.md                  # 설계 초안
│   ├── board-check.md, mac-setup.md     # 보드 점검 절차, Mac 개발 환경 설치 기록
│   ├── prototype-state-machine.html     # 상태기계 프로토타입(디바운스 동작 확인용)
│   └── presentation.md / .html / .pdf   # 발표 대본·슬라이드·인쇄본 (개편 전 기준, 한계 참고)
└── README.md
```

모델·오디오·외부 빌드는 코드 폴더가 아니라 **repo 루트**에 둔다(`platform_config.py`의 `REPO_ROOT = CODE_DIRECTORY.parent`). Jetson `~/vision-llm/`의 실제 배치와 같게 하려는 것이다.

```
<repo 루트>/                       # Jetson: ~/vision-llm/
├── src/models/YOLO/               # yolo11n_int8.engine (Jetson 기본 경로), yolo11n.pt (Mac)
├── src/models/Gemma4/             # google_gemma-4-E2B-it-Q4_K_M.gguf
├── src/models/Piper/              # ko_KR-kss-medium.onnx
├── src/audio/                     # input.wav(녹음), response.wav(답 음성), announcements/<id>.wav(안내방송 캐시)
├── whisper.cpp/                   # build-cpu/bin/whisper-cli (Jetson), models/ggml-base.bin
├── .piper_venv/                   # Piper 전용 venv (의존성 충돌 회피)
└── Project03/
```

처리 흐름:

```
[메인 스레드 — 프레임 1장]
카메라 /dev/video0 1280x720 BGR (camera.open_camera, GStreamer)
  → YOLO11n track (detector.detect_frame)            사람 [(track_id, 박스)] + 물체 [(클래스, 박스)]
  → Target State 판정 (state_machine.build_observations)
       Zone: 박스 높이/프레임 높이 ≥ 0.5 → near    Held Object: 물체 박스 중심이 든 사람(가장 가까운 중심)
  → 디바운스 1.0초 (state_machine.step)               바뀐 값이 1초 이어질 때만 인정
  → Event 0~N건 ─┬→ Event Log JSONL + 터미널 한 줄 (main.record_events)
                 ├→ reporter.submit_events  (대기열에 넣고 바로 돌아옴)
                 └→ announcer.handle_events (select_auto_announcement → put_down, 10초 간격 지났으면 대기열 뒤에)
  → Motion Level (motion.compute_motion_level) → HUD (hud.draw_hud) → 창
  → 키: v 녹음 질문 / 1·2·3 수동 방송 / q 종료

[reporter 작업 스레드]
대기열(질문 > Event Batch) → 프롬프트(Event 글 + Snapshot, 한국어 이름표)
  → Gemma 4 E2B 생성 → SITREP: [SITREP] + [Evidence] Event 원문 + [Snapshot] + [걸린 초] — 터미널만
                      → 질문:   [인식] [질문] [답] — 터미널 + Piper 합성 → 재생

[announcer 작업 스레드]
대기열 → 시작 때 합성해 둔 src/audio/announcements/<id>.wav → aplay 재생
  (수동 방송 announce_manual은 재생 중인 소리를 끊고 대기열 맨 앞으로)
```

### 기술 ↔ 코드 위치

| 기술 | 함수·상수 | 파일 |
|---|---|---|
| 객체 검출·추적 | `load_model`, `detect_frame`, `HELD_OBJECT_CLASS_NAMES` | detector.py |
| 관계성(물체↔사람, 거리) | `build_observations`, `judge_zone`, `NEAR_HEIGHT_RATIO` | state_machine.py |
| 상태기계·디바운스 | `step`, `read_targets`, `DEBOUNCE_SECONDS` | state_machine.py |
| Event Log | `EventLog`, `format_event` | event_log.py |
| 프롬프트 생성 | `build_sitrep_prompt`, `build_question_prompt`, `describe_snapshot`, `KOREAN_LABELS` | reporter.py |
| LLM 추론(Gemma 4 E2B) | `load_gemma`, `generate_text`, `Reporter.start`(워밍업) | reporter.py |
| 대기열·순서 | `Reporter.submit_events`, `Reporter.submit_spoken_question`, `Reporter._take_next_job` | reporter.py |
| SITREP + Evidence | `Reporter._write_sitrep` | reporter.py |
| 음성 질문(STT) | `speech_to_text`, `record_audio`, `compute_rms`, `transcribe_wav`, `SILENCE_RMS_THRESHOLD` | voice.py |
| 음성 출력(TTS)·끊기 | `synthesize_speech`, `play_wav`, `stop_playback` | voice.py |
| 안내방송 | `ANNOUNCEMENT_TEXTS`, `select_auto_announcement`, `Announcer.handle_events`(재방송 간격 `REBROADCAST_SECONDS`), `Announcer.announce_manual`, `MANUAL_ANNOUNCEMENT_KEYS`, `submit_manual_announcement` | announcer.py, main.py |
| 방송 wav 캐시 | `Announcer.start`, `build_wav_path`, `ANNOUNCEMENT_WAV_DIRECTORY` | announcer.py, platform_config.py |
| GPU 배열 연산 | `compute_motion_level`, `load_array_module`(Jetson은 CuPy, 실패 시 멈춤) | motion.py, platform_config.py |
| 트랙 분기 | `detect_track`, `build_record_command`, `build_play_command` | platform_config.py |
| 측정 | `run_components`, `run_fps`, `measure_ram` | bench.py |

### 왜 LLM이 판정하지 않나 (ADR 0001)

Gemma는 위험 여부도, 방송 여부도 정하지 않는다. Target State(Zone·Held Object)와 Event, 안내방송 조건은 전부 규칙 코드다. 같은 입력에 같은 결과가 나와야 unittest로 검증할 수 있고(`tests/test_state_machine.py`), LLM은 같은 장면에도 다른 답을 낼 수 있어 검증할 방법이 없다. 그래서 system 프롬프트 규칙 3번에 "위험 여부를 판정하지 마라. 관측된 변화만 설명하라."를 넣었다. LLM의 역할은 Event Log를 사람이 읽기 쉬운 2~3문장으로 바꾸고 질문에 답하는 것뿐이다.

### 왜 이미지 대신 텍스트 Event만 넣나

Jetson Orin Nano 8GB에서 YOLO와 같이 돌려야 하므로 매 프레임 호출이나 이미지 입력은 비용이 맞지 않는다고 봤다(ADR 0001). 이미지 입력(mmproj)의 보드 비용은 미측정이다. 텍스트 Event는 변화가 있을 때만 생기고, 프롬프트가 짧아 생성이 빠르며(측정표의 16.3 tok/s), 문장을 Event 원문과 한 줄씩 대조할 수 있다.

### 왜 Evidence를 같이 출력하나

환각 방지를 두 겹으로 했다. 1차는 프롬프트 규칙("로그에 있는 사실만 써라. 로그에 없는 것은 '관측되지 않음'이라고 답하라."), 2차는 SITREP 바로 아래에 그 근거가 된 Event 원문(영어 JSON 그대로)과 Snapshot을 찍는 것이다. 프롬프트에서 생략된 앞부분 Event도 Evidence에는 다 나온다. Operator는 SITREP 문장이 Evidence에 없는 내용을 말하면 바로 알아챌 수 있다. Evidence는 소리로 읽지 않는다.

### 왜 방송 문구를 미리 합성하나

보드 실측에서 Piper는 호출 한 번에 3.3초(프로세스 시작 약 1.3초 포함)가 걸렸다. 물건을 든 사람에게 "내려놓으세요"가 3초 넘게 늦으면 의미가 없다. 문구가 3개로 고정이라 시작할 때 한 번씩 `src/audio/announcements/<id>.wav`로 합성해 두고, 방송 때는 aplay로 재생만 한다. 또 별도 작업 스레드에서 재생하므로 Gemma 생성(수 초)이 방송을 늦추지 않는다.

## 실행 방식

```bash
# Jetson Orin Nano Super (L4T R36.5.2), repo는 ~/vision-llm/ (위 루트 배치)
# 필요한 파일: src/models/YOLO/yolo11n_int8.engine, src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf,
#             src/models/Piper/ko_KR-kss-medium.onnx, whisper.cpp/build-cpu/bin/whisper-cli(CPU 빌드),
#             whisper.cpp/models/ggml-base.bin, .piper_venv/bin/python(Piper 전용 venv)
# 전력 모드: MAXN_SUPER (1차 점검 기록)
source ~/vision-llm/.venv/bin/activate
cd ~/vision-llm/Project03
python main.py
```

시작하면 Gemma 로드(보드 실측 74초) → 워밍업 생성 1회 → 방송 문구 3개 Piper 합성(`Announcer.start`) → 카메라를 연다. 카메라는 `camera.py`의 Jetson 전용 GStreamer 파이프라인(`JETSON_PIPELINE`, `/dev/video0` MJPG 1280x720 30fps)으로만 연다. 제출본에는 Mac 카메라 분기가 없다.

| 키 | 동작 |
|---|---|
| `v` | 5초 녹음 → Whisper → Gemma 답 → 터미널 + 스피커 |
| `1` / `2` / `3` | 수동 안내방송 `put_down` / `hands_up` / `hold_still` (재생 중인 소리를 끊고 맨 앞) |
| `q` | 종료 |

테스트(카메라·모델·마이크 불필요):

```bash
cd Project03
python -m unittest
```

측정표 생성(`bench.py`, 발표 숫자는 Jetson에서 잰 값만):

```bash
python bench.py components --wav <말이 든 wav>   # 구성 요소별 걸린 초 (카메라·마이크·스피커 안 씀)
python bench.py fps [--seconds 20]               # YOLO 루프 FPS 단독/Gemma 동시 + Event당 SITREP 초 (카메라 사용)
python bench.py ram [--tegrastats-log tegrastats.log]   # tegrastats 로그의 RAM 피크
# RAM: 터미널 1에서 `tegrastats --interval 1000 | tee tegrastats.log`를 켠 채 터미널 2에서 bench.py fps
```

### 측정 결과 (Jetson Orin Nano Super, `Labs/2026-11-07/jetson_precheck.md`)

| 항목 | 값 | 비고 |
|---|---|---|
| 보드·환경 | L4T R36.5.2, torch 2.8.0 CUDA, 전력 모드 MAXN_SUPER | |
| YOLO11n TensorRT fp16 엔진 단독 | 54.2 ms/프레임 (18.4 FPS) | R36.5.2 업그레이드 **전**(R36.4.7)에 측정 |
| Gemma 4 E2B Q4_K_M 로드 | 74초 | |
| Gemma 생성 속도 | 16.3 tok/s, 첫 토큰 0.09초, 38토큰 총 2.42초 | 두 번째 생성부터 |
| Gemma 첫 생성 | 첫 토큰 5.7초, 총 8.5초 | → 시작 때 워밍업 1회 |
| Gemma 속도 (YOLO 동시) | 16.3 tok/s 유지 | |
| RAM 피크 | 5377 / 7607 MB | |
| whisper.cpp base (CPU 빌드) | 5초 wav 1개에 7.3초 | |
| Piper TTS | 호출 1회 3.3초 (프로세스 시작 약 1.3초 포함) | |
| 마이크·스피커 | 동작 확인(사용자) | |
| YOLO FPS (Gemma 동시) | 미측정 | 카메라 문제로 |
| Event → SITREP 걸린 초 | 미측정 | |
| 앱 전체 실행(end-to-end) | 미확인 | 아래 한계 1번 |
| announcer 스레드 포함 RAM·FPS | 미측정 | |
| Motion Level CuPy vs NumPy | 미측정 | |

Mac 개발 사본(학습 repo)에서는 카메라 → Event → SITREP → 음성 흐름만 확인했고, 그 숫자는 결과로 쓰지 않는다.

## 기타 참고 사항

### 확인된 문제점·한계
- **보드에서 앱 전체를 끝까지 돌린 기록이 없다(미확인).** L4T R36.5.2로 올린 뒤 USB 카메라가 열리긴 하지만 1~2프레임 뒤 프레임이 끊긴다(11-07, `gst_sample_get_caps` assertion 후 종료). 원인은 찾지 못했다. 진단 순서(P1, 카메라만 단독)는 `jetson_precheck.md`에 있고 결과는 아직 기록되지 않았다. 대신 프레임이 멈추면 창이 30초 굳지 않도록 읽기 제한 시간을 3초(`JETSON_READ_TIMEOUT_MILLISECONDS`)로 줄였다. YOLO 54.2 ms도 업그레이드 전 값이라 R36.5.2에서 엔진이 그대로 맞는지 미확인이다.
- **엔진 파일 이름.** 앱 기본 경로는 `yolo11n_int8.engine`인데 측정한 54.2 ms는 `yolo11n_fp16.engine`의 값이다. INT8 엔진의 속도는 미측정.
- **STT 품질.** whisper `base` 모델의 한국어 인식은 직접 써 보니 약했다. 예상 어휘를 `--prompt`로 주는 힌트(`WHISPER_PROMPT_HINT`)를 넣었지만 효과는 미측정이다. CPU 빌드라 5초 녹음에 인식 7.3초가 더 걸린다.
- **Piper 호출 비용.** 질문 답은 매번 새로 합성하므로 호출당 3.3초가 답 뒤에 붙는다. 안내방송은 미리 합성해 이 비용을 피했다.
- **시작이 느리다.** Gemma 로드 74초 + 워밍업 1회(첫 생성 비용 8.5초를 시연 첫 SITREP 대신 시작 때 치름).
- **임시값이 많다.** `DEBOUNCE_SECONDS`=1.0, `NEAR_HEIGHT_RATIO`=0.5, Held Object 4종, `MAX_BATCH_EVENTS`=10, `MAX_GENERATED_TOKENS`=120, `SILENCE_RMS_THRESHOLD`=500(Mac 조용한 방 RMS 기준, 말소리 미측정), `REBROADCAST_SECONDS`=10.0, `JETSON_RECORD_DEVICE`=None. 카메라 실측 뒤 바꿔야 한다.
- **스피커 하나를 질문 답과 방송이 같이 쓴다.** 수동 방송은 재생 중인 소리를 끊으므로 질문 답을 읽는 도중에 `1`~`3`을 누르면 답 음성이 끊긴다(글자는 터미널에 남는다). 재생 프로세스 손잡이(`voice._playback_process`)가 한 칸이라, 답과 방송이 동시에 재생되면 늦게 시작한 쪽이 칸을 덮어 먼저 시작한 소리는 끊을 수 없다.
- **재방송 제한이 문구 단위다.** `put_down`을 10초에 한 번만 내므로, 두 번째 사람이 그 10초 안에 물건을 들면 따로 방송하지 않는다(Target별 제한이 아님).
- **물건 판정은 박스 중심 포함 규칙이다.** 사람 박스 안에 물체 박스 중심만 들어가면 "보유"라서, 사람 앞을 지나가는 물체도 1초 넘게 겹치면 보유로 잡힐 수 있다. 실측 비율은 미측정.
- **방송 시작 시 합성 비용.** 문구 3개를 시작 때 합성하므로 시작이 그만큼 늦다(문구당 Piper 1회, 보드에서 합친 초는 미측정).
- **스펙·발표 자료는 개편 전 기준이다.** `docs/spec.md`(User Story 24: `1`·`2` Preset Question, SITREP을 Piper로 읽음)와 `docs/presentation.*`는 11-08 안내방송 개편 전 동작을 설명한다. 현재 동작은 이 README가 기준이다.
- 프로세스 구조는 단일 프로세스 + 작업 스레드(reporter, announcer)다. Jetson에서 이 구조로 YOLO FPS가 떨어지는지는 미측정(별도 프로세스로 바꾸는 것은 보드 실측 후 결정으로 남김).

<!-- 사용자 검토: 본인 말로 고쳐 쓸 것 -->
### AI 활용 vs 본인 기여
- 설계 결정은 AI가 번호 붙은 질문과 추천안을 내고 **본인이 답하는 방식(grilling)으로 정했다**: 주제(정찰 SITREP 보조관), LLM 역할 분리(규칙이 판정, LLM은 설명 — ADR 0001), Event 5필드, 프레임이 아닌 초 단위 디바운스, Jetson/Mac 두 트랙, 음성 흐름(`v` 5초 녹음 → Whisper → Gemma → Piper), 질문의 기준 시각(키를 누른 순간), 무음 검사, 그리고 11-08 스피커 청중을 Operator → 카메라 앞 사람으로 바꾼 안내방송 개편과 그 범위(오늘은 put_down 자동 + 수동 3키, Pose/Hand 자동 판정은 제외).
- **코드는 AI(Claude) 서브 에이전트가 티켓 단위로 작성했다.** 스펙에 없던 구현 세부는 AI가 「임시 결정」으로 정하고 `docs/spec.md`에 기록했다. 작성된 코드는 별도 AI 리뷰어가 검토했고, 메인 세션이 실행해 흐름을 확인했다.
- 보드 측정(YOLO·Gemma·Whisper·Piper·RAM)과 마이크·스피커 동작 확인은 본인이 Jetson에서 직접 했다. 카메라 프레임 끊김도 본인이 보드에서 발견해 기록했다.
- 본인 몫의 핵심은 "무엇을 규칙이 정하고 무엇을 LLM에 맡길지"와 "언제 LLM을 부를지"의 결정이다.

## 창의성 및 확장성

- **역할 분리: 규칙이 판정하고 LLM은 설명한다.** 경량 LLM(E2B)의 약점인 비결정성·환각을 판정 경로에서 빼고, 사람이 읽기 좋은 보고와 자유 질문 답에만 쓴다. 판정 로직은 순수 함수라 테스트된다.
- **Event 기반 기록 → 질문 응답.** 변화만 쌓은 Event Log가 그대로 LLM의 기억이 된다. "지난 1분 동안 무엇이 바뀌었나?" 같은 질문에 코드가 최근 60초(`QUESTION_WINDOW_SECONDS`)로 걸러 넣는다. 실행 뒤에는 JSONL로 다시 볼 수 있다.
- **Evidence를 두 번째 환각 방어선으로.** 프롬프트 규칙만 믿지 않고, 근거 원문을 문장 옆에 같이 보여 사람이 대조하게 했다.
- **코드 한 벌, 두 트랙.** OS로 Jetson/Mac을 자동 판별하고, 갈리는 값은 `platform_config.py` 한 곳에만 둔다. Mac에서 흐름을 만들고 Jetson에서 같은 코드를 돌린다.
- **측정표가 "병목이 어디냐"에 답한다.** Gemma는 YOLO와 동시에도 16.3 tok/s를 유지했고, 느린 쪽은 음성(Whisper CPU 7.3초, Piper 3.3초)이었다. 그래서 방송은 미리 합성하는 쪽으로 설계를 바꿨다.
- **스피커 청중 개편(11-08).** 처음엔 SITREP을 Operator에게 읽어 줬지만, 스피커를 카메라 앞 사람에게 돌려 "현장 안내방송"으로 바꿨다. 보고(터미널)와 현장 통제(스피커)가 서로 막지 않도록 스레드를 나눴다.
- 확장(미구현): Project01의 MediaPipe Pose + Hand로 손 가림·항복 자세를 판정해 `hands_up`·`hold_still`을 자동으로 내기. 추가 모델을 얹었을 때의 RAM·FPS는 미측정이다.
- 확장(미구현): Gemma 4의 이미지 입력(mmproj)으로 Event가 생긴 Target의 박스 crop 1장만 설명하게 하기. 보드에서의 비용은 미측정이다.
