---
title: 최종 프로젝트 — 설계 초안 v0
brand: 온디바이스 AI
nav:
  - 후보 비교 | topic-candidates.html
  - 설계 초안 | design-draft.html
  - 보드 체크 | board-check.html
  - Mac 설치 | mac-setup.html
  - 조사 보고서 | research-report.html
  - 선행 사례 | landscape-sweep.html
---

# 설계 초안 — 정찰 SITREP 보조관 (Project03, 초안 v2, 2026-11-06)

상태: **보드 체크(`board-check.md`) 결과 전 초안.** 수치·프로세스 구조는 실측 후 확정. 코드 없음.

v1 변경(그릴링 3라운드 반영): Zone에서 `gone` 제거, 디바운스를 초 단위로, "2초 창"을 Event Batch 규칙으로, SITREP 표시를 터미널로, Project02 재사용 제거(Project02 취소). 용어는 루트 `GLOSSARY.md`, 핵심 결정은 `docs/adr/0001-llm-reads-event-log-only.md`.

v2 변경(2026-11-06 그릴링 2라운드 반영): 강의 06 노트북 셀 108이 최종 프로젝트에 "유저 음성 입력을 받아 음성으로 출력"을 요구 → 음성 질문·음성 출력을 스트레치에서 필수로. Jetson 트랙(제출용, CUDA 기본, CuPy)과 Mac 트랙(실험용) 두 갈래. INT8·전력 모드 비교는 스트레치로 내림. 지난 세션 미결정 3건 확정(4절). 06 노트북은 다음 커밋에서 삭제됐으니 11-07 수업에서 요구가 바뀌면 그쪽을 따른다.

## 1. 의도
Project01(항복 의사 HUD)의 정찰 자산 상황을 잇는다. 카메라 영상에서 YOLO11n(TensorRT)이 사람·물체를 검출·추적하고, 규칙 코드가 상태(사람 수, 물체 보유, 접근/이탈)를 결정한다. 상태가 바뀔 때만 그 사실을 JSON 이벤트로 기록하고, 로컬 Gemma 4 E2B가 이벤트 로그를 읽어 **한국어 상황 보고(SITREP)** 를 쓰고 운용자 질문에 답한다. LLM은 위험을 판정하지 않는다. 판정은 규칙, LLM은 설명·요약·질의응답.

v2: Operator는 키를 누르고 말로 질문하고(Whisper가 글로 옮김), SITREP과 답변은 Piper가 소리로 읽는다. 흐름은 06 노트북 A절(YOLO → JSON → LLM)과 C절(STT·TTS)을 이은 것이다.

왜 이렇게: 8GB 보드에서 매 프레임 LLM 호출은 불가능하고, 이미지 입력은 비용 미확인(4B 벤치 16초/사이클). 검출→텍스트→텍스트 전용 SLM 조합은 Jetson에서 선례를 찾지 못함(`landscape-sweep.md` B절).

## 2. 목표 / 비목표 / 실패 조건
- 목표: (1) YOLO 루프가 Gemma 생성 중에도 끊기지 않음, (2) 상태 변화 후 수 초 내 SITREP 출력, (3) "지난 1분간 뭐가 바뀌었나" 질문에 로그 근거로 답, (4) 측정표 1개(FP16 × 단독/동시, STT·TTS 걸린 초, 움직임 양 계산의 NumPy/CuPy 시간), (5) 키 `v` → 말로 질문 → 소리로 답변.
- 비목표: 얼굴 인식, 손 상세 분석(Project01 범위), 원격 서버, 웹 UI, 상시 듣기·호출어.
- 실패 조건: Gemma가 로그에 없는 사실을 지어냄(환각)을 막을 장치가 없음 / 동시 실행 시 OOM / 사용자가 설명 못 하는 코드.

## 3. 제약
- 하드웨어: Orin Nano 8GB, R36.5 필수(R36.4.7 OOM 버그). GUI 끄면 약 +800MB.
- 모델: yolo11n.engine(FP16. INT8 비교는 스트레치), google_gemma-4-E2B-it-Q4_K_M.gguf, n_ctx 2048(05 nb 설정).
- 음성: whisper.cpp `base` 모델(Jetson은 CUDA 빌드가 기본), Piper `ko_KR-kss-medium`(별도 `.piper_venv`). 마이크·스피커는 수업 지급품을 쓰고, 그 전에는 미리 녹음한 wav로 STT 경로만 확인.
- GPU: Jetson 트랙은 CUDA가 기본. YOLO(TensorRT)·Gemma·Whisper·CuPy가 8GB를 같이 쓴다. Piper만 CPU로 시작한다(5절 6번).
- 시간: 11-07 오후 설계·착수, 11-08 시연. 사전에는 보드 검증 스파이크와 Mac 트랙 실험만.
- 코드 위치: 11-07 전에는 학습 repo `final_project/code/`, 11-07에 제출 repo `Project03/`로 옮긴다.
- 제출: `Project03/` + README(파일 구조·실행 방식·기타 참고·창의성 및 확장성) + 발표 pdf. 주석 한국어, 네이밍 규칙 CLAUDE.md.

## 4. 구조 (사용자가 결정한 부분)

```
camera.py          Project01 재사용 (GStreamer MJPG 1280x720)
detector.py        YOLO11n .engine 추론 + ultralytics 내장 추적(track id)
state_machine.py   추적 결과 + 시각 → Target별 Target State {held_objects 집합, zone(near/far)} + 초 단위 디바운스 → Event 목록
event_log.py       Event만 기록 (timestamp, target_id, kind, before, after). 메모리 목록 + 실행별 JSONL 파일
reporter.py        Gemma 4 E2B: Event Batch + Snapshot → SITREP, Preset Question·Spoken Question + 최근 60초 Event + Snapshot → 답변. Project02 재사용 없음
hud.py             Project01 overlay 재사용, 영어 오버레이만(박스·Target id·Zone·Held Object·FPS·Motion Level). SITREP·Evidence는 터미널 출력
main.py            루프: 카메라 → 검출 → 상태 → (변화 시) 로그 → reporter 큐에 투입 → HUD
bench.py           측정표 생성 (FPS 단독/동시, 이벤트당 응답 초, STT·TTS 초, Motion Level NumPy/CuPy 시간, tegrastats RAM 피크)
voice.py           speech_to_text(): 녹음 → Whisper → 글 / text_to_speech(): 글 → Piper → 재생
motion.py          Motion Level: 연속 두 프레임의 차이 (Jetson CuPy / Mac NumPy)
platform_config.py 트랙별 모델 경로·녹음/재생 명령·배열 모듈 (아래 「두 트랙」)
```

- Event 종류: `appeared`, `gone`, `zone_changed`, `object_changed`. 출현·이탈은 Target State 전체를 담은 한 건, 그 외는 변화 한 가지당 한 건. 다시 들어온 사람은 새 Target.
- Held Object 판정: 물체 박스 중심이 사람 박스 안이면 보유. 두 박스에 다 들면 중심이 가까운 쪽. 대상은 COCO 3~5종.
- 디바운스: 초 단위 T 하나(출현·이탈·Zone·Held Object 공통). 시각은 상태기계가 재지 않고 인자로 받는다. Held Object는 **물체 종류별로 따로** 디바운스한다(v2). 통째로 하면 칼 검출이 깜빡이는 동안 꺼낸 휴대폰이 늦게 기록된다(프로토타입 탭 6a/6b).
- 출현 확정 순간(v2): `appeared`는 확정 프레임에 보인 Zone·물체만 담는다. 그 프레임에 물체가 안 잡혔으면 T초 뒤 `object_changed`가 따로 난다.
- 같은 추적 id로 다시 들어온 사람(v2): Event Log에 `#1 gone` 다음 `#1 appeared`가 남는다. 상태기계가 번호를 따로 매기지 않고 한계로 설명한다. 추적기가 id를 얼마나 기억하는지는 보드에서 확인.
- 트리거: Event가 생길 때만. 매 프레임 호출 금지. 생성은 한 번에 하나이고, 생성 중 쌓인 Event는 한 묶음(Event Batch)으로 다음 SITREP에 넣는다. Operator 질문이 Event Batch보다 먼저.
- 프롬프트: system에 "로그에 있는 사실만, 없는 건 '관측되지 않음'이라고 답하라" 고정(환각 방지 1차). 입력은 Event Batch(최신 N건, 넘치면 "이전 M건 생략" 표기, 상대 시간) + Snapshot. 출력은 2~3문장 한국어.
- Evidence: SITREP 옆에 근거 Event 원문을 같이 출력(환각 방지 2차, 사람이 대조). 끝에 프롬프트에 넣은 것과 같은 Snapshot 한 줄을 글자로만 덧붙인다(2026-11-06).
- Q&A: Preset Question 2개(`1` 지난 1분 변화, `2` 현재 상황) → 코드가 최근 60초로 거른 Event + Snapshot → Gemma. Event가 없어도 그대로 넘긴다. 질문의 시점은 키를 누른 순간이다(2026-11-06): Snapshot은 그 프레임 것, Event는 그 시각까지 것만 넣고, Spoken Question은 `v`를 누른 시각 기준이라 답이 몇 초 전의 상황을 말할 수 있다. Summary Memory는 스트레치.
- Spoken Question(v2): 키 `v` → 고정 5초 녹음 → Whisper → 그 글을 질문으로 Gemma에 넘긴다(Preset Question과 같은 입력: 최근 60초 Event + Snapshot). 로그에 없는 것을 물으면 "관측되지 않음". Preset Question `1`/`2`는 마이크가 안 될 때를 위해 남긴다.
- 음성 출력(v2): SITREP과 질문 답변을 둘 다 Piper로 읽는다. Evidence는 소리로 내지 않고 터미널 글자로만.
- 한국어 낱말(v2): Event의 `near`, `knife`, `#1` 같은 영어·기호를 한국어 Piper 음성이 제대로 읽는지는 미확인. 코드에 한국어 이름표(knife → 칼, near → 가까운 구역)를 두어 프롬프트에 넣고, system에 "한국어 낱말로만 쓰라"를 더한다. Evidence는 영어 원문 그대로.
- 못 알아들었을 때(v2): Whisper 결과가 빈 글이면 Gemma를 부르지 않고 "질문을 듣지 못했습니다"를 출력·재생한다.
- 말하는 중에 `v`(v2): 재생 중인 음성을 끊고 바로 녹음한다. 그때 생성 중이던 SITREP은 끝까지 쓰되 글자로만 내고, 그다음 질문을 처리한다. 끊지 않으면 스피커 소리가 마이크에 섞인다.
- Motion Level(v2): 연속 두 프레임의 차이로 숫자 하나를 구해 HUD에 표시한다. Jetson은 CuPy, Mac은 NumPy. 두 방식의 시간을 재서 측정표에 넣고, CuPy가 느리게 나와도 그대로 적는다. Zone·Held Object 판정은 박스 몇 개짜리 계산이라 GPU로 올리지 않는다(복사 비용이 더 큼).
- 자동 테스트 경계: (a) 상태기계: 검출 결과 + 시각 → Event 목록, (b) 프롬프트 생성: Event Batch + Snapshot → 프롬프트 문자열, (c) Motion Level 계산(NumPy 경로, v2). 검출기·Gemma 출력·HUD는 보드에서 눈으로, 음성은 귀로 확인.

### 두 트랙 (v2)

코드는 한 벌이다. 플랫폼마다 갈리는 부분만 파일 단위로 바꿔 끼운다. 상태기계·Event Log·프롬프트 생성은 공통.

| 갈리는 부분 | Jetson 트랙 (제출용) | Mac 트랙 (실험용) |
|---|---|---|
| 검출 모델 | `yolo11n.engine` (TensorRT) | `yolo11n.pt` |
| 배열 모듈 | CuPy | NumPy |
| 녹음·재생 | `arecord` · `aplay` | `ffmpeg` · `afplay` |
| Whisper | CUDA 빌드 `whisper-cli` | Mac 빌드 |
| Gemma | llama-cpp-python CUDA | llama-cpp-python (Mac) |

- Mac 트랙은 TensorRT와 CuPy만 빼고 전체 흐름을 돌린다. FPS·초·RAM 숫자는 Jetson 것만 발표에 쓴다.
- 제출 `Project03/`에는 Mac용 파일도 넣고 README에 "실험용"이라고 적는다.
- 트랙은 실행할 때 OS로 자동 판별한다(Linux → Jetson, macOS → Mac). Jetson에서 CuPy가 안 불러와지면 NumPy로 넘어가지 않고 에러로 멈춘다. 조용히 넘어가면 제출용이 CPU로 도는 것을 모른 채 시연하게 된다.
- Mac 트랙은 전용 venv `final_project/code/.venv`를 쓴다. 기존 `.venv`(numpy 1.21.5·cv2 4.7 고정)는 건드리지 않는다. 설치 순서는 `mac-setup.md`.

### 구현 세부 임시 결정 (v2, 2026-11-06)

구현을 AI에 맡기면서 원본에 없던 세부 10가지(Motion Level 계산식, 한국어 이름표, Snapshot 글 모양, 상대 시간 표기, 답변 길이, 처리 중 키 입력, 동시 실행 방식, JSONL 이름, `timestamp` 기준, Piper 실패 시 동작)는 AI가 임시로 정했다. 내용은 `.scratch/project03/spec.md`의 「임시 결정」 절. 마음에 안 들면 바꾼다.

## 5. 구현자(=사용자)가 보드 실측 후 정할 것
1. **프로세스 구조**: (a) 단일 프로세스 + reporter 스레드/큐, (b) reporter 별도 프로세스(NVIDIA 권장, CUDA 컨텍스트 분리). llama-cpp-python이 생성 중 GIL을 놓는지 미확인 → (a)에서 YOLO FPS가 떨어지면 (b). 체크리스트 5번 결과로 결정.
2. `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` 필요 여부(error 12 재현 시).
3. (v2에서 스트레치로 내림) INT8 엔진.
4. 수치: 디바운스 T(초), near/far 임계값(박스 높이 비율), Event 상한 N, Held Object 클래스 목록 — Project01 방식대로 카메라 캡처 실측으로.
5. 스트레치 채택: mmproj 크롭 1장 비교 / 툴 호출 1개 / INT8 엔진 비교 / 전력 모드(15W·MAXN) 비교 / 움직임 양으로 YOLO 건너뛰기(추적 id가 끊길 위험).
6. Piper는 CPU로 시작한다(v2 확정). `board-check.md` 8-3의 합성 초를 보고 느리면 GPU(`--cuda` + Jetson용 `onnxruntime-gpu`)를 시도한다. "느리다"의 기준 숫자는 실측 뒤에 정한다. 이유: GPU 휠 설치 위험, 문장마다 프로세스를 새로 띄워 GPU 준비 시간이 붙음, 8GB 경쟁.
7. 메모리가 모자랄 때 내리는 순서(v2 확정): ① Whisper를 CPU 빌드로 → ② Gemma `n_ctx` 1024 → ③ Gemma GPU 레이어 수 줄이기. YOLO는 실시간 루프라 끝까지 GPU에 둔다.

## 6. 성공 판정(코드 전 작성할 검증 항목)
- [ ] 상태기계 단위 테스트: 같은 입력 시퀀스 → 같은 이벤트 열(카메라 없이).
- [ ] 이벤트 로그가 비어 있을 때 Gemma 답변이 로그에 없는 사실을 지어내지 않는다("기록된 변화 없음"·"관측되지 않음" 같은 답이면 통과).
  - 2026-11-06 문구 변경(사용자 결정, 프롬프트는 그대로): Mac 트랙 소수 실행에서 로그가 빈 채 Preset Question을 물으면 "기록된 변화는 없습니다" 같은 답이 나왔고 "관측되지 않음" 글자 그대로는 4번 중 1번, 로그에 없는 것을 직접 물었을 때는 2번 중 2번이었다. 표본이 작아 일반화하지 않는다.
- [ ] 동시 실행 중 YOLO FPS가 단독 대비 X% 이내(X는 실측 후 설정).
- [ ] 이벤트 발생 → SITREP 표시까지 초 측정값 기록.
- [ ] tegrastats RAM 피크 < 7.0GB, OOM 0회(시연 10분).
- [ ] 키 `v` → 말로 질문 → 소리로 답변이 끝까지 한 번 돈다. 녹음·STT·TTS 동안 YOLO 루프가 멈추지 않는다.
- [ ] SITREP이 소리로 나오고, 재생 중 `v`를 누르면 소리가 끊기고 녹음이 시작된다.
- [ ] 마이크를 뽑아도 Preset Question `1`/`2`로 답을 받는다.
- [ ] 같은 프레임 쌍에서 NumPy와 CuPy의 움직임 양이 같은 값이다(오차 범위는 실측 후 설정).

## 7. 발표에서 예상 질문
1. 왜 Gemma에 이미지를 안 보여 주나 → 비용(벤치)·메모리·환각, 텍스트 로그가 근거 추적 가능.
2. 왜 LLM이 위험 판정을 안 하나 → 결정론·검증 가능·안전. LLM은 설명 층.
3. 동시 실행에서 무엇이 병목인가 → 측정표로 답(실측값).
4. Project01과 무엇이 다른가 → 검출기 교체(TRT), 판정 단순화, 보고 층 추가, 측정.
5. 왜 Zone·Held Object 계산에는 CuPy를 안 썼나 → 값 몇 개짜리 계산은 GPU로 올리는 복사 비용이 더 큼. 프레임 전체 연산에만 쓰고 시간을 재서 비교.
6. 왜 상시 듣기가 아니라 키 + 5초 녹음인가 → 상시 듣기는 CPU를 계속 쓰고 스피커 소리·잡음에 오작동. 노트북 C절 방식 그대로.
7. 왜 Mac 버전이 따로 있나 → 보드 없이 흐름·로직을 먼저 확인하려고. 갈리는 부분만 파일로 분리, 숫자는 Jetson 것만 사용.
