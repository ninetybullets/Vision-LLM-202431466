이 문서는 구현 중 쓴 작업용 스펙이다.
`.scratch/…` 경로와 티켓 번호는 학습 repo 내부 참조라 이 폴더에는 없다.

# Recon SITREP Assistant (Project03) — 스펙

Status: ready-for-agent
작성: 2026-11-06
근거: `final_project/design-draft.md` v2, `GLOSSARY.md`, `docs/adr/0001-llm-reads-event-log-only.md`, `final_project/board-check.md`, `final_project/mac-setup.md`, `final_project/prototype-state-machine.html`, `handoff/2026-11-05-1023-project03-grilling-prototype.md`

## Problem Statement

정찰 자산의 카메라 앞에서 Operator가 화면을 계속 지켜보지 않아도, 그동안 무엇이 바뀌었는지 알고 싶다. 카메라 영상에는 검출 박스가 계속 쏟아지지만, Operator에게 필요한 것은 "Target 한 명이 가까운 구역으로 들어와 칼을 꺼냈다" 같은 변화와 그 근거다. Operator는 손이 바쁠 때 화면을 읽는 대신 말로 묻고 소리로 답을 듣고 싶다.

제약 때문에 쉬운 방법은 막혀 있다. Jetson Orin Nano 8GB 한 대에서 YOLO와 LLM을 같이 돌려야 하므로 매 프레임 LLM을 부를 수도, LLM에 이미지를 보여 줄 수도 없다. LLM이 로그에 없는 사실을 지어내면(환각) 보고를 믿을 수 없다. 위험 여부를 LLM이 정하면 같은 장면에 다른 답이 나와 검증할 수 없다.

수강생(구현자) 쪽 문제도 있다. 강의 06 노트북 셀 108은 최종 프로젝트에 "객체 탐지 + 이벤트 감지 + 유저 음성 입력 → 음성 출력"을 요구한다. 11-07 오후 착수, 11-08 시연이라 시간이 짧고, 채점에서 본인이 설명하지 못하는 코드는 이해도 가중치로 감점된다. 보드 없이도 흐름을 먼저 확인할 방법이 필요하다.

## Solution

카메라 영상에서 YOLO11n(TensorRT)이 COCO `person` 클래스와 미리 정한 물체를 검출·추적한다. 규칙 코드(상태기계)가 추적 결과와 시각을 받아 Target별 Target State(Held Object 집합, Zone)를 정하고, 초 단위 디바운스를 거쳐 바뀐 것만 Event로 Event Log에 남긴다. 변화가 없는 프레임은 남기지 않는다.

Event가 생기면 로컬 Gemma 4 E2B가 Event Batch와 Snapshot만 텍스트로 읽어 2~3문장 한국어 SITREP을 쓴다. SITREP 옆에는 근거가 된 Event 원문(Evidence)을 터미널에 함께 출력해 Operator가 한 줄씩 대조할 수 있다. LLM은 위험을 판정하지 않는다(ADR 0001). 판정은 규칙, LLM은 설명·요약·질의응답이다.

Operator는 키 `v`를 눌러 5초 동안 말로 묻는다(Spoken Question). Whisper가 글로 옮기고, Gemma가 최근 60초 Event와 Snapshot을 근거로 답한다. 마이크가 안 될 때를 위해 Preset Question `1`(지난 1분 변화)·`2`(현재 상황)도 남는다. SITREP과 답변은 Piper가 한국어 음성으로 읽는다. 화면 HUD에는 영어 오버레이(박스, Target id, Zone, Held Object, FPS, Motion Level)만 그린다.

코드는 한 벌이고 두 트랙으로 돈다. Jetson 트랙(제출용, CUDA 기본, CuPy)과 Mac 트랙(실험용, TensorRT·CuPy 제외 전체 흐름)이다. 실행 시 OS로 트랙을 자동 판별한다. 마지막으로 측정표 하나(FP16 × 단독/동시 FPS, STT·TTS 걸린 초, Motion Level의 NumPy/CuPy 시간 등)를 만들어 발표 근거로 쓴다.

## User Stories

### Operator — 관측과 SITREP

1. As an Operator, I want 화면에 들어온 Target이 잠깐 머문 뒤에야 기록되기를, so that 화면 끝을 스치거나 오검출된 것에 SITREP이 나오지 않는다.
2. As an Operator, I want Target이 화면을 떠나면 `gone` Event 한 건에 떠나기 직전의 Target State가 담기기를, so that 누가 무엇을 든 채 어느 구역에서 사라졌는지 알 수 있다.
3. As an Operator, I want 이미 Target인 대상이 잠깐 가려졌다 돌아올 때는 Event가 나지 않기를, so that 가림 때문에 `gone`·`appeared`가 반복되는 소음을 듣지 않는다.
4. As an Operator, I want Target이 `far`에서 `near`로 다가오면 `zone_changed` Event가 나기를, so that 접근을 놓치지 않는다.
5. As an Operator, I want near/far 경계에서 Zone이 떨릴 때는 Event가 나지 않기를, so that 같은 접근이 여러 번 보고되지 않는다.
6. As an Operator, I want Target이 물체를 꺼내거나 내려놓으면 `object_changed` Event가 나기를, so that Held Object 변화를 알 수 있다.
7. As an Operator, I want 한 물체의 검출이 깜빡여도 다른 물체의 변화가 제때 기록되기를, so that 칼 검출이 깜빡이는 동안 꺼낸 휴대폰이 늦게 보고되지 않는다.
8. As an Operator, I want 여러 물체를 한꺼번에 내려놓으면 `object_changed` 한 건으로 기록되기를, so that 같은 순간의 변화가 쪼개져 보고되지 않는다.
9. As an Operator, I want 다가오면서 동시에 물체를 꺼내면 같은 시각의 `zone_changed`와 `object_changed`가 각각 남기를, so that 서로 다른 종류의 변화를 구분해 볼 수 있다.
10. As an Operator, I want Target이 물체를 든 채 들어오면 `appeared` 한 건에 Zone과 Held Object가 함께 담기기를, so that 출현 상황을 한 줄로 파악한다.
11. As an Operator, I want Event가 생길 때만 SITREP이 나오기를, so that 변화가 없는데 같은 말을 반복해 듣지 않는다.
12. As an Operator, I want Event가 생긴 뒤 수 초 안에 SITREP을 받기를, so that 상황에 늦지 않게 대응한다.
13. As an Operator, I want SITREP이 2~3문장 한국어이기를, so that 짧게 읽고 들을 수 있다.
14. As an Operator, I want SITREP 생성 중에 쌓인 Event가 다음 SITREP 한 번에 묶여 나오기를, so that 생성이 밀려 SITREP이 줄줄이 쌓이지 않는다.
15. As an Operator, I want Event Batch가 너무 길면 최신 N건만 쓰고 "이전 M건 생략"이 표시되기를, so that 오래된 Event가 생략됐다는 사실을 안다.
16. As an Operator, I want SITREP 옆에 근거 Event 원문(Evidence)이 터미널에 함께 나오기를, so that SITREP 문장을 근거와 한 줄씩 대조해 환각을 직접 확인한다.
17. As an Operator, I want SITREP이 로그에 없는 사실을 말하지 않고 없는 것은 "관측되지 않음"이라고 하기를, so that 보고를 믿을 수 있다.
18. As an Operator, I want SITREP이 위험 여부를 판정하지 않기를, so that 판정은 규칙 코드의 결정론적 결과로만 받는다.
19. As an Operator, I want SITREP과 Evidence가 깨지지 않는 한글로 터미널에 보이기를, so that 모니터에서 바로 읽는다.

### Operator — 질문과 음성

20. As an Operator, I want 키 `v`를 누르고 5초 동안 말로 질문하기를, so that 손을 쓰지 않고 상황을 묻는다.
21. As an Operator, I want 내 Spoken Question에 최근 60초 Event와 Snapshot을 근거로 한 답을 받기를, so that 답이 실제 기록에 근거한다.
22. As an Operator, I want 로그에 없는 것을 물으면 "관측되지 않음"이라는 답을 받기를, so that 지어낸 답에 속지 않는다.
23. As an Operator, I want 말이 인식되지 않았으면 "질문을 듣지 못했습니다"를 듣기를, so that 다시 물어야 한다는 것을 안다.
24. As an Operator, I want `1`을 누르면 지난 1분 변화를, `2`를 누르면 현재 상황을 답으로 받기를, so that 마이크가 고장 나도 질문할 수 있다.
25. As an Operator, I want 최근 60초에 Event가 없어도 Preset Question·Spoken Question에 답을 받기를, so that "변화 없음"도 확인할 수 있다.
26. As an Operator, I want SITREP과 질문 답변을 소리로 듣기를, so that 화면을 보지 않아도 상황을 안다.
27. As an Operator, I want Evidence는 소리로 읽히지 않고 터미널 글자로만 나오기를, so that 음성이 길어지지 않는다.
28. As an Operator, I want 음성에 `near`, `knife`, `#1` 같은 영어·기호 대신 한국어 낱말이 쓰이기를, so that 한국어 음성이 알아들을 수 있게 읽는다.
29. As an Operator, I want 소리가 나오는 중에 `v`를 누르면 재생이 끊기고 바로 녹음이 시작되기를, so that 스피커 소리가 내 질문에 섞이지 않는다.
30. As an Operator, I want 내 질문이 대기 중인 Event Batch보다 먼저 처리되기를, so that 물어본 것에 먼저 답을 받는다.
31. As an Operator, I want 녹음·인식·합성·생성이 도는 동안에도 HUD 영상이 멈추지 않기를, so that 그사이 생긴 변화도 놓치지 않는다.
32. As an Operator, I want 상시 듣기가 아니라 키를 누를 때만 녹음되기를, so that 주변 소리나 스피커 소리로 오작동하지 않는다.

### Operator — HUD

33. As an Operator, I want HUD에 Target 박스, Target id, Zone, Held Object, FPS가 영어로 겹쳐 보이기를, so that 규칙이 지금 무엇을 판정하고 있는지 눈으로 확인한다.
34. As an Operator, I want HUD에 Motion Level 숫자가 보이기를, so that 화면 전체가 얼마나 움직이는지 한눈에 본다.

### 구현자(수강생 본인) — 설계·구조

35. As a 구현자, I want 상태기계가 시계를 직접 읽지 않고 시각을 인자로 받기를, so that 카메라 없이 같은 입력 시퀀스로 같은 Event 열을 재현해 테스트한다.
36. As a 구현자, I want 디바운스가 프레임 수가 아니라 초 단위 T 하나이기를, so that Gemma 생성 중 FPS가 떨어져도 확정까지 걸리는 시간이 같다.
37. As a 구현자, I want Held Object 디바운스가 물체 종류별로 따로 돌기를, so that 한 물체의 깜빡임이 다른 물체의 기록을 막지 않는다.
38. As a 구현자, I want Event 모양이 `timestamp`, `target_id`, `kind`, `before`, `after` 다섯 필드로 고정되기를, so that Event Log, 프롬프트 생성, Evidence 출력이 같은 모양을 공유한다.
39. As a 구현자, I want Event Log가 메모리 목록과 실행별 JSONL 파일 두 곳에 남기를, so that 실행 중에는 최근 Event를 빨리 거르고 실행 뒤에는 기록을 다시 볼 수 있다.
40. As a 구현자, I want 프롬프트 생성이 Gemma 호출과 분리된 순수 함수이기를, so that 모델 없이 프롬프트 문자열을 테스트한다.
41. As a 구현자, I want 코드에 한국어 이름표(knife → 칼, near → 가까운 구역 등)를 두어 프롬프트에 넣기를, so that Piper가 읽을 SITREP이 한국어 낱말로 나온다.
42. As a 구현자, I want 코드는 한 벌이고 플랫폼마다 갈리는 부분만 파일 단위로 바꿔 끼우기를, so that 상태기계·Event Log·프롬프트 생성을 두 트랙에서 공통으로 쓴다.
43. As a 구현자, I want 실행 시 OS로 트랙이 자동 판별되기를(Linux → Jetson, macOS → Mac), so that 실행 옵션을 잘못 줘서 엉뚱한 트랙으로 돌지 않는다.
44. As a 구현자, I want Jetson에서 CuPy를 불러오지 못하면 NumPy로 넘어가지 않고 에러로 멈추기를, so that 제출용이 CPU로 도는 것을 모른 채 시연하지 않는다.
45. As a 구현자, I want 보드 없이 Mac 트랙에서 전체 흐름(검출 → 상태 → Event → SITREP → 음성)을 먼저 돌려 보기를, so that 11-07 전에 로직을 확인한다.
46. As a 구현자, I want Mac 트랙이 기존 venv를 건드리지 않고 전용 venv를 쓰기를, so that 다른 실습 환경(numpy·cv2 고정 버전)이 깨지지 않는다.
47. As a 구현자, I want Piper가 별도 venv에서 돌기를, so that Piper 의존성이 메인 환경과 충돌하지 않는다.
48. As a 구현자, I want Project01의 카메라 모듈과 HUD 오버레이를 재사용하기를, so that 검증된 GStreamer MJPG 1280x720 입력과 그리기 코드를 다시 쓰지 않는다.
49. As a 구현자, I want Zone·Held Object 판정은 CPU에서 하고 Motion Level처럼 프레임 전체 연산만 CuPy로 하기를, so that 박스 몇 개짜리 계산에 GPU 복사 비용을 치르지 않는다.
50. As a 구현자, I want Motion Level을 NumPy와 CuPy 두 경로로 계산하고 시간을 재기를, so that CuPy가 느리게 나와도 그 숫자를 그대로 측정표에 적는다.
51. As a 구현자, I want 메모리가 모자랄 때 내리는 순서가 정해져 있기를(① Whisper CPU 빌드 → ② Gemma `n_ctx` 1024 → ③ Gemma GPU 레이어 수 축소, YOLO는 끝까지 GPU), so that 보드에서 OOM이 나도 무엇부터 바꿀지 고민하지 않는다.
52. As a 구현자, I want Piper를 CPU로 시작하고 실측 합성 초를 보고 GPU를 시도할지 정하기를, so that GPU 휠 설치 위험과 8GB 경쟁을 처음부터 떠안지 않는다.
53. As a 구현자, I want 프로세스 구조(단일 프로세스 + reporter 스레드 vs reporter 별도 프로세스)를 보드 실측 뒤 정하기를, so that llama-cpp-python의 GIL 동작을 확인한 뒤 YOLO FPS를 지키는 쪽을 고른다.
54. As a 구현자, I want 디바운스 T, near/far 임계값, Event 상한 N, Held Object 클래스 목록을 카메라 캡처 실측으로 정하기를, so that Project01처럼 근거 있는 숫자를 쓴다.
55. As a 구현자, I want 측정표 생성 도구(bench)가 FPS 단독/동시, Event당 응답 초, STT·TTS 초, Motion Level NumPy/CuPy 시간, RAM 피크를 모으기를, so that 발표 질문 "무엇이 병목인가"에 숫자로 답한다.
56. As a 구현자, I want 같은 추적 id로 다시 들어온 Target이 `#1 gone` 다음 `#1 appeared`로 남는 것을 한계로 설명하기를, so that 상태기계에 번호 재부여 로직을 더하지 않는다.
57. As a 구현자, I want `appeared`가 확정 프레임에 보인 Zone·물체만 담고, 그때 안 잡힌 물체는 T초 뒤 `object_changed`로 나기를, so that 출현 규칙이 단순하고 테스트로 설명된다.
58. As a 구현자, I want 상태기계·프롬프트 생성·Motion Level(NumPy 경로)에 자동 테스트가 있기를, so that 판정 로직을 바꿀 때 회귀를 바로 잡는다.
59. As a 구현자, I want 마이크·스피커 지급 전에는 미리 녹음한 wav로 STT 경로만 확인하기를, so that 장비가 없어도 Whisper 연결을 검증한다.
60. As a 구현자, I want 모든 코드를 내가 설계하고 설명할 수 있는 단위로 만들기를, so that 이해도 가중치 w로 감점되지 않는다.

### 채점자(강사)

61. As a 채점자, I want 강의 06 노트북 셀 108 요구(객체 탐지 + 이벤트 감지 + 유저 음성 입력 → 음성 출력)가 시연에서 끝까지 한 번 도는 것을 보기를, so that 필수 요구 충족을 확인한다.
62. As a 채점자, I want 제출 폴더 `Project03/`에 실행 파일·부가 파일·README·발표 pdf가 있기를, so that 제출 형식을 채점한다.
63. As a 채점자, I want README에 「파일 구조」「실행 방식」「기타 참고 사항(문제점·AI 활용 vs 본인 기여도)」「창의성 및 확장성」 절이 있기를, so that 구조와 기여도를 판단한다.
64. As a 채점자, I want Mac용 파일이 README에 "실험용"으로 표시되기를, so that 어느 쪽이 제출 대상인지 헷갈리지 않는다.
65. As a 채점자, I want 발표 숫자(FPS·초·RAM)가 Jetson 실측값뿐이기를, so that 온디바이스 성능 주장을 신뢰한다.
66. As a 채점자, I want 함수 분리, `main()`, `if __name__ == "__main__"` 가드가 있는 구조를 보기를, so that 코드 구조/가독성을 채점한다.
67. As a 채점자, I want 한국어 주석·docstring과 줄임말 없는 이름을 보기를, so that 가독성을 채점한다.
68. As a 채점자, I want 학생에게 왜 LLM에 이미지를 안 보여 주는지, 왜 LLM이 위험 판정을 안 하는지, 무엇이 병목인지, Project01과 무엇이 다른지, 왜 Zone 계산에 CuPy를 안 썼는지, 왜 키 + 5초 녹음인지, 왜 Mac 버전이 따로 있는지 물었을 때 근거 있는 답을 듣기를, so that 이해도 w를 매긴다.
69. As a 채점자, I want 자동 테스트가 카메라·모델 없이 돌기를, so that 판정 로직의 결정론을 직접 확인한다.
70. As a 채점자, I want 커밋 메시지가 변경 내용을 구체적으로 말하기를, so that 제출 형식 감점 여부를 판단한다.

## Implementation Decisions

### 모듈

사용자가 정한 모듈과 역할이다(design-draft v2 4절).

- **camera**: Project01 재사용. GStreamer MJPG 1280x720.
- **detector**: YOLO11n 추론 + ultralytics 내장 추적(추적 id). Jetson은 TensorRT 엔진(FP16), Mac은 PyTorch 가중치.
- **state_machine**: 추적 결과 + 시각 → Target별 Target State {held_objects 집합, zone(near/far)} + 초 단위 디바운스 → Event 목록. 시계를 직접 읽지 않는다.
- **event_log**: Event만 기록. 메모리 목록 + 실행별 JSONL 파일.
- **reporter**: Gemma 4 E2B. Event Batch + Snapshot → SITREP, Preset Question·Spoken Question + 최근 60초 Event + Snapshot → 답변. Project02 재사용 없음(Project02 취소).
- **hud**: Project01 오버레이 재사용. 영어 오버레이만(박스·Target id·Zone·Held Object·FPS, 그리고 Motion Level). Hershey 글꼴이라 한글을 못 그리므로 SITREP·Evidence는 터미널 출력.
- **main**: 루프. 카메라 → 검출 → 상태 → (Event가 생기면) 로그 → reporter로 넘김 → HUD.
- **bench**: 측정표 생성.
- **voice**: `speech_to_text()` 녹음 → Whisper → 글, `text_to_speech()` 글 → Piper → 재생.
- **motion**: Motion Level. 연속 두 프레임의 차이로 숫자 하나. Jetson CuPy / Mac NumPy.
- **platform_config**: 트랙별 모델 경로, 녹음·재생 프로그램, 배열 모듈.

### Event

- 종류는 `appeared`, `gone`, `zone_changed`, `object_changed` 넷뿐.
- 필드: `timestamp`, `target_id`, `kind`, `before`, `after`.
- `appeared`: `before`는 없음(null), `after`는 확정 순간의 Target State 전체(`zone`, `held_objects`).
- `gone`: `before`는 마지막 Target State 전체, `after`는 없음(null). 그 뒤 Target을 지운다.
- `zone_changed`: `before`·`after`는 Zone 값.
- `object_changed`: `before`·`after`는 Held Object 목록(정렬). 같은 프레임에 여러 물체가 바뀌어도 한 건.
- 같은 프레임에 Zone과 Held Object가 같이 바뀌면 두 건.
- 다시 들어온 대상은 새 Target. 같은 추적 id로 다시 들어오면 Event Log에 `#1 gone` 다음 `#1 appeared`가 남고, 상태기계는 번호를 따로 매기지 않는다(한계로 설명). 추적기가 id를 얼마나 기억하는지는 보드에서 확인.

### Target State 판정

- Held Object: 물체 박스 중심이 Target 박스 안이면 보유. 두 Target 박스에 다 들면 중심이 가까운 쪽. 대상은 COCO 3~5종(목록은 미정).
- Zone: `near`/`far` 둘. 박스 높이 비율 임계값으로 나눈다(임계값 미정). 화면에 없는 것은 Zone이 아니다.
- Zone·Held Object 계산은 CPU. GPU로 올리지 않는다(박스 몇 개짜리 계산은 복사 비용이 더 큼).

### 디바운스 규칙

- 초 단위 T 하나를 출현·이탈·Zone·Held Object에 공통으로 쓴다(T 값은 미정).
- 출현: 보이는 상태가 T 동안 이어지면 확정. 그 전에 사라지면 Event 없음(Target이 된 적 없으므로 `gone`도 없음).
- 이탈: 이미 Target인 대상이 T 동안 안 보이면 확정. T보다 짧은 끊김은 Event 없음. 안 보이는 동안 Zone·Held Object는 건드리지 않는다.
- Zone: 새 값이 T 동안 이어져야 바뀐다. 경계에서 떨리면 Event가 없고, 실제 접근도 떨림이 멈춘 뒤 T가 지나야 기록된다(프로토타입 탭 5에서 확인한 지연).
- Held Object: **물체 종류별로 따로** 디바운스한다. 종류마다 "들고 있다/아니다"를 독립 값으로 거르고, 하나라도 바뀌면 그 프레임에 `object_changed` 한 건.
- 출현 확정 순간: `appeared`는 확정 프레임에 보인 Zone·물체만 담는다. 그 프레임에 물체가 안 잡혔으면 T초 뒤 `object_changed`가 따로 난다.

아래는 프로토타입 `<script id="logic">`에서 가져온 핵심 부분이다. 비교용으로 함께 있던 통째(whole-set) 변형은 버리고 종류별(per-class) 경로만 남겼다. 프로토타입은 시각을 ms로 다루지만 규칙은 초 단위 T와 같다. Python으로 다시 쓸 때 참고하는 조각이며, 그대로 옮기는 코드가 아니다.

```js
// (프로토타입에서 발췌, per-class 경로만)
// 값 하나의 안정 상태. stable은 인정된 값, candidate는 바뀌려고 대기 중인 값.
function createDebouncer(stable) {
  return { stable, candidate: null, candidateSince: null };
}

// 이번 프레임의 값(raw)이 인정된 값과 다른 채로 holdMs 동안 계속되면 인정된 값을 바꾼다.
// 반환: [새 디바운서, 이번에 바뀌었는지]
function debounce(debouncer, raw, now, holdMs) {
  if (raw === debouncer.stable) return [createDebouncer(debouncer.stable), false];
  const since = raw === debouncer.candidate ? debouncer.candidateSince : now;
  if (now - since >= holdMs) return [createDebouncer(raw), true];
  return [{ stable: debouncer.stable, candidate: raw, candidateSince: since }, false];
}

// Held Object: 물체 종류마다 "들고 있다/아니다"를 따로 거른다.
function createHeld(rawClasses) {
  const perClass = {};
  for (const name of rawClasses) perClass[name] = createDebouncer(true);
  return perClass;
}

function updateHeld(held, rawClasses, now, holdMs) {
  const next = {};
  let hasChanged = false;
  for (const name of new Set([...Object.keys(held), ...rawClasses])) {
    const previous = held[name] ?? createDebouncer(false);
    const [debouncer, changed] = debounce(previous, rawClasses.includes(name), now, holdMs);
    next[name] = debouncer;
    hasChanged = hasChanged || changed;
  }
  return [next, hasChanged];
}

function readHeld(held) {
  return Object.keys(held).filter((name) => held[name].stable).sort();
}

// detections: [{ trackId, zone: "near"|"far", heldObjects: [종류, ...] }]  이번 프레임에 보인 대상
// now: 이 프레임의 시각. 상태기계는 시계를 직접 읽지 않는다.
// 반환: { state, events }
function step(state, detections, now, config) {
  const { holdMs } = config;
  // ... trackIds = 이전 상태의 id ∪ 이번 프레임의 id (오름차순)
  const addEvent = (targetId, kind, before, after) =>
    events.push({ timestamp: now, target_id: targetId, kind, before, after });

  for (const trackId of trackIds) {
    const track = state.tracks[trackId] ?? { presence: createDebouncer(false), zone: null, held: null };
    const detection = detectionById[trackId];
    const isSeen = detection !== undefined;
    const wasTarget = track.presence.stable;
    const [presence, hasPresenceChanged] = debounce(track.presence, isSeen, now, holdMs);

    if (!wasTarget) {
      if (hasPresenceChanged) {
        // 출현 확정: 이 순간 보이는 Zone과 물체를 그대로 첫 Target State로 삼는다.
        const confirmed = { presence, zone: createDebouncer(detection.zone), held: createHeld(detection.heldObjects) };
        tracks[trackId] = confirmed;
        addEvent(trackId, "appeared", null, readTargetState(confirmed));
      } else if (presence.candidate !== null) {
        tracks[trackId] = { presence, zone: null, held: null };
      }
      continue;
    }

    if (hasPresenceChanged) {
      // 이탈 확정: 마지막 Target State를 담은 한 건만 내고 Target을 지운다.
      addEvent(trackId, "gone", readTargetState(track), null);
      continue;
    }

    if (!isSeen) {
      // 안 보이는 동안에는 Zone과 Held Object를 건드리지 않는다.
      tracks[trackId] = { ...track, presence };
      continue;
    }

    const [zone, hasZoneChanged] = debounce(track.zone, detection.zone, now, holdMs);
    const [held, hasHeldChanged] = updateHeld(track.held, detection.heldObjects, now, holdMs);
    if (hasZoneChanged) addEvent(trackId, "zone_changed", track.zone.stable, zone.stable);
    if (hasHeldChanged) addEvent(trackId, "object_changed", readHeld(track.held), readHeld(held));
    tracks[trackId] = { presence, zone, held };
  }
  return { state: { tracks }, events };
}
```

### reporter 트리거와 순서

- Event가 생길 때만 생성한다. 매 프레임 호출 금지.
- 생성은 한 번에 하나. 생성 중 쌓인 Event는 한 묶음(Event Batch)으로 다음 SITREP에 넣는다.
- Operator 질문(Preset Question·Spoken Question)이 대기 중인 Event Batch보다 먼저.

### 프롬프트

- system 고정 규칙: "로그에 있는 사실만, 없는 건 '관측되지 않음'이라고 답하라"(환각 방지 1차), "한국어 낱말로만 쓰라".
- SITREP 입력: Event Batch(최신 N건, 넘치면 "이전 M건 생략" 표기, 상대 시간) + Snapshot. 출력은 2~3문장 한국어.
- 질문 입력: 질문 글 + 코드가 최근 60초로 거른 Event + Snapshot. Event가 없어도 그대로 넘긴다.
- 질문의 시점(2026-11-06): 키를 누른 순간. Snapshot은 그 프레임 것, Event는 그 시각까지 것만, "N초 전"도 그 시각 기준. Spoken Question도 같다(`v`를 누른 시각). 대가: 답이 몇 초 전의 상황을 말할 수 있다.
- 한국어 이름표: Event의 영어 값(예: knife → 칼, near → 가까운 구역)을 코드의 이름표로 바꿔 프롬프트에 넣는다. Evidence는 영어 원문 그대로.
- Snapshot: 지금 화면에 있는 모든 Target과 각자의 Target State를 적은 한 줄.

### Evidence

- SITREP 하나의 근거가 된 Event 원문을 SITREP과 함께 터미널에 출력한다(환각 방지 2차, Operator가 대조).
- 끝에 프롬프트에 넣은 것과 같은 Snapshot 글 한 줄을 덧붙인다(2026-11-06). 글자로만.
- 소리로 읽지 않는다.

### 음성

- Spoken Question: 키 `v` → 고정 5초 녹음(16kHz, mono, 16비트) → Whisper(`base` 모델, 한국어) → 그 글을 질문으로 reporter에 넘긴다.
- 빈 인식 결과: Gemma를 부르지 않고 "질문을 듣지 못했습니다"를 출력·재생한다.
  - 음량 검사(2026-11-06): 말 없이 녹음해도 Whisper가 글을 지어내므로, 녹음의 음량(RMS)이 기준보다 작으면 Whisper를 부르지 않고 빈 인식 결과와 같은 경로("질문을 듣지 못했습니다")로 간다. 기준값은 임시값이고 사용자 목소리를 실측한 뒤 바꾼다.
- Preset Question `1`(지난 1분 변화)·`2`(현재 상황)는 마이크가 안 될 때를 위해 유지. 입력은 Spoken Question과 같다.
- 출력: SITREP과 질문 답변을 둘 다 Piper(`ko_KR-kss-medium`)로 읽는다. Piper는 별도 venv에서 돈다.
- 재생 중 `v`: 재생 중인 음성을 끊고 바로 녹음한다. 그때 생성 중이던 SITREP은 끝까지 쓰되 글자로만 내고, 그다음 질문을 처리한다.
- 상시 듣기·호출어는 없다.
- 녹음·STT·TTS 동안 YOLO 루프가 멈추지 않는다.

### Motion Level

- 연속 두 프레임의 차이로 숫자 하나를 구해 HUD에 표시한다. Target·Event와 무관하다.
- Jetson은 CuPy, Mac은 NumPy. 두 방식의 시간을 재서 측정표에 넣고, CuPy가 느리게 나와도 그대로 적는다.

### 두 트랙

| 갈리는 부분 | Jetson 트랙 (제출용) | Mac 트랙 (실험용) |
|---|---|---|
| 검출 모델 | YOLO11n TensorRT 엔진 | YOLO11n PyTorch 가중치 |
| 배열 모듈 | CuPy | NumPy |
| 녹음·재생 | `arecord` · `aplay` | `ffmpeg` · `afplay` |
| Whisper | whisper.cpp CUDA 빌드 | whisper.cpp Mac 빌드 |
| Gemma | llama-cpp-python CUDA | llama-cpp-python (Mac) |

- 코드는 한 벌. 플랫폼마다 갈리는 부분만 파일 단위로 교체. 상태기계·Event Log·프롬프트 생성은 공통.
- 실행 시 OS로 자동 판별(Linux → Jetson, macOS → Mac).
- Jetson에서 CuPy 불러오기 실패 시 NumPy로 넘어가지 않고 에러로 멈춘다.
- Mac 트랙은 TensorRT·CuPy만 빼고 전체 흐름을 돌린다. 발표 숫자는 Jetson 것만.
- Mac 트랙은 전용 venv를 쓰고 기존 venv(numpy 1.21.5·cv2 4.7 고정)는 건드리지 않는다.
- 제출 폴더에 Mac용 파일도 넣고 README에 "실험용"이라고 적는다.

### GPU·메모리

- Jetson 트랙은 CUDA가 기본. YOLO(TensorRT)·Gemma·Whisper·CuPy가 8GB를 같이 쓴다.
- Whisper는 CUDA 빌드가 기본.
- Piper는 CPU로 시작. 보드 체크 8-3의 합성 초를 보고 느리면 GPU를 시도한다(기준 숫자 미정).
- 메모리가 모자랄 때 내리는 순서: ① Whisper를 CPU 빌드로 → ② Gemma `n_ctx` 1024 → ③ Gemma GPU 레이어 수 줄이기. YOLO는 끝까지 GPU.

### 모델·환경

- 검출: YOLO11n, FP16 엔진.
- LLM: Gemma 4 E2B instruct Q4_K_M GGUF, `n_ctx` 2048(05 노트북 설정), llama-cpp-python.
- 보드: Orin Nano 8GB, L4T R36.5 필수(R36.4.7 OOM 버그).

### 구현 제약(프로젝트 CLAUDE.md)

- 구조: 함수 분리 + `main()` + `if __name__ == "__main__"` 가드. 노트북 셀 덤프·셀 번호 주석 금지.
- 주석: 한국어. 파일 상단에 요약·실행 방법 한두 줄, 함수 docstring에 역할·입력·출력, 블록 주석은 무엇을/왜만, 매직 넘버(T, 임계값, 해상도, 녹음 초 등)에 의미·출처 주석, 죽은 코드 삭제.
- 네이밍: PEP 8. 줄임말 금지(`bbox` → `bounding_box`, `conf` → `confidence`, `msg` → `message` 등). 불리언은 `is_`/`has_`/`should_`, 함수는 동사로 시작. 관용 import 별칭(`np`, `cv2`)만 예외. Bash는 지역 변수 `snake_case`, 환경 변수 `UPPER_SNAKE_CASE`.
- 제출: 제출 repo의 `Project03/`에 실행 파일·부가 파일 + README(「파일 구조」「실행 방식」「기타 참고 사항(문제점·AI 활용 vs 본인 기여도)」「창의성 및 확장성」) + 발표 pdf. README를 `(작성 예정)` 상태로 커밋하지 않는다.
- 코드 위치: 11-07 전에는 학습 repo에서 작업하고 제출 repo에 `Project03/` 코드를 만들지 않는다. 11-07에 제출 repo로 옮긴다.
- 커밋·푸시는 사용자가 직접 한다. 커밋 메시지는 구체적으로.

## Testing Decisions

### 좋은 테스트

- 외부 행동만 검사한다. 입력(검출 결과 목록 + 시각, Event Batch + Snapshot, 프레임 쌍)을 넣고 출력(Event 목록, 프롬프트 문자열, Motion Level 숫자)만 본다. 디바운서 내부 필드나 중간 상태 구조는 검사하지 않는다.
- 카메라·모델·마이크 없이 돈다. 시각은 인자로 주므로 실제 시간을 기다리지 않는다.
- 같은 입력 시퀀스 → 같은 Event 열(결정론).
- 임계값 상수를 이름으로 불러 경계 바로 아래·위를 검사한다(T 값이 실측으로 바뀌어도 테스트가 따라온다).

### 자동 테스트 seam (사용자 확정, 3개)

- **(a) 상태기계**: 검출 결과 + 시각 → Event 목록. 프로토타입 시나리오가 그대로 테스트 후보다.
  - 기본 흐름: `far` 출현 → `near` 접근 → 칼 꺼냄 → 이탈. 장면이 바뀐 직후에는 Event 없음, T가 지나야 한 건. `gone`에 마지막 Target State.
  - 짧은 끊김: T보다 짧은 끊김은 Event 없음. T 넘게 안 보이면 `gone`. 같은 추적 id로 다시 들어오면 `gone` 다음 `appeared`.
  - 스쳐 지나감: T보다 짧게 보였다 사라지면 Event 0건(`gone`도 없음).
  - 한순간 여러 변화: 물체를 든 채 출현하면 `appeared` 한 건에 Zone·물체가 다 들어감. 접근 + 물체 꺼냄이 같은 시각이면 `zone_changed`·`object_changed` 두 건. 두 물체를 같이 내려놓으면 `object_changed` 한 건.
  - Zone 떨림: T보다 짧은 간격으로 near/far가 번갈아 잡히면 Event 없음. near로 T 유지 뒤 한 건.
  - 물체 깜빡임(종류별): 칼 검출이 깜빡이는 동안 꺼낸 휴대폰이 T 뒤에 기록됨. 칼의 깜빡임은 칼에만 영향.
  - FPS 저하: 프레임 간격이 커져도 확정까지 걸리는 시간은 T(프레임 수가 아니라 초 기준).
  - 출현 확정 프레임에 물체가 없으면 `appeared`의 Held Object는 비어 있고 T 뒤 `object_changed`.
- **(b) 프롬프트 생성**: Event Batch + Snapshot → 프롬프트 문자열(질문이면 질문 글 + 최근 60초 Event + Snapshot).
  - system에 "로그에 있는 사실만 / 없으면 '관측되지 않음'"과 "한국어 낱말로만" 규칙이 들어 있다.
  - Event의 영어 값이 한국어 이름표로 바뀌어 들어간다.
  - Event가 N건을 넘으면 최신 N건만 들어가고 "이전 M건 생략"이 붙는다.
  - 최근 60초 밖의 Event는 질문 프롬프트에 들어가지 않는다. Event가 0건이어도 프롬프트가 만들어진다.
  - Snapshot 한 줄이 들어간다.
- **(c) Motion Level 계산(NumPy 경로)**: 합성 프레임 쌍 → 숫자 하나. 같은 두 프레임은 0.0, 전부 0인 프레임과 전부 10인 프레임은 10.0, 더 많이 다른 쌍은 더 큰 값(계산식은 Further Notes 「임시 결정」).

### 자동 테스트 없음

- 검출기, Gemma 출력, HUD는 보드에서 눈으로 확인. 음성(녹음·Whisper·Piper·재생 끊기)은 귀로 확인.
- Jetson에서 NumPy와 CuPy의 Motion Level이 같은 프레임 쌍에서 같은 값인지는 보드에서 확인(오차 범위 미정).

### 선례

- 제출 repo `Project01/tests/`: 표준 `unittest`, 가짜 객체·합성 입력으로 모델 없이 규칙을 검사한다.
  - `test_threat_classifier`의 `ThreatDebouncerTest`: 안정 프레임 수 직전까지 안 바뀜 / 깜빡임은 안 바뀜 / 현재 값이 한 번 끼면 카운트 초기화. Project03은 같은 모양을 프레임 수 대신 초 + 시각 인자로 쓴다.
  - `test_process_frame_rules`: 가짜 검출기(`FakeDetector`)로 디바운스 후 상태, 가림 중 상태 유지, 추적 대상 소멸 시 정리를 검사. (a) seam의 선례.
  - `test_preprocess`의 `make_frame`: 합성 프레임 생성. (c) seam의 선례.
  - `test_held_object`: 박스 겹침 규칙 단위 테스트. Held Object 판정(중심이 Target 박스 안)의 선례.

### 성공 판정(design-draft v2 6절, 보드·시연에서 확인)

- [ ] 상태기계 단위 테스트: 같은 입력 시퀀스 → 같은 Event 열(카메라 없이).
- [ ] Event Log가 비어 있을 때 Gemma 답변이 로그에 없는 사실을 지어내지 않는다("기록된 변화 없음"·"관측되지 않음" 같은 답이면 통과).
  - 2026-11-06 문구 변경(사용자 결정, 프롬프트는 그대로): Mac 트랙 소수 실행에서 Event Log가 빈 채 Preset Question을 물으면 "기록된 변화는 없습니다" 같은 답이 나왔고 "관측되지 않음" 글자 그대로는 4번 중 1번이었다. 로그에 없는 것을 직접 물었을 때는 2번 중 2번 "관측되지 않음"이었다. 표본이 작아 일반화하지 않는다.
- [ ] 동시 실행 중 YOLO FPS가 단독 대비 X% 이내(X 미정).
- [ ] Event 발생 → SITREP 표시까지 초 측정값 기록.
- [ ] tegrastats RAM 피크 < 7.0GB, OOM 0회(시연 10분).
- [ ] 키 `v` → 말로 질문 → 소리로 답변이 끝까지 한 번 돈다. 녹음·STT·TTS 동안 YOLO 루프가 멈추지 않는다.
- [ ] SITREP이 소리로 나오고, 재생 중 `v`를 누르면 소리가 끊기고 녹음이 시작된다.
- [ ] 마이크를 뽑아도 Preset Question `1`/`2`로 답을 받는다.
- [ ] 같은 프레임 쌍에서 NumPy와 CuPy의 Motion Level이 같은 값이다(오차 범위 미정).

## Out of Scope

- 얼굴 인식.
- 손 상세 분석(Project01 범위).
- 원격 서버, 웹 UI.
- 상시 듣기·호출어.
- LLM에 이미지 입력(ADR 0001). mmproj 크롭 1장 비교는 스트레치로만.
- LLM의 위험 판정·위험 등급(ADR 0001).
- 같은 추적 id 재진입 시 Target 번호 재부여.
- Project02 재사용(Project02 취소).
- Mac 트랙 숫자를 발표에 쓰는 것.
- 스트레치(필수 아님): 지난 Event를 압축해 두는 기억, mmproj 크롭 1장 비교, 툴 호출 1개, INT8 엔진 비교, 전력 모드(15W·MAXN) 비교, Motion Level로 YOLO 건너뛰기(추적 id가 끊길 위험).

## Further Notes

### 채점 제약

- 최종 점수 = (기술 완성도 50% · 코드 구조/가독성 40% · 제출 형식 10%) × 이해도 가중치 w. 본인이 직접 설명하지 못하는 코드는 감점된다. 설계는 사용자가 하고 AI는 보조한다. 이 스펙의 라벨은 `ready-for-agent`지만, 구현은 사용자가 직접 설명할 수 있는 단위로 쪼개고 코드를 통째로 대신 쓰지 않는다. 코드를 보조할 때는 주석·흐름 설명·예상 질문·확인 질문을 함께 준다(프로젝트 CLAUDE.md 「코드 이해도 대비」).
- 06 노트북은 강의 repo 다음 커밋에서 삭제됐다. 11-07 수업에서 요구가 바뀌면 수업 내용을 따른다.

### 미정(보드 실측 후)

- 프로세스 구조: (a) 단일 프로세스 + reporter 스레드, (b) reporter 별도 프로세스(CUDA 컨텍스트 분리). llama-cpp-python이 생성 중 GIL을 놓는지 미확인. 보드 체크 5번 결과로 결정.
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` 필요 여부(error 12 재현 시).
- 디바운스 T(초).
- near/far 임계값(박스 높이 비율).
- Event 상한 N.
- Held Object 클래스 목록(COCO 3~5종). 프로토타입의 knife·bottle·cell phone·backpack은 시연용 예시일 뿐 확정 목록이 아니다.
- Piper "느리다"의 기준 숫자(GPU 시도 여부).
- 동시 실행 YOLO FPS 허용 비율 X%.
- NumPy/CuPy Motion Level 오차 범위.
- 모든 성능 수치(FPS, tok/s, STT·TTS 초, RAM 피크). 보드 체크·Mac 설치 문서의 결과란은 전부 비어 있다.

### 임시 결정 (2026-11-06, 구현 위임에 따라 AI가 정함 — 사용자가 바꿀 수 있음)

원본에 결정이 없던 구현 세부다. 사용자가 구현을 AI에 맡기면서("전부, 구현까지") 가장 단순한 쪽으로 정했다. 숫자는 전부 임시값이고 실측 뒤 바꾼다.

- **Motion Level 계산식**: 두 프레임을 각각 채널 평균으로 흑백(float32)으로 만든 뒤, 픽셀별 절대 차의 평균. 범위는 0.0~255.0. 같은 두 프레임이면 0.0. 첫 프레임(직전 프레임 없음)은 0.0.
- **한국어 이름표**: Zone `near` → "가까운 구역", `far` → "먼 구역". Target id `#1` → "표적 1번". Held Object는 클래스 목록(임시: knife·bottle·cell phone·backpack)에 대해 칼·병·휴대폰·배낭. 물체가 없으면 "물체 없음". 이름표에 없는 값은 영어 원문 그대로 넣는다.
- **Snapshot 글 모양**: 한국어 이름표를 적용한 한 줄. 예: `현재: 표적 2명. 표적 1번 가까운 구역, 칼 보유. 표적 2번 먼 구역, 물체 없음.` 화면에 Target이 없으면 `현재: 표적 없음.`
- **상대 시간 표기**: 프롬프트를 만드는 시각 기준 정수 초로 "N초 전".
- **질문 답변 길이**: SITREP과 같이 2~3문장. 생성 토큰 상한은 임시값.
- **처리 중 키 입력**: 녹음 중이거나 질문을 처리하는 중(STT·생성·답변 재생 전)에는 새 `v`·`1`·`2`를 무시한다. 한 번에 질문 하나. 음성 재생 중 `v`만 예외(재생을 끊고 녹음 — 기존 결정).
- **동시 실행 방식(Mac 트랙)**: 단일 프로세스. YOLO 루프는 메인 스레드, reporter는 작업 스레드 하나 + 큐. 녹음·STT·TTS는 그 작업 스레드 안에서 외부 프로그램을 불러 기다린다. Jetson에서 이 구조로 YOLO FPS가 떨어지면 별도 프로세스로 바꾼다(보드 실측 후).
- **Event Log JSONL**: 실행마다 `logs/events-YYYYMMDD-HHMMSS.jsonl`. 한 줄에 Event 한 건.
- **`timestamp`**: 실행 시작 기준 상대 초(소수). 상태기계는 시각을 인자로만 받는다. 벽시계 시각은 파일 이름에만 있다.
- **Piper·재생 실패**: 글자 출력은 그대로 하고, 터미널에 경고 한 줄을 낸 뒤 계속 돈다(시연 중 멈추지 않게).

2026-11-06 오후 추가(음성 연결 구현 중 정함):

- **음성 끄기 스위치**: Reporter에 음성 사용 여부 인자를 둔다(기본 켬). 끄면 합성·재생·녹음을 전혀 부르지 않고 글자 출력만 한다. `bench.py`는 끈다 — 재생이 작업 스레드를 붙잡으면 다음 묶음의 "Event → SITREP 표시까지 초"에 앞 SITREP의 재생 시간이 섞이기 때문.
- **합성과 재생을 나눠 부름**: 재생 시작 전에는 재생 끊기가 효과가 없어서, 합성 → 확인 → 재생 순서로 부른다. 확인에서 녹음 질문이 대기 중이거나 종료 요청이 있으면 재생을 건너뛴다. 확인과 재생 시작 사이의 아주 짧은 틈은 남는다.
- **SITREP 재생 건너뛰기**: SITREP 글이 나온 시점에 Spoken Question(`v`)이 대기 중이면 그 SITREP은 글자로만 낸다. 대기 중인 것이 Preset Question(`1`·`2`)뿐이면 평소대로 재생하고, 질문은 재생이 끝난 뒤 처리한다.
- **"질문 처리 중"의 범위**: 질문이 대기열에 들어간 때부터 답 글(또는 "질문을 듣지 못했습니다")이 출력될 때까지(녹음·STT·생성 포함). 답 재생이 시작되기 전에 풀린다 — 그래서 답 재생 중 `v`는 재생을 끊고 녹음하고, `1`·`2`는 받아서 재생 뒤에 처리한다. `v`를 받으면 녹음 질문을 먼저 넣고 그다음 재생을 끊는다.
- **종료**: 종료 요청 시 재생 중인 소리를 끊고, 작업 스레드는 단계 사이(녹음 전·생성 전·재생 전)에 남은 단계를 건너뛴다. 이미 시작된 녹음(5초)과 생성은 끝까지 기다리므로 `q` 뒤 종료가 그만큼 늦어질 수 있다.
- **출력 flush**: 작업 스레드의 모든 `print`는 `flush=True`.

### 미정(원본에 결정 없음)

- 터미널 한글이 깨질 때의 대안(글꼴 파일 + PIL로 HUD 창에 그리기)은 보드 체크 7번 결과에 따라 결정.

### 미확인 사실(단정 금지)

- 보드가 R36.5로 업그레이드됐는지.
- llama-cpp-python CUDA 빌드 여부, Gemma 4 chat handler 지원 여부.
- TensorRT 엔진이 CPU 빌드 torch 환경의 ultralytics에서 추론되는지(현재 torch 2.8.0 CPU 빌드).
- YOLO + Gemma + Whisper + Piper 동시 실행 RAM 피크.
- ultralytics 추적기가 사라진 id를 얼마나 기억하는지.
- 한국어 Piper 음성이 영어 낱말·숫자·기호를 어떻게 읽는지.
- CuPy가 현재 Jetson venv에서 불러와지는지.
- 모니터에 연결한 Jetson 터미널에서 한글 SITREP이 깨지지 않는지.

### 참고

- 상태기계 프로토타입은 throwaway이며 제출물이 아니다. 여기서 확인한 규칙만 Python으로 다시 쓴다.
- 흐름은 06 노트북 A절(YOLO → JSON → LLM)과 C절(STT·TTS)을 이은 것이다.
