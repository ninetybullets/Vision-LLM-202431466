# Recon SITREP Assistant (Project03)

카메라 영상에서 검출·추적한 사람의 변화를 기록하고, 그 기록을 근거로 로컬 LLM이 상황 보고를 쓰는 최종 프로젝트의 용어집.

## Language

### 관측 대상

**Target**:
추적 id 하나가 붙은, 화면에 보이는 사람 한 명. 화면을 벗어났다가 다시 들어오면 다른 Target이다.
_Avoid_: person, 사람, track

**Target State**:
Target 하나가 지금 가진 Held Object들과 Zone의 조합.
_Avoid_: status, scene state

**Zone**:
카메라에서 본 Target의 거리 구분. `near`와 `far` 둘뿐이며, 화면에 없는 것은 Zone이 아니다.
_Avoid_: distance, range, gone

**Held Object**:
Target이 보유한 것으로 판정된 물체. 미리 정한 물체 종류만 대상이다.
_Avoid_: item, weapon, prop

**Motion Level**:
직전 화면과 지금 화면이 얼마나 다른지 나타내는 숫자 하나. Target이나 Event와 무관하다.
_Avoid_: motion score, 움직임 감지

### 기록

**Event**:
Target 하나에 생긴 변화 한 건. 종류는 `appeared`, `gone`, `zone_changed`, `object_changed` 넷뿐이다.
_Avoid_: alert, detection, state change

**Event Log**:
Event만 시간순으로 쌓은 기록. 변화가 없는 프레임은 남기지 않는다.
_Avoid_: history, frame log

**Event Batch**:
SITREP 한 번에 함께 들어가는 Event 묶음.
_Avoid_: window, queue

### 보고

**Snapshot**:
지금 화면에 있는 모든 Target과 각자의 Target State를 적은 한 줄. Event와 달리 변화가 아니라 현재를 말한다.
_Avoid_: scene state, current status

**SITREP**:
Event Batch를 근거로 LLM이 쓰는 짧은 상황 보고. 위험 판정은 담지 않는다. 터미널로만 낸다.
_Avoid_: summary, alert, report

**Evidence**:
SITREP 하나의 근거가 된 Event들. SITREP과 함께 보여 준다.
_Avoid_: source, citation

**Operator**:
SITREP을 읽고 질문하며, 화면을 보고 수동 방송을 고르는 사람.
_Avoid_: user

**Spoken Question**:
Operator가 말로 한 질문을 글로 옮긴 것.
_Avoid_: voice command, 음성 명령

### 방송

**Announcement / 안내방송**:
스피커로 화면 속 사람에게 내보내는 고정 문구 하나. 규칙 코드가 고르고 LLM은 쓰지 않는다.
_Avoid_: alert, warning, 음성 출력

**Manual Announcement / 수동 방송**:
Operator가 키 1·2·3으로 고른 Announcement.
_Avoid_: preset, hotkey
