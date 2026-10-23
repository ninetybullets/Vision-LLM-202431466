# Project01 — 항복 의사 검증 HUD: 정찰 자산 영상에서 대상의 양손 상태로 DANGER / UNKNOWN / SURRENDER 판단

Section 2 Computer Vision 프로젝트. 경찰·군의 정찰 자산(드론·로봇)이 용의자나 적을 원격으로 관찰할 때, 대상이 **항복 의사를 보이는지, 항복하는 척하며 공격을 준비하는지**(손을 숨기거나 물건을 쥠)를 접근 전에 영상만으로 가려내는 상황을 가정했다.
카메라 영상에서 **MediaPipe Pose로 사람(최대 3명)을 찾고(신체 인식) → 사람마다 몸 중심을 Kalman Filter로 추적해 ID를 유지하고(객체 추적) →
Pose 손목 주변을 잘라(이미지 조정) HandLandmarker로 손가락이 펴졌는지 보고 → 전체 프레임 객체 검출(EfficientDet)이 손과 겹치면 "뭔가 들고 있음"으로 보고,
검출 박스 안을 LAB 색 분할로 윤곽을 뽑아(Color Segmentation·Contour) → 양손 상태와 손 높이(손목 vs 어깨)의 관계로 사람 상태를 정한다(관계성)**.

| 상태 | 조건 | HUD |
|---|---|---|
| `DANGER` | 한 손이라도 **가려짐**(손목이 안 보이거나, 손목은 보이는데 손을 약 1초(15프레임) 연속 못 찾음) 또는 **물체를 들고 있음**(손 근처에 객체 검출) | 빨강 |
| `SURRENDER` | **양손 모두 펴짐**(검지~소지 관절 각도 ≥ 160°) + **물체 없음** + **양 손목이 어깨보다 위** | 초록 |
| `UNKNOWN` | 그 외(손 내림, 주먹, 손은 보이나 인식 실패, 손목이 화면 밖) | 노랑 |

우선순위는 DANGER > UNKNOWN > SURRENDER. 상태는 연속 5프레임 같은 판정일 때만 바뀐다(디바운스). 사람이 잠시 가려지면 Kalman 예측으로 최대 30프레임 위치와 직전 상태를 유지한다.

## 파일 구조

```
Project01/
├── main.py               # 진입점. 카메라 루프, 프레임 처리 순서(process_frame), 손↔검출 겹침으로 OBJECT 판정·유지, fps·lock 시간
├── camera.py             # 카메라 백엔드 선택 (mac 웹캠 / Jetson GStreamer MJPG 1280x720)
├── preprocess.py         # 축소·crop(이미지 조정), GaussianBlur(필터링), BGR→LAB(색공간 변환), L 채널 CLAHE(Enhancement)
├── person_detector.py    # MediaPipe PoseLandmarker(여러 명) + 사람 판정(관절 수·어깨 너비)·몸 중심·어깨 너비 규칙
├── target_tracker.py     # 사람별 cv2.KalmanFilter + 최근접 매칭으로 ID 유지 (MultiTargetTracker)
├── hand_analyzer.py      # 손목 기준 crop 박스 계산 + HandLandmarker + 손가락 각도 → 손 소분류
├── held_object.py        # MediaPipe ObjectDetector(EfficientDet-Lite0) 전체 프레임 검출 + 손 박스 겹침 규칙 + 검출 박스 안 LAB 색 분할 윤곽
├── threat_classifier.py  # 양손 소분류 + 손 높이 → DANGER/UNKNOWN/SURRENDER, 디바운스
├── hud.py                # overlay (조준점·상태 라벨·손 박스·물체 윤곽·뼈대·예측 점선·fps·인원 요약)
├── models/
│   ├── pose_landmarker_full.task   # MediaPipe Pose (약 9MB)
│   ├── hand_landmarker.task        # MediaPipe Hand (약 8MB)
│   └── efficientdet_lite0.tflite   # MediaPipe ObjectDetector, COCO 80종 (약 14MB)
├── tests/                # unittest (카메라 없이 실행 가능)
├── docs/                 # 발표 자료 (presentation.md 대본·예상 질문, presentation.html 16:9 슬라이드)
├── requirements.txt
└── README.md
```

처리 흐름 (프레임 1장):

```
카메라 1280x720 BGR
  → resize 640폭                                   [이미지 조정]
  → PoseLandmarker(BGR→RGB, num_poses=3) 33관절     [신체 인식, 색공간 변환]
  → visibility≥0.5 관절 ≥5개 + 어깨 너비 ≥40px면 사람 (책상 위 소품 오검출 방지)
  → 어깨 중점 = 몸 중심(어깨가 안 보이면 보이는 몸통 관절 평균) → 사람별 KalmanFilter, 이전 중심과 max(100px, 0.8×어깨너비) 안이면 같은 ID   [객체 추적]
  → 짝수 프레임: ObjectDetector(EfficientDet-Lite0) 전체 프레임 → person 제외, 점수 ≥0.25 검출 목록 (홀수 프레임은 직전 결과)
  → 사람마다 왼손·오른손 (처음 본 사람은 양손, 이후 홀수 프레임에 한 손씩 번갈아 분석, 나머지 손은 직전 결과 재사용):
       손목이 화면 밖 → OUT / 손목 visibility<0.5 → HIDDEN / 손을 15프레임 연속 못 찾음(NOT_FOUND) → HIDDEN
       손목·검지 중점에 반폭 0.9×어깨너비(40~200px) crop        [이미지 조정]
       crop → HandLandmarker 21점 → 검지~소지 둘째 마디 각도 모두 ≥160° → OPEN, 아니면 CLOSED, 못 찾으면 NOT_FOUND
       검출 박스가 손 박스와 20% 이상 겹치면 → OBJECT (활짝 편 OPEN 손은 제외, 검출이 끊겨도 10프레임 유지)   [관계성: 객체↔손]
           검출 박스 → GaussianBlur → LAB → L CLAHE → 중심부 a·b 중앙값 ±12로 inRange → open/close → 최대 윤곽
                                                        [필터링, 색공간, Enhancement, Segmentation, Contour]
  → 양손 소분류 + (양 손목 y < 어깨 y) → DANGER / UNKNOWN / SURRENDER   [관계성: 자세]
  → 연속 5프레임 디바운스
  → HUD overlay
```

### 8개 기술 ↔ 코드 위치

| 기술 | 함수 | 파일 |
|---|---|---|
| 이미지 조정 | `resize_for_processing`, `crop_frame`, `locate_hand`(crop 박스) | preprocess.py, hand_analyzer.py |
| 필터링 | `blur_frame` (검출 박스 안 색 분할 전처리) | preprocess.py, held_object.py |
| 색공간 변환 | `convert_to_lab`(BGR→LAB), `PersonDetector.detect`·`HandAnalyzer.analyze`·`HeldObjectDetector.detect`(BGR→RGB) | preprocess.py 외 |
| Image Enhancement | `enhance_lightness` (L 채널 CLAHE) | preprocess.py |
| Color Segmentation | `find_object_contour` 안 `cv2.inRange`(물체 색 ± 허용폭) | held_object.py |
| Contour Detection | `find_object_contour` (`findContours` → 최대 면적, HUD 윤곽) | held_object.py |
| 객체 추적 | `MultiTargetTracker`, `TargetTracker`, `create_kalman_filter` | target_tracker.py |
| 신체 인식 | `PersonDetector`, `is_human`, `HandAnalyzer`, `is_hand_open` | person_detector.py, hand_analyzer.py |
| 관계성 (객체↔신체·자세) | `does_overlap_hand`, `apply_held_objects`, `classify_person`, `are_hands_raised` | held_object.py, main.py, threat_classifier.py |

### 왜 객체 검출기인데 클래스를 안 보나

처음엔 "파란 물건 = 무기 모형"으로 색 분할만 썼다. 실측에서 파란 노트(LAB b≈110)가 임계값 경계에 걸리고 물체가 손 crop 밖으로 나가 놓쳤다.
다음으로 "손 자체 피부색과 다른 색이 손바닥 안에 있으면 물체"를 시도했지만, 물체를 쥐면 손바닥 랜드마크가 주먹(피부) 위에 놓이고 물체는 손 밖으로 튀어나가며,
영역을 넓히면 옷·벽이 들어와 빈 주먹과 구분이 안 됐다(기각). 최종적으로 EfficientDet-Lite0(COCO)를 전체 프레임에 돌렸더니 **클래스 이름은 거의 틀리지만**(펜→toothbrush, 텀블러→cell phone)
"손 근처에 뭔가 있다"는 꽤 잡혔다. 접근 전 안전 판단엔 "무엇인지"보다 "들렸는지"가 중요하므로 **person 외 어떤 검출이든 손과 겹치면 OBJECT**로 규칙을 정했다. HUD에는 라벨을 `OBJECT(book?)`처럼 참고로만 표시한다.
색 분할·윤곽은 검출 박스(거침) 안에서 물체 영역을 정밀하게 남기는 데 쓴다. 색공간은 강의 흐름(HSV 한계 → LAB → +CLAHE)을 따라 LAB.

## 실행 방식

```bash
# Jetson (USB 카메라 /dev/video0, MJPG 1280x720 30fps)
source ~/vision-llm/.venv/bin/activate
python Project01/main.py --camera jetson

# 맥 (내장 웹캠)
python Project01/main.py --camera mac
# 맥 + 아이폰 연속성 카메라 (보통 1번, 장치 순서에 따라 0번일 수도 있음)
python Project01/main.py --camera mac --camera-index 1

# 종료: 영상 창에서 q
```

시연(서서, 카메라는 가슴 높이, 거리 **1.5~2.5m(2m 권장)**, 손 뒤에 가구·책상 물건이 없게). 실측: 가슴 높이면 1.5·2·2.5m 모두 양손 OPEN ≥97%·SURRENDER 86~100%, 1.5m는 내린 손이 화면 아래로 나가 OUT 처리됨:  (1) 양손 펴서 들기 → `SURRENDER`, (2) 펜·종이·폰 등 아무 물건 들기 → `DANGER`, (3) 한 손을 등 뒤로 → `DANGER`, (4) 손 내리고 서기 → `UNKNOWN`.

의존성: `requirements.txt`. mediapipe는 설치 시 최신 opencv-contrib를 끌어와 기존 cv2를 덮어쓰므로
`pip install --no-deps mediapipe==1.1.1` 후 `absl-py flatbuffers`를 따로 설치한다.
모델 파일 3개는 `models/`에 포함되어 있고 각 모듈이 자기 파일 위치 기준으로 찾으므로 어느 폴더에서 실행해도 된다.

테스트(카메라 불필요, 약 2초, 81개):

```bash
cd Project01
python -m unittest discover -s tests -t .
```

## 기타 참고 사항

### 확인된 문제점·한계
- **손 주변·배경의 물체가 "든 물체"로 잡힐 수 있다.** 손을 내리고 책상 앞에 앉으면 마우스·컵이(실측: 손 내림 56프레임 중 33 DANGER), 서 있어도 손 뒤에 침대가 있으면 `bed` 검출이 손 박스와 겹쳐 `OBJECT`→`DANGER`가 떴다. 대책으로 겹침 기준을 손 박스 자체(1.5배 확대 → 1.0)로 좁히고, 활짝 편 손은 물건을 못 쥐므로 OBJECT에서 제외했다. 손을 내린 채 쥔 물건도 위협이므로 손 높이로 검사를 건너뛰진 않고, 시연은 손 뒤 배경을 비운다. 점수 임계값(0.25)·유지 프레임(10)은 거리 보정 실측에서도 바꿀 근거가 없어 유지.
- **검출기 클래스는 신뢰할 수 없다**(위 설명). 그래서 위협 물체만 골라내는 화이트리스트는 포기했다. 빈 손이 가끔 `remote`/`toothbrush`로 잡혀(실측 2~10% 프레임) 10프레임 유지와 합쳐지면 짧게 `DANGER`가 뜰 수 있다.
- **검출률이 물체마다 다르다.** 펜 85%, 폰 71%, 연한 파란 종이 43%(DANGER 프레임 비율, 실측). 종이처럼 얇고 밝은 물건은 약하다.
- **사람이 아닌 물체를 사람으로 잡은 사례.** 책상 위 마우스·케이블에 Pose가 관절 20개 이상을 "보임"으로 찍었다. 관절 수로는 못 거르고 어깨 너비가 10~39px로 작았기에 `is_human`에 **어깨 너비 ≥ 40px** 조건을 넣어 해결했다. 대가로 아주 멀리 있는 사람은 무시된다.
- **몸을 크게 돌리면 ID가 바뀔 수 있다.** 몸 중심(어깨 중점)이 한 프레임에 허용 거리(100px, 가까우면 어깨 너비×0.8)보다 멀리 움직이면 새 ID가 된다. 실측에서 가까운 거리(어깨 160px 이상)에서 ID가 바뀐 뒤 허용 거리를 어깨 너비에 비례시켰다.
- **fps.** 맥(M2) 실측 10~15fps(1명). 프레임당 카메라 읽기 13ms + Pose 24ms(사람 있을 때) + EfficientDet 37ms(격프레임, 평균 18) + HandLandmarker 23ms(격프레임, 손 보일 때 평균 12). Jetson은 아직 측정하지 않았다. 느리면 검출 주기(`DETECT_EVERY_FRAMES`)를 늘리거나 `pose_landmarker_lite.task`로 교체한다.
- **등 뒤로 숨긴 손을 Pose가 "보인다"고 추정할 때가 있다.** 그래서 visibility만으로는 가림을 못 잡은 적이 있고(실측 4/58), 손 crop에서 손을 1초(15프레임) 연속 못 찾으면(NOT_FOUND) 가림으로 간주하도록 했다. 거리 보정 실측에서 2.5m 빈손 내림은 NOT_FOUND가 0, 자세 전환 중 최장 연속 12라 15를 유지했고, 등 뒤 손은 2m에서 DANGER 92~100%.
- `PREDICTING` 동안 몸통 박스 크기는 고정값(반폭 60px)이고 손 분석은 하지 않는다(Pose가 없으면 손목 위치를 모름).
- 몸 중심을 처음엔 "어깨·엉덩이 중 보이는 관절 평균"으로 뒀더니 앉은 자세에서 엉덩이가 보였다 안 보였다 할 때 중심이 약 100px씩 점프해 같은 사람이 새 ID로 바뀌었다. 어깨 중점 우선으로 바꿔 해결했다.

### AI 활용 vs 본인 기여
- 설계(시나리오, 3상태 정의와 우선순위, 손 소분류, 항복 조건에 손 높이 추가, 물체 판별 방식의 세 차례 교체 결정, crop 크기·디바운스 수치, 다중 인원 매칭 방식)는 AI와 질문-답(grilling) 방식으로 **본인이 결정**했고, 결정 기록을 남겼다.
- 물체 판별 방식은 본인이 카메라 앞에서 동작하고 AI가 캡처 스크립트로 수치를 모아 비교한 뒤(색 범위 / 피부색 대비 / 검출기) 데이터를 보고 본인이 골랐다.
- 코드 초안은 AI(Claude)가 모듈 단위로 작성했고, 본인이 함수 단위로 역할을 질문하며 검토했다.
- 검증 기준(assertion)을 먼저 정하고 unittest로 먼저 작성한 뒤 구현했다. 실측에서 ID가 바뀌는 문제를 발견하고 원인(엉덩이 visibility 깜빡임 → 몸 중심 점프)을 기록·수정했다.
- 구현 후 별도 코드 리뷰에서 "화면 밖 손목도 MediaPipe가 visibility를 낮게 줘서 HIDDEN(위험)으로 처리되는 버그"를 찾아 화면 밖 검사를 먼저 하도록 고쳤다.

## 창의성 및 확장성

- **객체 추적 + 신체 인식 + 관계성을 한 판정에 묶었다.** 사람(Pose) → 손(Hand) → 손 근처 물체(검출기+색 윤곽) 순으로 좁혀 가며 보고, 손 높이(손목 vs 어깨)까지 합쳐 "위협 / 보류 / 협조 의사"라는 의미 있는 결론을 낸다.
- **검출기를 "분류기"가 아니라 "존재 감지기"로 쓴다.** 경량 모델의 약점(클래스 오류)을 규칙 설계로 피해 갔다.
- **손 영역 crop을 어깨 너비에 비례**시켜 사람이 멀든 가깝든 손이 crop에 같은 비율로 들어오게 했다.
- **다중 인원**: 사람별 Kalman + 최근접 매칭으로 ID를 유지하고 사람마다 상태를 따로 디바운스한다. 요약(`N PERSONS  k DANGER`)으로 전체 상황을 한 줄로 보여 준다.
- 확장: 검출 클래스를 믿을 수 있는 더 큰 모델(EfficientDet-Lite2, YOLO)로 바꾸면 위협 물체 화이트리스트(knife, scissors, bat)가 가능. "항복 = 손바닥을 카메라로"(handedness + 랜드마크 외적 부호로 손바닥/손등 판별, 실측에서 일관성 확인) 조건 추가. Pose 속도 벡터로 접근/이탈 방향 표시. Jetson에서 TensorRT 가속.
