# 2026-10-17 과제 — 모폴로지 연산으로 최종 농구공 Contour 선택 및 출력

2회차(2026-10-17) 수업 노트북 `02_Computer-Vision.ipynb`의 Contour Detection 절 TODO(모폴로지 연산 → 외곽선 검출 → 최종 농구공 영역 선택)를 독립 실행 가능한 `.py` 파일로 제출합니다.

## 파일 구조

- **basketball_contour_detection.py**: 과제 전체. `main()`이 아래 함수를 순서대로 호출합니다.
  - `load_enhanced_basketball()`: `src/images/basketball_crop.jpg`를 읽어 LAB 색공간의 L 채널에 CLAHE를 적용하고, 농구공 LAB 색 범위로 binary mask 생성 (수업 CLAHE 셀과 동일)
  - `apply_morphological_operations()`: 타원 5×5 커널로 opening 2회 → closing 1회 적용해 노이즈 제거
  - `find_ball_contours()`: `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`로 바깥 외곽선 검출
  - `select_and_draw()`: 면적이 가장 큰 contour를 농구공으로 선택, 그 영역만 남기고 초록 외곽선을 그림
  - 출력: 색 마스크 / 모폴로지 후 마스크 / 최종 결과 3장을 한 창에 표시
- 입력 이미지는 저장소 루트의 `src/images/basketball_crop.jpg`를 파일 위치 기준 상대 경로(`Path(__file__)` 두 단계 위)로 참조하므로 어느 위치에서 실행해도 됩니다.

## 실행 방식

```bash
cd Assignments/2026-10-17
python basketball_contour_detection.py
```

인자나 입력 없이 실행되며, matplotlib 창에 결과 3장이 표시됩니다.

## 의존성

- Python 3.10
- OpenCV 4.7.0, NumPy 1.21.5, Matplotlib 3.7.5 (맥 venv에서 확인. Jetson JetPack 6 기본 설치 버전과 동일 계열)

## 기타 참고 사항

- 모폴로지 설정은 노트북의 오토바이 예제처럼 횟수·순서·커널을 바꿔 가며 contour 수와 최대 면적을 비교해 정했습니다.
  - opening 횟수: 원본 마스크 contour 37개 → 1회 8개 → 2회 3개 → 3회 3개. 3회부터는 노이즈 감소 없이 공 면적만 줄어 2회로 결정.
  - closing: 면적 변화 1~3% 수준으로 효과가 작았지만 경계의 작은 틈을 메우기 위해 1회 적용. closing을 먼저 하면 노이즈가 뭉쳐 opening으로 지워지지 않아 opening → closing 순서 유지.
  - 커널: 타원 5×5 vs 사각 5×5 vs 타원 7×7 모두 결과 차이가 거의 없어 둥근 물체에 맞는 타원 5×5 사용.
  - 가우시안 블러: 오토바이 예제처럼 마스크 생성 전 7×7 블러를 시험했으나, 조명 반사 영역이 공 경계에 번져 contour가 공 밖으로 늘어나(최소 외접원 채움 비율 0.66 → 0.57) 제외했습니다.
- 최종 마스크에서 면적 1위(약 11100)와 2위(약 390)의 차이가 커서 `max(contours, key=cv2.contourArea)`로 농구공이 안정적으로 선택됩니다.
- 한계: 공 윗부분이 조명에 날아가 LAB 색 범위에서 빠지기 때문에 contour가 완전한 원이 되지 않습니다. 이는 색 segmentation 단계의 한계로, 모폴로지로는 보정되지 않았습니다.
- 과제 파일은 `.py` 형식으로 제출했습니다. 수업 중 작성한 노트북 원본은 `Labs/2026-10-17/`에 있습니다.
- AI 활용: 파일 뼈대(import, 상대 경로 설정, 수업 CLAHE 셀을 옮긴 `load_enhanced_basketball()`)와 모폴로지 설정 비교 실험 스크립트, docstring·주석 정리, README 작성에 AI 보조를 받았습니다. 모폴로지·외곽선 검출·최종 선택 함수 3개와 `main()`의 호출 흐름은 노트북 셀을 참고해 직접 작성했고, 실험 결과를 보고 opening 2회·closing 1회·블러 제외를 결정한 것도 본인입니다.
