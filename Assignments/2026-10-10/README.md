# 2026-10-10 과제 — `01_Linux-and-Python.ipynb` NumPy 섹션 TODO

1회차(2026-10-10) 수업 노트북의 NumPy 섹션 TODO 3개와 Section Project(학생 성적 집계)를 독립 실행 가능한 `.py` 파일로 제출합니다.

## 파일 구조

- **01_Linux-and-Python.py**: 과제 전체. 위에서 아래로 순서대로 실행됩니다.
  - 점수 배열 정규화(z-score): `(scores - mean) / std`
  - 점수 배열을 0.0~1.0 범위의 `float32`로 변환(min-max): `(scores - min) / (max - min)`
  - Section Project: 학생 5명 × 과목 3개 성적 배열에서 명단 추출, 1차원 펼치기, 85점 이상 개수, 학생별 평균, 수석, 단일 과목 최고 점수(과목·학생은 `argmax` 인덱스를 `//`·`%`로 분해)

## 실행 방식

```bash
cd Assignments/2026-10-10
python 01_Linux-and-Python.py
```

인자나 입력 파일 없이 실행되며, 결과는 표준 출력으로 나옵니다.

## 의존성

- Python 3.10
- NumPy (Jetson JetPack 6 기본 설치 버전 1.21.5에서 확인, 맥 venv 1.21.5에서도 동일 출력)

## 기타 참고 사항

- 과제 파일은 `.py` 형식으로 제출했습니다. 수업 중 작성한 노트북 원본은 `Labs/2026-10-10/`에 있습니다.
- 과제는 `Assignments/YYYY-MM-DD/` 형식으로 날짜(과제 부여일)별로 정리되어 있습니다.
- 코드 작성은 직접 했고, 반복문을 `zip`·`join`으로 정리하는 부분과 README 작성에 AI 보조를 받았습니다.
