# 카메라 및 관련 환경 보고서

- 확인일: 2026-10-18
- 작업 디렉터리: `/home/ninetybullets/vision-llm`
- 기준 환경: `/home/ninetybullets/vision-llm/.venv`

## 카메라 사양

카메라 코드의 GStreamer 파이프라인 기준으로 확인되는 입력 사양입니다.

| 항목 | 내용 |
|---|---|
| 장치 경로 | `/dev/video0` |
| 입력 방식 | V4L2 (`v4l2src`) |
| 압축 형식 | MJPG / JPEG |
| 해상도 | `1280 x 720` |
| 프레임률 | `30 fps` |
| 디코더 | `jpegdec` |
| OpenCV 입력 | `cv2.CAP_GSTREAMER` |
| 출력 프레임 형식 | BGR |

- 제조사, 제품명, 센서 모델, 렌즈 사양은 현재 코드와 기존 보고서에 기록되어 있지 않습니다.
- 실제 연결 장치의 상세 정보가 필요하면 `v4l2-ctl --all --device=/dev/video0`로 추가 확인해야 합니다.

## 설치된 관련 소프트웨어

### Python 가상환경

| 항목 | 버전 |
|---|---:|
| Python | `3.10.12` |
| OpenCV (`cv2`) | `4.8.0` |
| NumPy | `1.21.5` |

### Jetson 영상 및 AI 환경

| 항목 | 버전 또는 상태 |
|---|---:|
| Ubuntu | `22.04.5 LTS` |
| L4T | `R36.4.7` |
| JetPack 계열 | `6.2.1` 계열로 판단 |
| CUDA | `12.6` |
| cuDNN | `9.3.0` |
| TensorRT | `10.3.0` |
| GStreamer | 카메라 파이프라인에서 사용 |
| V4L2 | 카메라 장치 입력에 사용 |

## 카메라 관련 프로젝트 파일

- `Camera_MJPG.py`
- `Camera_test.py`
- `Camera_test2.py` ~ `Camera_test7.py`
- `Camera_test3_explanation.html`

## 실행 예시

```bash
source .venv/bin/activate
python Camera_MJPG.py
```

카메라 창에서 `q` 키를 누르면 종료합니다.

## 참고

이 문서는 코드에 정의된 카메라 입력 설정과 기존 시스템 환경 보고서, 현재 활성 가상환경의 Python 패키지 버전을 바탕으로 작성되었습니다.
