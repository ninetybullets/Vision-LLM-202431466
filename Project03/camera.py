"""Jetson USB 카메라(GStreamer MJPG 640x480 30fps) 열기. 파이프라인은 JETSON_PIPELINE 한 곳에만 있다 — 보드에서 확인한 파이프라인이 다르면 그 문자열만 바꾼다.

사용: capture = open_camera()
맥 개발용 사본(학습 repo final_project/code/camera.py)에는 mac 분기가 남아 있고, 제출본은 Jetson 전용이다.
"""
import cv2

# Jetson USB 카메라 파이프라인 — 이 상수 하나가 유일한 Jetson 파이프라인이다. 보드 진단(jetson_precheck.md P1)에서
# 되는 파이프라인이 다르면 이 문자열만 바꾼다. MJPG 640x480 30fps → jpegdec → BGR, 지연 최소화를 위해 버퍼 1장.
# 해상도 640x480인 이유(2026-11-08): 1280x720에서는 jpegdec(CPU)·Motion Level 복사·imshow 비용으로 앱 FPS가 7~8이었고
# Gemma 생성 중에는 5까지 떨어졌다. YOLO 입력은 어차피 640이라 검출에는 손해가 없고, 프레임당 픽셀이 1/3로 준다.
# Zone 판정은 박스 높이/프레임 높이 비율이라 해상도와 무관하다
JETSON_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=640,height=480,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
# Jetson에서 프레임 한 장을 기다리는 제한 시간. OpenCV 4.8 GStreamer 백엔드 기본값은 30초라 카메라가 멈추면 창이 30초 동안
# 하얗게 굳는다(11-07 보드 실측). open 때 1장이 미리 받아지고 정상이면 30fps이므로 3초 안에 안 오면 고장으로 본다
JETSON_READ_TIMEOUT_MILLISECONDS = 3000


def open_camera():
    """Jetson USB 카메라를 JETSON_PIPELINE으로 열어 VideoCapture를 돌려준다. 못 열면 RuntimeError."""
    capture = cv2.VideoCapture(JETSON_PIPELINE, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. /dev/video0 연결·점유 프로세스·GStreamer 파이프라인(JETSON_PIPELINE)을 확인하세요.")
    capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, JETSON_READ_TIMEOUT_MILLISECONDS)
    return capture
