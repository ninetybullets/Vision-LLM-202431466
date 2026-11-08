"""Jetson USB 카메라(GStreamer MJPG 1280x720 30fps) 열기 + 처리용 축소(640x360). 파이프라인은 JETSON_PIPELINE 한 곳에만 있다 — 보드에서 확인한 파이프라인이 다르면 그 문자열만 바꾼다.

사용: capture = open_camera(); is_read, frame = capture.read(); frame = shrink_frame(frame)
맥 개발용 사본(학습 repo final_project/code/camera.py)에는 mac 분기가 남아 있고, 제출본은 Jetson 전용이다.
"""
import cv2

# Jetson USB 카메라 파이프라인 — 이 상수 하나가 유일한 Jetson 파이프라인이다. 보드 진단(jetson_precheck.md P1)에서
# 되는 파이프라인이 다르면 이 문자열만 바꾼다. MJPG 1280x720 30fps → jpegdec → BGR, 지연 최소화를 위해 버퍼 1장.
# 2026-11-08 카메라 자체를 640x480으로 열어 봤으나 이 카메라에서 열리지 않아 1280x720으로 되돌리고, 대신 읽은 프레임을
# 코드에서 줄인다(shrink_frame). jpegdec 비용은 남지만 그 뒤 단계(YOLO 전처리·Motion Level·copy·HUD·imshow)는 1/4 픽셀로 돈다
JETSON_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
# Jetson에서 프레임 한 장을 기다리는 제한 시간. OpenCV 4.8 GStreamer 백엔드 기본값은 30초라 카메라가 멈추면 창이 30초 동안
# 하얗게 굳는다(11-07 보드 실측). open 때 1장이 미리 받아지고 정상이면 30fps이므로 3초 안에 안 오면 고장으로 본다
JETSON_READ_TIMEOUT_MILLISECONDS = 3000
# 처리용 프레임 크기(폭, 높이). 1280x720의 절반 — YOLO 입력이 640이라 검출 손해가 없고, Zone 판정은 높이 비율이라 해상도와 무관하다.
# Project01에서도 같은 방식(큰 해상도로 열고 코드에서 축소)으로 문제가 없었다
PROCESS_FRAME_SIZE = (640, 360)


def open_camera():
    """Jetson USB 카메라를 JETSON_PIPELINE으로 열어 VideoCapture를 돌려준다. 못 열면 RuntimeError."""
    capture = cv2.VideoCapture(JETSON_PIPELINE, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. /dev/video0 연결·점유 프로세스·GStreamer 파이프라인(JETSON_PIPELINE)을 확인하세요.")
    capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, JETSON_READ_TIMEOUT_MILLISECONDS)
    return capture


def shrink_frame(frame):
    """카메라 프레임을 처리용 크기(PROCESS_FRAME_SIZE)로 줄인다. 읽은 직후 한 번 부른다(main·bench).

    입력: BGR 프레임(numpy 배열). 출력: PROCESS_FRAME_SIZE 크기의 새 BGR 프레임.
    INTER_AREA는 축소 때 픽셀을 평균 내는 방식이라 가장자리 계단이 적다(OpenCV 권장).
    """
    return cv2.resize(frame, PROCESS_FRAME_SIZE, interpolation=cv2.INTER_AREA)
