"""카메라 백엔드 선택: 맥 내장 웹캠(인덱스 0) 또는 Jetson USB 카메라(GStreamer MJPG 1280x720 30fps).

사용: capture = open_camera("mac" | "jetson")
"""
import cv2

# Jetson 원본 Camera_MJPG.py 파이프라인 그대로. MJPG 1280x720 30fps → jpegdec → BGR, 지연 최소화를 위해 버퍼 1장
JETSON_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
# 맥 기본 카메라 인덱스(AVFoundation). 연속성 카메라(아이폰)가 연결되면 보통 1번으로 잡힘 → --camera-index로 지정
MAC_CAMERA_INDEX = 0
# 맥 웹캠 기본 1920x1080을 Jetson과 같은 1280x720으로 낮춰 캡처 지연·축소 비용을 줄인다
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


def build_capture_arguments(backend, camera_index=MAC_CAMERA_INDEX):
    """backend 이름을 cv2.VideoCapture에 넘길 인자 튜플로 바꾼다. 모르는 이름이면 ValueError.

    camera_index는 mac 백엔드에서만 쓰인다(내장 0, 아이폰 연속성 카메라 등 외부 장치는 1 이상).
    """
    if backend == "mac":
        return (camera_index,)
    if backend == "jetson":
        return (JETSON_PIPELINE, cv2.CAP_GSTREAMER)
    raise ValueError(f"지원하지 않는 카메라 백엔드: {backend} (mac | jetson)")


def open_camera(backend, camera_index=MAC_CAMERA_INDEX):
    """카메라를 열어 VideoCapture를 돌려준다. 못 열면 RuntimeError."""
    capture = cv2.VideoCapture(*build_capture_arguments(backend, camera_index))
    if backend == "mac":
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not capture.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다 (backend={backend}, index={camera_index}). 권한·연결·GStreamer 지원을 확인하세요.")
    return capture
