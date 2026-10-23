"""프레임 전처리: 축소·crop(이미지 조정), 가우시안 블러(필터링), BGR→LAB(색공간 변환), L 채널 CLAHE(Image Enhancement).

사용: lab = enhance_lightness(convert_to_lab(blur_frame(crop_frame(frame, box))))
"""
import cv2

# 처리 해상도 폭. 1280→640으로 줄이면 픽셀 수 1/4라 Jetson에서도 실시간 유지 (이미지 조정)
PROCESS_WIDTH = 640
# MJPG 압축 노이즈·카메라 잡음을 눌러 inRange 마스크가 자잘하게 찢기는 것을 막는 커널 크기 (필터링)
BLUR_KERNEL_SIZE = (5, 5)
# CLAHE: 노트북 Enhancement 절 기본값(clipLimit 2.0, 8x8 타일). 그늘진 손 안 물체의 밝기(L)를 끌어올린다.
# 객체 생성 비용이 있어 모듈 로드 때 한 번만 만든다
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def resize_for_processing(frame):
    """프레임 폭을 PROCESS_WIDTH로 축소한다(비율 유지). 이후 모든 처리·표시는 이 축소 프레임 좌표계에서 한다.

    입력: BGR 프레임. 출력: 축소된 BGR 프레임. INTER_AREA는 축소 시 모아레가 적은 보간법.
    """
    height, width = frame.shape[:2]
    scale = PROCESS_WIDTH / width
    return cv2.resize(frame, (PROCESS_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA)


def crop_frame(frame, box):
    """(x, y, w, h) 박스 영역만 잘라낸다. numpy 뷰라 복사 비용 없음. 박스는 프레임 안으로 이미 잘려 있어야 한다."""
    x, y, width, height = box
    return frame[y : y + height, x : x + width]


def blur_frame(frame):
    """가우시안 블러로 고주파 잡음을 줄인다. sigma 0은 커널 크기에서 자동 계산."""
    return cv2.GaussianBlur(frame, BLUR_KERNEL_SIZE, 0)


def convert_to_lab(frame):
    """BGR → LAB. 밝기(L)와 색(a: 초록~빨강, b: 파랑~노랑)이 분리돼 조명이 바뀌어도 색 범위를 유지할 수 있다.

    OpenCV uint8 LAB은 L 0~255, a·b는 128이 중립이고 파랑은 b가 128보다 작은 쪽.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)


def enhance_lightness(lab):
    """L 채널에만 CLAHE를 적용해 어두운 곳의 대비를 올린다. a/b는 건드리지 않아 색 임계값이 유지된다."""
    l_channel, a_channel, b_channel = cv2.split(lab)
    return cv2.merge((CLAHE.apply(l_channel), a_channel, b_channel))
