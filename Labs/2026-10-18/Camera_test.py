"""
MJPG 카메라에서 농구공의 중심과 최소 외접원을 실시간으로 추적한다.

실행:
    python Camera_test.py

종료:
    카메라 창에서 q 키 입력
"""

import cv2
import numpy as np


PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

LOWER_BASKETBALL_LAB = np.array([70, 130, 70])
UPPER_BASKETBALL_LAB = np.array([180, 230, 185])
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
WINDOW_NAME = "Basketball Tracking"


def create_basketball_mask(frame):
    """BGR 프레임에 LAB 변환과 CLAHE를 적용해 농구공 binary mask를 만든다."""
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(frame_lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l_channel = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l_channel, a_channel, b_channel))

    return cv2.inRange(
        enhanced_lab,
        LOWER_BASKETBALL_LAB,
        UPPER_BASKETBALL_LAB,
    )


def apply_morphological_operations(mask):
    """Mask의 작은 노이즈를 제거하고 끊어진 영역을 연결한다."""
    opened_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        KERNEL,
        iterations=2,
    )
    return cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, KERNEL)


def find_largest_contour(mask):
    """Mask에서 가장 큰 외곽선을 반환한다. 검출되지 않으면 None을 반환한다."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def calculate_centroid(contour):
    """Contour의 moments로 무게중심 좌표를 계산한다."""
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    return center_x, center_y


def draw_tracking_result(frame, mask, contour):
    """가장 큰 contour의 영역, 외접원, 무게중심을 프레임에 표시한다."""
    result = frame.copy()
    if contour is None:
        cv2.putText(
            result,
            "Basketball: not detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return result

    object_mask = np.zeros_like(mask)
    cv2.drawContours(object_mask, [contour], -1, 255, cv2.FILLED)

    overlay = np.zeros_like(result)
    overlay[:, :, 1] = object_mask
    result = cv2.addWeighted(result, 1.0, overlay, 0.25, 0)

    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    circle_center = int(center_x), int(center_y)
    circle_radius = max(1, int(radius))
    cv2.circle(result, circle_center, circle_radius, (255, 0, 0), 3)

    centroid = calculate_centroid(contour)
    if centroid is not None:
        cv2.circle(result, centroid, 6, (0, 255, 255), cv2.FILLED)
        cv2.line(result, centroid, circle_center, (0, 255, 255), 1)
        cv2.putText(
            result,
            f"Centroid: {centroid}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        result,
        f"Radius: {circle_radius}px",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )
    return result


def process_frame(frame):
    """한 프레임에서 mask 생성부터 추적 결과 그리기까지 수행한다."""
    basketball_mask = create_basketball_mask(frame)
    final_mask = apply_morphological_operations(basketball_mask)
    largest_contour = find_largest_contour(final_mask)
    result = draw_tracking_result(frame, final_mask, largest_contour)
    return result, final_mask, largest_contour


def run_camera():
    """MJPG 카메라를 열고 프레임별 농구공 추적 결과를 표시한다."""
    capture = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise RuntimeError("MJPG 카메라를 열 수 없습니다.")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            result, _, _ = process_frame(frame)
            cv2.imshow(WINDOW_NAME, result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main():
    """카메라 기반 농구공 추적을 시작한다."""
    run_camera()


if __name__ == "__main__":
    main()
