"""
MJPG 카메라에서 사용자가 지정한 네모난 물체를 실시간으로 추적한다.

실행:
    python Camera_test2.py

사용법:
    1. 첫 프레임에서 마우스로 네모난 물체를 드래그한다.
    2. Enter 또는 Space를 눌러 선택을 확정한다.
    3. 추적 중 q 키를 누르면 종료한다.
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

WINDOW_NAME = "Square Object Tracking"
SELECTION_WINDOW_NAME = "Select Square Object"
SEARCH_PADDING = 1.5
MATCH_THRESHOLD = 0.45


def crop_template(frame, bounding_box):
    """프레임에서 bounding box에 해당하는 template을 잘라낸다."""
    x, y, width, height = bounding_box
    return frame[y:y + height, x:x + width].copy()


def create_search_region(frame_shape, bounding_box):
    """이전 bounding box 주변의 template 검색 영역을 계산한다."""
    frame_height, frame_width = frame_shape[:2]
    x, y, width, height = bounding_box
    padding_x = int(width * SEARCH_PADDING)
    padding_y = int(height * SEARCH_PADDING)

    left = max(0, x - padding_x)
    top = max(0, y - padding_y)
    right = min(frame_width, x + width + padding_x)
    bottom = min(frame_height, y + height + padding_y)
    return left, top, right - left, bottom - top


def track_template(frame, template, previous_box):
    """이전 위치 주변에서 template을 찾아 새 bounding box를 반환한다."""
    search_box = create_search_region(frame.shape, previous_box)
    search_x, search_y, search_width, search_height = search_box
    search_region = frame[
        search_y:search_y + search_height,
        search_x:search_x + search_width,
    ]

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    template_height, template_width = template_gray.shape[:2]

    if (
        template_width > search_gray.shape[1]
        or template_height > search_gray.shape[0]
    ):
        return None, 0.0

    matching_result = cv2.matchTemplate(
        search_gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED,
    )
    _, score, _, match_location = cv2.minMaxLoc(matching_result)

    if score < MATCH_THRESHOLD:
        return None, score

    match_x, match_y = match_location
    tracked_box = (
        search_x + match_x,
        search_y + match_y,
        template_width,
        template_height,
    )
    return tracked_box, score


def calculate_box_center(bounding_box):
    """Bounding box의 중심 좌표를 계산한다."""
    x, y, width, height = bounding_box
    return x + width // 2, y + height // 2


def draw_tracking_result(frame, bounding_box, score):
    """추적된 사각형, 중심점, 최소 외접원을 프레임에 그린다."""
    result = frame.copy()
    if bounding_box is None:
        cv2.putText(
            result,
            "Object: not detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return result

    x, y, width, height = bounding_box
    center = calculate_box_center(bounding_box)
    radius = max(width, height) // 2
    rectangle_contour = np.array(
        [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.drawContours(result, [rectangle_contour], -1, (0, 255, 0), 2)
    cv2.circle(result, center, radius, (255, 0, 0), 2)
    cv2.circle(result, center, 6, (0, 255, 255), cv2.FILLED)

    cv2.putText(
        result,
        f"Center: {center}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        result,
        f"Size: {width}x{height}  Score: {score:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    return result


def select_object(frame):
    """첫 프레임에서 추적할 네모난 물체를 마우스로 선택한다."""
    bounding_box = cv2.selectROI(
        SELECTION_WINDOW_NAME,
        frame,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyWindow(SELECTION_WINDOW_NAME)

    x, y, width, height = map(int, bounding_box)
    if width == 0 or height == 0:
        return None, None
    return (x, y, width, height), crop_template(frame, (x, y, width, height))


def run_camera():
    """카메라를 열고 선택된 네모난 물체를 추적한다."""
    capture = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise RuntimeError("MJPG 카메라를 열 수 없습니다.")

    try:
        success, first_frame = capture.read()
        if not success:
            raise RuntimeError("카메라에서 첫 프레임을 읽을 수 없습니다.")

        bounding_box, template = select_object(first_frame)
        if bounding_box is None or template is None:
            return
        while True:
            success, frame = capture.read()
            if not success:
                break

            tracked_box, score = track_template(frame, template, bounding_box)
            result = draw_tracking_result(frame, tracked_box, score)
            cv2.imshow(WINDOW_NAME, result)

            if tracked_box is not None:
                bounding_box = tracked_box

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main():
    """네모난 물체 추적을 시작한다."""
    run_camera()


if __name__ == "__main__":
    main()
