"""
MJPG 카메라에서 노란색 원을 검출하고 칼만필터로 추적한다.

실행:
    python Camera_test4.py

종료:
    카메라 창에서 q 키 입력
"""

import cv2
import numpy as np

# USB MJPG 카메라의 영상을 OpenCV가 BGR 프레임으로 받기 위한 GStreamer 파이프라인
PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# HSV 색공간에서 노란색으로 인정할 최소/최대 범위
YELLOW_LOWER = np.array([20, 100, 100], dtype=np.uint8)
YELLOW_UPPER = np.array([40, 255, 255], dtype=np.uint8)
# 작은 노이즈를 제거하고 객체 영역을 정리할 5x5 타원형 커널
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# 이 면적보다 작은 contour는 탁구공이 아닌 노이즈로 간주
MIN_CONTOUR_AREA = 300
# 결과 영상과 mask를 표시할 창 이름
WINDOW_NAME = "Yellow Circle Kalman Tracking"
MASK_WINDOW_NAME = "Yellow Mask"

DT = 1.0
A_t = np.array(
    [
        [1, 0, DT, 0],
        [0, 1, 0, DT],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=float,
)
C_t = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
PROCESS_NOISE = np.diag([1, 1, 4, 4]).astype(float)
MEASUREMENT_NOISE = np.diag([25, 25]).astype(float)

def create_yellow_mask(frame):
    """BGR 프레임에서 HSV 색공간 기반 노란색 mask를 생성한다."""
    # Gaussian Blur로 카메라 노이즈와 작은 색상 변화를 완화
    blurred_frame = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    # 범위 안에 있는 픽셀은 흰색(255), 나머지는 검은색(0)으로 변환
    mask = cv2.inRange(hsv_frame, YELLOW_LOWER, YELLOW_UPPER)

    # Opening으로 작은 흰색 노이즈 제거
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        KERNEL,
        iterations=2,
    )
    # Closing으로 객체 내부의 작은 검은 구멍과 끊어진 부분 연결
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        KERNEL,
        iterations=2,
    )


def find_yellow_circle(mask):
    """mask에서 충분히 큰 노란색 영역의 contour를 반환한다."""
    # 흰색 영역들의 외곽선을 찾고, 내부 구멍은 무시
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    # 너무 작은 영역은 노이즈로 보고 후보에서 제외
    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= MIN_CONTOUR_AREA
    ]

    if not valid_contours:
        return None
    # 남은 후보 중 면적이 가장 큰 영역을 탁구공으로 선택
    return max(valid_contours, key=cv2.contourArea)


def get_circle_measurement(contour):
    """contour의 외접원 중심과 반지름을 칼만필터 측정값으로 반환한다."""
    if contour is None:
        return None

    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    return np.array([[center_x], [center_y]], dtype=float), max(1, int(radius))


def draw_tracking(frame, contour, predicted_state, corrected_state):
    """검출값, 예측값, 보정값을 프레임에 표시한다."""
    # 원본 프레임을 보존하면서 결과를 그릴 복사본 생성
    result = frame.copy()

    measurement = get_circle_measurement(contour)
    if measurement is not None:
        measured_center, radius = measurement
        measured_point = tuple(np.round(measured_center.ravel()).astype(int))
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
        cv2.circle(result, measured_point, radius, (0, 255, 0), 2)
        cv2.circle(result, measured_point, 5, (0, 0, 255), cv2.FILLED)
        cv2.putText(result, "Measured", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    predicted_point = tuple(np.round(predicted_state[:2, 0]).astype(int))
    corrected_point = tuple(np.round(corrected_state[:2, 0]).astype(int))
    cv2.circle(result, predicted_point, 10, (0, 255, 255), 2)
    cv2.circle(result, corrected_point, 6, (255, 0, 255), cv2.FILLED)
    cv2.putText(result, "Yellow: predicted / Magenta: corrected", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return result


def kalman_filter(mu_prev, sigma_prev, measurement):
    """측정값이 없으면 예측만 하고, 있으면 예측값을 측정값으로 보정한다."""
    mu_predicted = A_t @ mu_prev
    sigma_predicted = A_t @ sigma_prev @ A_t.T + PROCESS_NOISE
    if measurement is None:
        return mu_predicted, sigma_predicted, mu_predicted

    innovation = measurement - C_t @ mu_predicted
    innovation_covariance = C_t @ sigma_predicted @ C_t.T + MEASUREMENT_NOISE
    kalman_gain = sigma_predicted @ C_t.T @ np.linalg.inv(innovation_covariance)
    mu_corrected = mu_predicted + kalman_gain @ innovation
    sigma_corrected = (np.eye(4) - kalman_gain @ C_t) @ sigma_predicted
    return mu_predicted, sigma_corrected, mu_corrected

def run_camera():
    """카메라 프레임을 읽어 노란색 원을 검출하고 추적한다."""
    # GStreamer 파이프라인으로 카메라 연결
    capture = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise RuntimeError("MJPG 카메라를 열 수 없습니다.")

    mu_t = np.zeros((4, 1), dtype=float)
    sigma_t = np.eye(4, dtype=float) * 1000
    initialized = False

    try:
        while True:
            # 카메라에서 한 장의 프레임을 읽음
            success, frame = capture.read()
            if not success:
                break

            mask = create_yellow_mask(frame)
            contour = find_yellow_circle(mask)
            measurement_data = get_circle_measurement(contour)
            measurement = measurement_data[0] if measurement_data is not None else None

            if not initialized and measurement is not None:
                mu_t[:2] = measurement
                initialized = True

            mu_predicted, sigma_t, mu_t = kalman_filter(mu_t, sigma_t, measurement)
            result = draw_tracking(frame, contour, mu_predicted, mu_t)

            # mask와 검출 결과를 각각 별도 창에 출력
            cv2.imshow(MASK_WINDOW_NAME, mask)
            cv2.imshow(WINDOW_NAME, result)

            # 1ms 동안 키 입력을 확인하고 q를 누르면 반복 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # 정상 종료와 오류 종료 모두 카메라와 창을 정리
        capture.release()
        cv2.destroyAllWindows()


def main():
    # 프로그램의 카메라 실행 시작점
    run_camera()


if __name__ == "__main__":
    main()
