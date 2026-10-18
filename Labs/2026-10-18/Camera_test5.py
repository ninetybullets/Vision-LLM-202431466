"""
MJPG 카메라에서 초록색 탁구공을 실시간으로 검출하고 추적한다.

실행:
    python Camera_test5.py

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

# LAB 색공간에서 초록색으로 인정할 최소/최대 범위
GREEN_LOWER = np.array([30, 60, 90], dtype=np.uint8)
GREEN_UPPER = np.array([230, 115, 180], dtype=np.uint8)
# 작은 노이즈를 제거하고 객체 영역을 정리할 5x5 타원형 커널
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# 이 면적보다 작은 contour는 탁구공이 아닌 노이즈로 간주
MIN_CONTOUR_AREA = 300
# 결과 영상과 mask를 표시할 창 이름
WINDOW_NAME = "Green Ping-Pong Ball Tracking"
MASK_WINDOW_NAME = "Green Mask"


def create_green_mask(frame):
    """BGR 프레임에서 LAB 색공간 기반 초록색 mask를 생성한다."""
    # Gaussian Blur로 카메라 노이즈와 작은 색상 변화를 완화
    blurred_frame = cv2.GaussianBlur(frame, (7, 7), 0)
    # BGR은 조명에 민감하므로 LAB 색공간으로 변환
    lab_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2LAB)
    # 범위 안에 있는 픽셀은 흰색(255), 나머지는 검은색(0)으로 변환
    mask = cv2.inRange(lab_frame, GREEN_LOWER, GREEN_UPPER)

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


def find_ping_pong_ball(mask):
    """mask에서 충분히 큰 초록색 영역의 contour를 반환한다."""
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


def draw_detection(frame, contour, predicted_center=None, predicted_radius=0):
    """검출 결과와 Kalman Filter 예측 결과를 프레임에 표시한다."""
    # 원본 프레임을 보존하면서 결과를 그릴 복사본 생성
    result = frame.copy()

    if contour is None:
        # 검출 대상이 없을 때도 Kalman 예측 위치를 표시
        cv2.putText(
            result,
            "Green ball: predicted",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        if predicted_center is not None:
            cv2.circle(
                result,
                predicted_center,
                max(1, int(predicted_radius)),
                (0, 255, 255),
                2,
            )
            cv2.circle(result, predicted_center, 5, (0, 255, 255), cv2.FILLED)
        return result

    # contour 전체를 포함하는 최소 외접원의 중심과 반지름 계산
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    center = int(center_x), int(center_y)
    radius = max(1, int(radius))

    # 초록색 contour, 파란색 외접원, 빨간색 중심점 표시
    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
    cv2.circle(result, center, radius, (255, 0, 0), 2)
    cv2.circle(result, center, 5, (0, 0, 255), cv2.FILLED)
    if predicted_center is not None:
        # 검출 중심과 구분하기 위해 예측 결과는 노란색으로 표시
        cv2.circle(
            result,
            predicted_center,
            max(1, int(predicted_radius)),
            (0, 255, 255),
            2,
        )
        cv2.circle(result, predicted_center, 5, (0, 255, 255), cv2.FILLED)
    # 중심 좌표와 반지름을 영상 왼쪽 위에 표시
    cv2.putText(
        result,
        f"Center: {center}  Radius: {radius}px",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    return result


def run_camera():
    """카메라 프레임을 읽어 초록색 탁구공을 검출하고 추적한다."""
    # GStreamer 파이프라인으로 카메라 연결
    capture = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise RuntimeError("MJPG 카메라를 열 수 없습니다.")

    # 노트북의 cv2.KalmanFilter 예제와 동일한 상태/측정 벡터 설정
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kalman.measurementMatrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    kalman.processNoiseCov = np.array(
        [
            [1e-2, 0, 0, 0],
            [0, 1e-2, 0, 0],
            [0, 0, 5e-2, 0],
            [0, 0, 0, 5e-2],
        ],
        dtype=np.float32,
    )
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    found = False
    predicted_center = None
    predicted_radius = 0

    try:
        while True:
            # 카메라에서 한 장의 프레임을 읽음
            success, frame = capture.read()
            if not success:
                break

            # 현재 프레임에서 초록색 mask 생성
            mask = create_green_mask(frame)
            # mask에서 탁구공 후보 contour 선택
            contour = find_ping_pong_ball(mask)

            if contour is not None:
                (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
                measured_center = int(center_x), int(center_y)
                predicted_radius = max(1, int(radius))

                # 최초 검출 시 Kalman Filter의 초기 상태를 측정값으로 설정
                if not found:
                    kalman.statePre = np.array(
                        [[center_x], [center_y], [0.0], [0.0]],
                        dtype=np.float32,
                    )
                    kalman.statePost = kalman.statePre.copy()
                    found = True

            if found:
                # 매 프레임 위치를 먼저 예측하고, 검출 성공 시 측정값으로 보정
                predicted_state = kalman.predict()
                if contour is not None:
                    measurement = np.array(
                        [[measured_center[0]], [measured_center[1]]],
                        dtype=np.float32,
                    )
                    predicted_state = kalman.correct(measurement)

                predicted_center = (
                    int(predicted_state[0, 0]),
                    int(predicted_state[1, 0]),
                )

            # 검출 결과와 Kalman 예측 결과를 원본 프레임 위에 그리기
            result = draw_detection(
                frame,
                contour,
                predicted_center,
                predicted_radius,
            )

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
