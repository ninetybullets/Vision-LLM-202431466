"""
2026-10-25 과제 (2/5): NumPy(CPU)로 이미지 잔상(motion trail) 효과를 구현하고 처리 시간을 잰다.

원본 이미지를 오른쪽으로 0, 1, 2, ... 픽셀 이동한 복사본에 점점 작아지는 가중치를 곱해 모두 더한다.
출력 픽셀 하나하나를 Python 반복문으로 계산하므로 일부러 느리다 (GPU 버전과 비교용).

실행: 아무 위치에서 `python motion_trail_numpy.py`
입력 이미지: 저장소 루트의 src/images/seagull.jpg
"""

import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

SOURCE_DIRECTORY = Path(__file__).resolve().parents[2] / "src"
IMAGE_PATH = SOURCE_DIRECTORY / "images" / "seagull.jpg"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"

# 수업 노트북(G절 4) 글로벌 변수와 동일
TARGET_WIDTH = 1280   # 처리 시간 비교를 위해 HD 크기로 통일
TARGET_HEIGHT = 720
TRAIL_LENGTH = 32     # 잔상 길이 = 이동 픽셀 수 = 가중치 개수


def load_input_image():
    """
    이미지를 흑백으로 읽어 1280x720으로 맞추고 0.0~1.0 float32 배열로 돌려준다.

    반환: shape (720, 1280)의 np.float32 배열
    """
    image_gray = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    # imread는 파일이 없어도 None을 돌려주므로 여기서 확인
    if image_gray is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

    # INTER_AREA: 축소할 때 픽셀을 평균 내어 계단 현상이 적음
    image_gray = cv2.resize(image_gray, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    # 가중치(실수)를 곱해 누적하므로 uint8 대신 float32, 0~1 범위로 정규화
    return image_gray.astype(np.float32) / 255.0


def make_weights(trail_length=TRAIL_LENGTH):
    """
    이동 거리가 멀수록 작아지는 가중치(1.0 → 0.05)를 만들고 합이 1이 되게 정규화한다.

    합을 1로 맞춰야 누적 결과가 입력과 같은 0~1 밝기 범위에 머문다.
    반환: shape (trail_length,)의 np.float32 배열
    """
    weights = np.linspace(1.0, 0.05, trail_length, dtype=np.float32)
    return weights / np.sum(weights)


def motion_trail_np(image, weights):
    """
    NumPy를 이용해 이미지 잔상 효과를 생성한다 (노트북 TODO 303).

    출력 픽셀을 Python 반복문으로 하나씩 계산한다.
    출력 (y, x) = Σ_shift weights[shift] * image[y, x - shift]  (x - shift < 0 이면 빈 픽셀 = 0)

    입력:
        image: (높이, 너비) float32 배열, 0~1
        weights: (잔상 길이,) float32 배열
    반환: image와 같은 모양의 잔상 적용 결과
    """
    image_height, image_width = image.shape
    # 누적 시작값은 0. 원본을 복사해 시작하면 shift=0 항이 두 번 더해진다.
    output = np.zeros_like(image)

    for output_y in range(image_height):
        for output_x in range(image_width):
            accumulated_value = 0.0

            # TRAIL_LENGTH 상수 대신 len(weights): 짧은 가중치(워밍업 등)가 들어와도 동작
            for shift in range(len(weights)):
                # 오른쪽으로 shift칸 이동한 이미지의 (y, x)는 원본의 (y, x - shift)
                input_x = output_x - shift

                # 왼쪽 경계 밖은 더할 값이 없음. shift는 커지기만 하므로 이후도 전부 밖 → 중단
                if input_x < 0:
                    break

                accumulated_value += weights[shift] * image[output_y, input_x]

            # 가중치 합이 1이라 0~1을 벗어나지 않으므로 별도 클램프 불필요
            output[output_y, output_x] = accumulated_value

    return output


def main():
    """이미지 로드 → 가중치 생성 → CPU 잔상 처리(시간 측정) → 결과 출력."""
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    input_np = load_input_image()
    weights_np = make_weights()

    cpu_start = time.perf_counter()
    output_np = motion_trail_np(image=input_np, weights=weights_np)
    cpu_time = time.perf_counter() - cpu_start

    print("========== NumPy CPU ==========")
    print(f"Image size: {input_np.shape[1]}x{input_np.shape[0]}, trail length: {len(weights_np)}")
    print(f"Processing time: {cpu_time:.4f} seconds")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
    plt.subplot(1, 2, 2), plt.imshow(output_np, cmap="gray"), plt.title("Motion Trail (CPU)")
    for axis in plt.gcf().axes:
        axis.axis("off")
    plt.savefig(OUTPUT_DIRECTORY / "motion_trail_cpu.png")
    plt.show()


if __name__ == "__main__":
    main()
