"""
2026-10-25 과제 (4/5): NumPy(CPU)로 2차원 convolution을 수행해 세로/가로 edge Feature Map을 만들고 처리 시간을 잰다.

convolution2d()는 수업 노트북(E절)에서 정의한 함수를 그대로 옮긴 것으로,
출력 픽셀 하나마다 커널 크기의 영역을 잘라 가중합을 구한다 (GPU 버전과 비교용).

실행: 아무 위치에서 `python convolution_numpy.py`
입력 이미지: 저장소 루트의 src/images/seagull.jpg (motion_trail_numpy.py와 같은 1280x720 흑백)
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from motion_trail_numpy import OUTPUT_DIRECTORY, load_input_image

# 수업 노트북(E절)과 동일한 edge 검출 필터. 왼쪽/위를 빼고 오른쪽/아래를 더해 밝기 변화(기울기)를 구한다.
VERTICAL_EDGE_FILTER = np.array(    # x축 방향 변화 → 세로 선 검출
    [[-1, 0, 1],
     [-1, 0, 1],
     [-1, 0, 1]],
    dtype=np.float32,
)
HORIZONTAL_EDGE_FILTER = np.array(  # y축 방향 변화 → 가로 선 검출
    [[-1, -1, -1],
     [0, 0, 0],
     [1, 1, 1]],
    dtype=np.float32,
)
# 편향 -0.2: 가중합이 0.2 이하인 약한 edge는 ReLU를 지나며 0이 되어 무시됨 (수업 설정값)
VERTICAL_EDGE_BIAS = -0.2
HORIZONTAL_EDGE_BIAS = -0.2


def relu(x):
    """0보다 작은 값을 0으로 만드는 활성화 함수."""
    return np.maximum(0, x)


def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """
    2차원 이미지에 하나의 CNN 필터를 적용한다 (수업 노트북 함수).

    입력:
        image: (높이, 너비) 입력 이미지
        kernel: (커널 높이, 커널 너비) 필터
        bias: 가중합에 더할 편향
        stride: 커널이 한 번에 이동하는 픽셀 수
        padding: 가장자리에 덧댈 0 픽셀 수
    반환: 편향까지 적용된 convolution 결과 (float32)
    """
    kernel_height, kernel_width = kernel.shape

    # 가장자리에 zero padding
    padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode="constant", constant_values=0)

    # 출력 크기 = floor((H + 2P - K) / S) + 1
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = np.zeros((output_height, output_width), dtype=np.float32)

    # 출력 픽셀 하나씩: 커널 크기만큼 잘라 원소별 곱의 합을 구한다
    for output_y in range(output_height):
        for output_x in range(output_width):
            start_y = output_y * stride
            start_x = output_x * stride
            image_region = padded_image[start_y:start_y + kernel_height, start_x:start_x + kernel_width]
            output[output_y, output_x] = np.sum(image_region * kernel) + bias

    return output


def run_edge_convolution_cpu(input_np):
    """
    세로/가로 edge 필터를 CPU convolution으로 적용하고 시간을 잰다 (노트북 TODO 332).

    반환: (relu_vertical, relu_horizontal, cpu_time)
    """
    cpu_start = time.perf_counter()
    # padding=1이면 3x3 커널을 지나도 출력 크기가 입력과 같다
    output_vertical = convolution2d(input_np, VERTICAL_EDGE_FILTER, bias=VERTICAL_EDGE_BIAS, stride=1, padding=1)
    output_horizontal = convolution2d(input_np, HORIZONTAL_EDGE_FILTER, bias=HORIZONTAL_EDGE_BIAS, stride=1, padding=1)
    cpu_time = time.perf_counter() - cpu_start

    # 활성화 함수(ReLU)로 음수(반대 방향 edge·약한 특징) 제거
    return relu(output_vertical), relu(output_horizontal), cpu_time


def main():
    """이미지 로드 → CPU convolution(시간 측정) → Feature Map 출력."""
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    input_np = load_input_image()
    relu_vertical_np, relu_horizontal_np, cpu_time = run_edge_convolution_cpu(input_np)

    print("========== NumPy CPU ==========")
    print(f"Image size: {input_np.shape[1]}x{input_np.shape[0]}, kernel: 3x3 x 2 filters")
    print(f"Processing time: {cpu_time:.4f} seconds")

    plt.figure(figsize=(18, 4))
    plt.subplot(1, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
    plt.subplot(1, 3, 2), plt.imshow(relu_vertical_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
    plt.subplot(1, 3, 3), plt.imshow(relu_horizontal_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
    for axis in plt.gcf().axes:
        axis.axis("off")
    plt.savefig(OUTPUT_DIRECTORY / "convolution_cpu.png")
    plt.show()


if __name__ == "__main__":
    main()
