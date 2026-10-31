"""
2026-10-25 과제 (3/3): 2차원 convolution을 NumPy(CPU)와 CuPy(GPU)로 각각 구현해 세로/가로 edge Feature Map을 만들고
결과 일치 여부와 처리 시간을 비교한다.

CPU 버전은 출력 픽셀을 하나씩 돌지만, GPU 버전은 커널 원소(3x3이면 9개)를 하나씩 고르고
그 원소와 곱해질 입력 픽셀 전체를 slicing으로 한 번에 선택해 출력 배열 전체에 누적한다.

실행: 아무 위치에서 `python convolution.py`
  - CuPy가 설치된 Jetson: CPU → GPU 순서로 실행하고 결과·시간 비교표 출력
  - CuPy가 없는 환경(맥 등): CPU만 실행하고 GPU 비교는 생략
입력 이미지: motion_trail.py와 같은 1280x720 흑백 (src/images/seagull.jpg)
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from motion_trail import OUTPUT_DIRECTORY, load_input_image

# 맥(Apple Silicon)에는 CUDA가 없어 import cupy가 실패한다. 실패해도 CPU 부분은 돌도록 None으로 둔다.
try:
    import cupy as cp
except ImportError:
    cp = None

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
    """0보다 작은 값을 0으로 만드는 활성화 함수 (NumPy·CuPy 배열 모두 동작)."""
    return x * (x > 0)


def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """
    2차원 이미지에 하나의 CNN 필터를 적용한다 (수업 노트북 함수, CPU).

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


def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution (노트북 TODO 330). convolution2d()와 같은 결과를 낸다.

    커널 원소 위치는 Python 반복문으로 순차 처리하고,
    각 원소에 대응하는 출력 배열 전체 연산은 GPU에서 처리한다.

    입력:
        image: GPU 메모리의 (높이, 너비) float32 배열
        kernel: GPU 메모리의 (커널 높이, 커널 너비) 배열
        bias, stride, padding: convolution2d()와 동일
    반환: GPU 메모리의 convolution 결과 (float32)
    """
    kernel_height, kernel_width = kernel.shape

    # zero padding도 GPU에서 수행
    padded_image = cp.pad(image, ((padding, padding), (padding, padding)), mode="constant", constant_values=0)

    # 출력 크기 = floor((H + 2P - K) / S) + 1 (CPU 버전과 동일)
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    # 커널 원소 (kernel_y, kernel_x)를 하나씩 선택 (3x3이면 9번 반복)
    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            # 출력 (y, x)가 이 커널 원소와 곱할 입력 픽셀은 padded[y*stride + kernel_y, x*stride + kernel_x].
            # 시작점을 커널 위치만큼 옮기고 stride 간격으로 slicing하면 출력 크기만큼 한 번에 선택된다.
            input_region = padded_image[
                kernel_y:kernel_y + stride * (output_height - 1) + 1:stride,
                kernel_x:kernel_x + stride * (output_width - 1) + 1:stride,
            ]
            # 출력 배열 전체에 "입력 픽셀 x 커널 가중치"를 누적 (GPU 병렬)
            output += input_region * kernel[kernel_y, kernel_x]

    return output + bias


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


def run_edge_convolution_gpu(input_np):
    """
    변환 → 워밍업 → 처리 파이프라인으로 세로/가로 edge convolution을 GPU에서 수행한다 (노트북 TODO 334).

    반환: (relu_vertical_np, relu_horizontal_np, upload_time, gpu_time, download_time) — 결과는 CPU(NumPy)로 복사해 돌려줌
    """
    # 1) 변환: 입력 이미지와 필터를 GPU 메모리로 복사
    upload_start = time.perf_counter()
    input_cp = cp.asarray(input_np)
    vertical_filter_cp = cp.asarray(VERTICAL_EDGE_FILTER)
    horizontal_filter_cp = cp.asarray(HORIZONTAL_EDGE_FILTER)
    cp.cuda.Stream.null.synchronize()
    upload_time = time.perf_counter() - upload_start

    # 2) 워밍업: 첫 실행의 CUDA 커널 준비·메모리 할당 시간을 측정에서 제외
    warmup_image = cp.zeros((64, 64), dtype=cp.float32)
    _ = convolution2d_cp(warmup_image, vertical_filter_cp, bias=VERTICAL_EDGE_BIAS, stride=1, padding=1)
    cp.cuda.Stream.null.synchronize()

    # 3) 처리: GPU 작업이 끝날 때까지 synchronize로 기다린 뒤 시간 측정
    gpu_start = time.perf_counter()
    output_vertical_cp = convolution2d_cp(input_cp, vertical_filter_cp, bias=VERTICAL_EDGE_BIAS, stride=1, padding=1)
    output_horizontal_cp = convolution2d_cp(input_cp, horizontal_filter_cp, bias=HORIZONTAL_EDGE_BIAS, stride=1, padding=1)
    relu_vertical_cp = relu(output_vertical_cp)
    relu_horizontal_cp = relu(output_horizontal_cp)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - gpu_start

    # 4) 복사: 시각화·비교를 위해 CPU(NumPy)로 가져옴
    download_start = time.perf_counter()
    relu_vertical_np = cp.asnumpy(relu_vertical_cp)
    relu_horizontal_np = cp.asnumpy(relu_horizontal_cp)
    download_time = time.perf_counter() - download_start

    return relu_vertical_np, relu_horizontal_np, upload_time, gpu_time, download_time


def print_comparison(cpu_time, upload_time, gpu_time, download_time, max_difference, is_same_result):
    """CPU·GPU 단계별 시간, 속도 비, 결과 일치 여부를 표로 출력한다 (노트북 TODO 336)."""
    gpu_total_time = upload_time + gpu_time + download_time

    print()
    print("========== CPU vs GPU 시간 비교 ==========")
    print(f"{'단계':<28}{'시간(초)':>12}")
    print("-" * 40)
    print(f"{'NumPy CPU 처리':<28}{cpu_time:>12.4f}")
    print(f"{'CuPy  CPU→GPU 복사':<28}{upload_time:>12.4f}")
    print(f"{'CuPy  GPU 처리':<28}{gpu_time:>12.4f}")
    print(f"{'CuPy  GPU→CPU 복사':<28}{download_time:>12.4f}")
    print(f"{'CuPy  합계(복사 포함)':<28}{gpu_total_time:>12.4f}")
    print("-" * 40)
    print(f"속도 비 (처리만):      {cpu_time / gpu_time:.2f}x")
    print(f"속도 비 (복사 포함):   {cpu_time / gpu_total_time:.2f}x")
    print(f"결과 최대 차이 (CPU vs GPU): {max_difference:.8f}")
    print(f"결과 일치 (allclose): {is_same_result}")


def main():
    """이미지 로드 → CPU convolution(시간) → (CuPy 있으면) GPU convolution(시간) → 비교표 → 이미지 출력."""
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    input_np = load_input_image()
    print(f"Image size: {input_np.shape[1]}x{input_np.shape[0]}, kernel: 3x3 x 2 filters")

    relu_vertical_np, relu_horizontal_np, cpu_time = run_edge_convolution_cpu(input_np)
    print(f"NumPy CPU processing time: {cpu_time:.4f} seconds")

    if cp is None:
        print("CuPy 없음 (CUDA 미지원 환경) → GPU 처리·시간 비교 생략")
        rows = [
            [(input_np, "Input Image"), (relu_vertical_np, "Vertical Convolution (CPU)")],
            [(input_np, "Input Image"), (relu_horizontal_np, "Horizontal Convolution (CPU)")],
        ]
    else:
        relu_vertical_cp_np, relu_horizontal_cp_np, upload_time, gpu_time, download_time = run_edge_convolution_gpu(input_np)
        # 계산 순서만 다르고 같은 가중합이므로 float32 오차 범위 안에서 같아야 함
        max_difference = max(
            np.max(np.abs(relu_vertical_np - relu_vertical_cp_np)),
            np.max(np.abs(relu_horizontal_np - relu_horizontal_cp_np)),
        )
        is_same_result = (
            np.allclose(relu_vertical_np, relu_vertical_cp_np, atol=1e-5)
            and np.allclose(relu_horizontal_np, relu_horizontal_cp_np, atol=1e-5)
        )
        print_comparison(cpu_time, upload_time, gpu_time, download_time, max_difference, is_same_result)
        rows = [
            [(input_np, "Input Image"), (relu_vertical_np, "Vertical (CPU)"), (relu_vertical_cp_np, "Vertical (GPU)")],
            [(input_np, "Input Image"), (relu_horizontal_np, "Horizontal (CPU)"), (relu_horizontal_cp_np, "Horizontal (GPU)")],
        ]

    columns = len(rows[0])
    plt.figure(figsize=(6 * columns, 8))
    for row_index, row in enumerate(rows):
        for column_index, (image, title) in enumerate(row):
            plt.subplot(2, columns, row_index * columns + column_index + 1)
            plt.imshow(image, cmap="gray"), plt.title(title)
    for axis in plt.gcf().axes:
        axis.axis("off")
    plt.savefig(OUTPUT_DIRECTORY / "convolution.png")
    plt.show()


if __name__ == "__main__":
    main()
