"""
2026-10-25 과제 (5/5): CuPy(GPU)로 2차원 convolution을 구현하고 NumPy(CPU) 버전과 결과·처리 시간을 비교한다.

CPU 버전은 출력 픽셀을 하나씩 돌지만, GPU 버전은 커널 원소(3x3이면 9개)를 하나씩 고르고
그 원소와 곱해질 입력 픽셀 전체를 slicing으로 한 번에 선택해 출력 배열 전체에 누적한다.
CPU 함수·필터·이미지 로드는 같은 폴더의 convolution_numpy.py / motion_trail_numpy.py에서 가져온다.

실행: Jetson(CUDA + CuPy 설치 환경)에서 `python convolution_cupy.py`
※ 맥에는 CUDA가 없어 실행 불가. 코드만 작성하고 cupy→numpy 치환으로 로직만 검증함.
"""

import time

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from motion_trail_numpy import OUTPUT_DIRECTORY, load_input_image
from convolution_numpy import (
    HORIZONTAL_EDGE_BIAS,
    HORIZONTAL_EDGE_FILTER,
    VERTICAL_EDGE_BIAS,
    VERTICAL_EDGE_FILTER,
    run_edge_convolution_cpu,
)


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


def run_edge_convolution_gpu(input_np):
    """
    변환 → 워밍업 → 처리 파이프라인으로 세로/가로 edge convolution을 GPU에서 수행한다 (노트북 TODO 334).

    반환: (relu_vertical_np, relu_horizontal_np, gpu_time)  — 결과는 CPU(NumPy)로 복사해 돌려줌
    """
    # 1) 변환: 입력 이미지와 필터를 GPU 메모리로 복사
    input_cp = cp.asarray(input_np)
    vertical_filter_cp = cp.asarray(VERTICAL_EDGE_FILTER)
    horizontal_filter_cp = cp.asarray(HORIZONTAL_EDGE_FILTER)

    # 2) 워밍업: 첫 실행의 CUDA 커널 준비·메모리 할당 시간을 측정에서 제외
    warmup_image = cp.zeros((64, 64), dtype=cp.float32)
    _ = convolution2d_cp(warmup_image, vertical_filter_cp, bias=VERTICAL_EDGE_BIAS, stride=1, padding=1)
    cp.cuda.Stream.null.synchronize()

    # 3) 처리: GPU 작업이 끝날 때까지 synchronize로 기다린 뒤 시간 측정
    cp.cuda.Stream.null.synchronize()
    gpu_start = time.perf_counter()
    output_vertical_cp = convolution2d_cp(input_cp, vertical_filter_cp, bias=VERTICAL_EDGE_BIAS, stride=1, padding=1)
    output_horizontal_cp = convolution2d_cp(input_cp, horizontal_filter_cp, bias=HORIZONTAL_EDGE_BIAS, stride=1, padding=1)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - gpu_start

    # ReLU는 GPU에서 적용한 뒤 시각화를 위해 CPU(NumPy)로 복사
    relu_vertical_np = cp.asnumpy(cp.maximum(0, output_vertical_cp))
    relu_horizontal_np = cp.asnumpy(cp.maximum(0, output_horizontal_cp))

    return relu_vertical_np, relu_horizontal_np, gpu_time


def main():
    """이미지 로드 → CPU convolution → GPU convolution → 결과 비교 출력 → 이미지 출력."""
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    input_np = load_input_image()

    relu_vertical_np, relu_horizontal_np, cpu_time = run_edge_convolution_cpu(input_np)
    relu_vertical_cp_np, relu_horizontal_cp_np, gpu_time = run_edge_convolution_gpu(input_np)

    # ---------- CPU/GPU 결과 비교 및 출력 (노트북 TODO 336) ----------
    # 계산 순서만 다르고 같은 가중합이므로 float32 오차 범위 안에서 같아야 함
    max_difference_vertical = np.max(np.abs(relu_vertical_np - relu_vertical_cp_np))
    max_difference_horizontal = np.max(np.abs(relu_horizontal_np - relu_horizontal_cp_np))
    is_same_result = (
        np.allclose(relu_vertical_np, relu_vertical_cp_np, atol=1e-5)
        and np.allclose(relu_horizontal_np, relu_horizontal_cp_np, atol=1e-5)
    )

    print("========== NumPy CPU ==========")
    print(f"Processing time: {cpu_time:.4f} seconds")
    print()
    print("========== CuPy GPU ==========")
    print(f"Processing time: {gpu_time:.4f} seconds")
    print()
    print("========== Comparison ==========")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    print(f"Max difference (vertical): {max_difference_vertical:.8f}")
    print(f"Max difference (horizontal): {max_difference_horizontal:.8f}")
    print(f"Same result: {is_same_result}")

    # ---------- 이미지 출력 ----------
    plt.figure(figsize=(18, 8))
    plt.subplot(2, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
    plt.subplot(2, 3, 2), plt.imshow(relu_vertical_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
    plt.subplot(2, 3, 3), plt.imshow(relu_vertical_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
    plt.subplot(2, 3, 4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
    plt.subplot(2, 3, 5), plt.imshow(relu_horizontal_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
    plt.subplot(2, 3, 6), plt.imshow(relu_horizontal_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")
    for axis in plt.gcf().axes:
        axis.axis("off")
    plt.savefig(OUTPUT_DIRECTORY / "convolution_cpu_vs_gpu.png")
    plt.show()


if __name__ == "__main__":
    main()
