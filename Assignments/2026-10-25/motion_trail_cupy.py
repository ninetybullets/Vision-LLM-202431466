"""
2026-10-25 과제 (3/5): CuPy(GPU)로 이미지 잔상 효과를 구현하고 NumPy(CPU) 버전과 처리 시간을 비교한다.

이동 거리(shift) 반복은 Python에서 순차 실행하지만, 각 shift마다 이미지 전체의 곱셈·덧셈은 GPU가 병렬 처리한다.
CPU 버전·이미지 로드·가중치는 같은 폴더의 motion_trail_numpy.py에서 가져온다.

실행: Jetson(CUDA + CuPy 설치 환경)에서 `python motion_trail_cupy.py`
※ 맥에는 CUDA가 없어 실행 불가. 코드만 작성하고 cupy→numpy 치환으로 로직만 검증함.
"""

import time

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from motion_trail_numpy import OUTPUT_DIRECTORY, load_input_image, make_weights, motion_trail_np


def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다 (노트북 TODO 305).

    입력:
        image: GPU 메모리의 (높이, 너비) float32 배열
        weights: GPU 메모리의 (잔상 길이,) float32 배열
    반환: image와 같은 모양의 GPU 배열
    """
    image_width = image.shape[1]
    # 출력 배열도 GPU 메모리에 0으로 생성
    output = cp.zeros_like(image)

    for shift in range(len(weights)):
        # 원본의 [0, W - shift) 열이 출력의 [shift, W) 열로 이동. 두 slice의 너비는 같다.
        input_region = image[:, :image_width - shift]
        output_region = output[:, shift:]

        # output_region은 output의 view이므로 += 결과가 output에 바로 쌓인다.
        # 이 한 줄(720 x (1280 - shift) 픽셀 곱셈·덧셈)을 GPU 코어들이 병렬로 처리한다.
        output_region += input_region * weights[shift]

    return output


def main():
    """CPU 처리 → GPU 변환-워밍업-처리-복사 시간 측정 → 비교 출력 → 이미지 출력."""
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    input_np = load_input_image()
    weights_np = make_weights()

    # ---------- NumPy CPU 처리 ----------
    cpu_start = time.perf_counter()
    output_np = motion_trail_np(image=input_np, weights=weights_np)
    cpu_time = time.perf_counter() - cpu_start

    # ---------- CuPy 배열 준비: CPU → GPU 복사 (노트북 TODO 311) ----------
    upload_start = time.perf_counter()
    input_cp = cp.asarray(input_np)
    weights_cp = cp.asarray(weights_np)
    # GPU 작업은 비동기이므로 복사가 끝날 때까지 기다린 뒤 시간을 잰다
    cp.cuda.Stream.null.synchronize()
    upload_time = time.perf_counter() - upload_start

    # ---------- GPU 워밍업 ----------
    # 첫 호출에는 CUDA 커널 준비·메모리 할당 시간이 섞이므로 작은 입력으로 미리 한 번 실행
    warmup_image = cp.zeros((64, 64), dtype=cp.float32)
    warmup_weights = cp.asarray(weights_np[:4])
    _ = motion_trail_cp(image=warmup_image, weights=warmup_weights)
    cp.cuda.Stream.null.synchronize()

    # ---------- CuPy GPU 처리 ----------
    cp.cuda.Stream.null.synchronize()
    gpu_start = time.perf_counter()
    output_cp = motion_trail_cp(image=input_cp, weights=weights_cp)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - gpu_start

    # ---------- GPU 결과를 CPU로 복사 (노트북 TODO 317) ----------
    download_start = time.perf_counter()
    # asnumpy는 복사가 끝나야 반환하므로 별도 synchronize 불필요
    output_cp_np = cp.asnumpy(output_cp)
    download_time = time.perf_counter() - download_start

    # ---------- 결과 비교 ----------
    compute_speedup = cpu_time / gpu_time
    gpu_total_time = upload_time + gpu_time + download_time
    total_speedup = cpu_time / gpu_total_time
    # 같은 계산을 순서만 다르게 했으므로 float32 오차 안에서 같아야 함
    max_difference = np.max(np.abs(output_np - output_cp_np))

    print("========== NumPy CPU ==========")
    print(f"Processing time: {cpu_time:.4f} seconds")
    print()
    print("========== CuPy GPU ==========")
    print(f"CPU to GPU: {upload_time:.4f} seconds")
    print(f"GPU processing: {gpu_time:.4f} seconds")
    print(f"GPU to CPU: {download_time:.4f} seconds")
    print(f"Total GPU time: {gpu_total_time:.4f} seconds")
    print()
    print("========== Comparison ==========")
    print(f"Compute-only speedup: {compute_speedup:.2f}x")
    print(f"Including transfers: {total_speedup:.2f}x")
    print(f"Max difference (CPU vs GPU): {max_difference:.8f}")

    # ---------- 이미지 출력 ----------
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
    plt.subplot(1, 3, 2), plt.imshow(output_np, cmap="gray"), plt.title("Motion Trail (CPU)")
    plt.subplot(1, 3, 3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")
    for axis in plt.gcf().axes:
        axis.axis("off")
    plt.savefig(OUTPUT_DIRECTORY / "motion_trail_cpu_vs_gpu.png")
    plt.show()


if __name__ == "__main__":
    main()
