"""
2026-10-25 과제 (2/3): 이미지 잔상(motion trail) 효과를 NumPy(CPU)와 CuPy(GPU)로 각각 구현하고 처리 시간을 비교한다.

원본 이미지를 오른쪽으로 0, 1, 2, ... 픽셀 이동한 복사본에 점점 작아지는 가중치를 곱해 모두 더한다.
CPU 버전은 출력 픽셀 하나하나를 Python 반복문으로, GPU 버전은 shift마다 이미지 전체를 slicing으로 한 번에 처리한다.

실행: 아무 위치에서 `python motion_trail.py`
  - CuPy가 설치된 Jetson: CPU → GPU 순서로 실행하고 시간 비교표 출력
  - CuPy가 없는 환경(맥 등): CPU만 실행하고 GPU 비교는 생략
입력 이미지: 저장소 루트의 src/images/seagull.jpg
"""

import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 맥(Apple Silicon)에는 CUDA가 없어 import cupy가 실패한다. 실패해도 CPU 부분은 돌도록 None으로 둔다.
try:
    import cupy as cp
except ImportError:
    cp = None

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


def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다 (노트북 TODO 305).

    shift 반복은 Python이 순차 실행하지만, 각 shift에서 이미지 전체의 곱셈·덧셈은 GPU가 병렬 처리한다.

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


def run_motion_trail_cpu(input_np, weights_np):
    """CPU 잔상 처리 시간을 잰다. 반환: (output_np, cpu_time)"""
    cpu_start = time.perf_counter()
    output_np = motion_trail_np(image=input_np, weights=weights_np)
    cpu_time = time.perf_counter() - cpu_start
    return output_np, cpu_time


def run_motion_trail_gpu(input_np, weights_np):
    """
    변환(업로드) → 워밍업 → 처리 → 복사(다운로드) 순서로 GPU 잔상 처리를 하고 단계별 시간을 잰다.

    반환: (output_cp_np, upload_time, gpu_time, download_time)  — 결과는 CPU(NumPy) 배열로 돌려줌
    """
    # ---------- CPU → GPU 복사 (노트북 TODO 311) ----------
    upload_start = time.perf_counter()
    input_cp = cp.asarray(input_np)
    weights_cp = cp.asarray(weights_np)
    # GPU 작업은 비동기이므로 복사가 끝날 때까지 기다린 뒤 시간을 잰다
    cp.cuda.Stream.null.synchronize()
    upload_time = time.perf_counter() - upload_start

    # ---------- 워밍업 ----------
    # 첫 호출에는 CUDA 커널 준비·메모리 할당 시간이 섞이므로 작은 입력으로 미리 한 번 실행
    warmup_image = cp.zeros((64, 64), dtype=cp.float32)
    warmup_weights = cp.asarray(weights_np[:4])
    _ = motion_trail_cp(image=warmup_image, weights=warmup_weights)
    cp.cuda.Stream.null.synchronize()

    # ---------- GPU 처리 ----------
    gpu_start = time.perf_counter()
    output_cp = motion_trail_cp(image=input_cp, weights=weights_cp)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - gpu_start

    # ---------- GPU → CPU 복사 (노트북 TODO 317) ----------
    download_start = time.perf_counter()
    # asnumpy는 복사가 끝나야 반환하므로 별도 synchronize 불필요
    output_cp_np = cp.asnumpy(output_cp)
    download_time = time.perf_counter() - download_start

    return output_cp_np, upload_time, gpu_time, download_time


def print_comparison(cpu_time, upload_time, gpu_time, download_time, max_difference):
    """CPU·GPU 단계별 시간과 속도 비를 표로 출력한다."""
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


def main():
    """이미지 로드 → CPU 잔상(시간) → (CuPy 있으면) GPU 잔상(시간) → 비교표 → 이미지 출력."""
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    input_np = load_input_image()
    weights_np = make_weights()
    print(f"Image size: {input_np.shape[1]}x{input_np.shape[0]}, trail length: {len(weights_np)}")

    output_np, cpu_time = run_motion_trail_cpu(input_np, weights_np)
    print(f"NumPy CPU processing time: {cpu_time:.4f} seconds")

    if cp is None:
        print("CuPy 없음 (CUDA 미지원 환경) → GPU 처리·시간 비교 생략")
        images = [(input_np, "Original Image"), (output_np, "Motion Trail (CPU)")]
    else:
        output_cp_np, upload_time, gpu_time, download_time = run_motion_trail_gpu(input_np, weights_np)
        # 같은 계산을 순서만 다르게 했으므로 float32 오차 안에서 같아야 함
        max_difference = np.max(np.abs(output_np - output_cp_np))
        print_comparison(cpu_time, upload_time, gpu_time, download_time, max_difference)
        images = [(input_np, "Original Image"), (output_np, "Motion Trail (CPU)"), (output_cp_np, "Motion Trail (GPU)")]

    plt.figure(figsize=(6 * len(images), 4))
    for position, (image, title) in enumerate(images, start=1):
        plt.subplot(1, len(images), position), plt.imshow(image, cmap="gray"), plt.title(title)
    for axis in plt.gcf().axes:
        axis.axis("off")
    plt.savefig(OUTPUT_DIRECTORY / "motion_trail.png")
    plt.show()


if __name__ == "__main__":
    main()
