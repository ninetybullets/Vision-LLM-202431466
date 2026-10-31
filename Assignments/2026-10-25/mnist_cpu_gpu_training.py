"""
2026-10-25 과제 (4/6): MNIST CNN을 CPU와 GPU에서 각각 3 epoch 학습하고 학습 시간을 비교한다
(노트북 03_DL-and-GPU.ipynb I절 TODO `main_gpu()`).

실행: 아무 위치에서 `python mnist_cpu_gpu_training.py`
  - 기본: CPU 학습 → CUDA GPU 학습 → 시간 비교. CUDA가 없으면 GPU 단계에서 오류로 종료(노트북과 동일).
  - 옵션: `--gpu-device mps` (맥에서 코드 확인용), `--epochs 1` (짧게 돌려보기)
모델(SimpleCNN)·데이터 로더·학습/평가 함수는 같은 폴더의 mnist_cnn_training_plots.py에서 가져온다.
"""

import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn

from mnist_cnn_training_plots import (
    EPOCHS,
    LEARNING_RATE,
    SEED,
    SimpleCNN,
    evaluate,
    load_mnist_loaders,
    train_one_epoch,
)


def synchronize(device):
    """
    GPU에 쌓인 작업이 모두 끝날 때까지 CPU를 기다리게 한다.

    GPU 연산은 비동기로 쌓이기만 하고 바로 끝나지 않으므로, 시간을 재기 전후에 호출하지 않으면
    GPU가 아직 계산 중인데 시간 측정이 끝나 버린다. (CuPy의 cp.cuda.Stream.null.synchronize()에 대응)
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    # CPU는 호출 즉시 연산이 끝나므로 할 일이 없다.


def train_on_device(device, epochs):
    """
    주어진 장치에서 MNIST CNN을 epochs만큼 학습·평가하고 전체 학습 시간을 돌려준다.

    CPU와 GPU가 같은 시드·데이터·모델·optimizer를 쓰므로 장치만 다른 공정한 비교가 된다.
    입력:
        device: torch.device("cpu") 또는 torch.device("cuda")
        epochs: 전체 학습 데이터를 반복하는 횟수
    반환: 전체 학습 시간(초)
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_loader, test_loader, _ = load_mnist_loaders()

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    synchronize(device)
    training_start_time = time.perf_counter()

    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        # GPU 작업이 끝난 시점을 기준으로 epoch 시간을 잰다
        synchronize(device)
        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.perf_counter() - training_start_time
    print(f"Total {device.type.upper()} training time: {total_training_time:.2f}s")

    return total_training_time


def main_cpu(epochs=EPOCHS):
    """CPU 학습 (노트북 I절 2) `main_cpu()`에 해당). 반환: 전체 학습 시간(초)"""
    return train_on_device(torch.device("cpu"), epochs)


def main_gpu(epochs=EPOCHS, device_name="cuda"):
    """
    GPU 학습 (노트북 I절 3) TODO `main_gpu()`). 반환: 전체 학습 시간(초)

    기본은 CUDA. 장치를 쓸 수 없으면 노트북과 같이 RuntimeError로 바로 알린다
    (CPU로 조용히 넘어가면 "GPU 시간"이 아닌 값을 비교하게 되기 때문).
    """
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS 장치를 사용할 수 없습니다.")

    return train_on_device(torch.device(device_name), epochs)


def parse_arguments():
    """명령행 옵션: GPU 장치 이름, epoch 수."""
    parser = argparse.ArgumentParser(description="MNIST CNN CPU vs GPU 학습 시간 비교")
    parser.add_argument(
        "--gpu-device",
        choices=["cuda", "mps"],
        default="cuda",
        help="GPU 학습에 쓸 장치 (기본 cuda = Jetson, mps = 맥 코드 확인용)",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"학습 epoch 수 (기본 {EPOCHS})")
    return parser.parse_args()


def main():
    """CPU 학습 → GPU 학습 → 두 시간을 비교해 출력한다 (노트북 I절 비교 셀)."""
    arguments = parse_arguments()

    print("=== CPU Training ===")
    cpu_training_time = main_cpu(arguments.epochs)

    print(f"\n=== GPU Training ({arguments.gpu_device}) ===")
    gpu_training_time = main_gpu(arguments.epochs, arguments.gpu_device)

    speedup = cpu_training_time / gpu_training_time
    time_difference = cpu_training_time - gpu_training_time

    print("\n=== CPU vs. GPU Comparison ===")
    print(f"CPU training time: {cpu_training_time:.2f}s")
    print(f"GPU training time: {gpu_training_time:.2f}s")
    print(f"Time difference: {time_difference:.2f}s")
    print(f"GPU speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
