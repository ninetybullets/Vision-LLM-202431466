"""
2026-10-25 과제 (5/6): CIFAR-10 분류 CNN을 CUDA GPU에서 5 epoch 학습하고(노트북 03_DL-and-GPU.ipynb J절 TODO),
학습한 모델을 .pth 파일로 저장한 뒤 다시 불러와 테스트 이미지 40장의 예측 결과를 확인한다.

실행: 아무 위치에서 `python cifar10_gpu_training.py` (Jetson, CUDA 필수)
  - 옵션: `--device mps|cpu` (맥에서 코드 확인용), `--epochs N` (짧게 돌려보기)
데이터: 저장소 루트 src/datasets/cifar-10-batches-py (없으면 자동 다운로드, 약 170MB)
출력: src/models/CIFAR10/CIFAR10_CNN.pth (모델 파일), output/cifar10_predictions.png (예측 결과 40장)
학습·평가 함수(train_one_epoch, evaluate)는 같은 폴더의 mnist_cnn_training_plots.py에서 가져온다.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_cnn_training_plots import evaluate, train_one_epoch

# 이 파일(Assignments/2026-10-25/)에서 두 단계 위가 저장소 루트 → 그 아래 src/ 를 참조
SOURCE_DIRECTORY = Path(__file__).resolve().parents[2] / "src"
DATA_ROOT = SOURCE_DIRECTORY / "datasets"
MODEL_PATH = SOURCE_DIRECTORY / "models" / "CIFAR10" / "CIFAR10_CNN.pth"  # 노트북과 같은 저장 위치
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"

# 수업 노트북 J절과 동일한 학습 설정
SEED = 42
BATCH_SIZE = 32         # MNIST(64)보다 이미지가 크고 3채널이라 batch를 줄임
TEST_BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001   # AdamW 학습률

# CIFAR-10 채널별 평균·표준편차 (PyTorch 공식 자료의 미리 계산된 값) → 입력 정규화에 사용
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class ConvBlock(nn.Module):
    """
    CIFAR-10 CNN에서 반복되는 구간을 묶은 블록 (수업 노트북과 동일).

    Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool
    입력 [B, in_channels, H, W] → 출력 [B, out_channels, H/2, W/2]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # BatchNorm이 뒤따르면 Conv의 bias는 정규화 과정에서 상쇄되므로 bias=False
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # 채널별 출력 분포를 정규화해 학습 안정화
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 공간 크기 절반
        )

    def forward(self, x):
        return self.block(x)


class CIFAR10CNN(nn.Module):
    """
    32x32 RGB 이미지를 10개 클래스로 분류하는 CNN (수업 노트북과 동일).

    ConvBlock(3→64) → ConvBlock(64→128) → ConvBlock(128→256) → 전역 평균 풀링 → Dropout → Linear(256→10)
    채널을 64→128→256으로 늘려 단순한 경계·색상부터 복잡한 형태까지 단계별로 특징을 뽑는다.
    """

    def __init__(self):
        super().__init__()
        # 32x32 → 16x16 → 8x8 → 4x4
        self.features = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
        )
        # 4x4 Feature Map을 채널별 평균 하나로 압축 → FC 입력이 256개로 줄어 파라미터 절약
        self.global_average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # 학습 때 특징 30%를 무작위로 꺼서 과적합 완화
            nn.Linear(in_features=256, out_features=10),
        )

    def forward(self, x):
        x = self.features(x)                  # [B, 256, 4, 4]
        x = self.global_average_pool(x)       # [B, 256, 1, 1]
        x = torch.flatten(x, start_dim=1)     # [B, 256]
        return self.classifier(x)             # [B, 10] (클래스별 점수 logit)


def load_cifar10_loaders(pin_memory):
    """
    CIFAR-10 학습/테스트 DataLoader를 만든다.

    학습 데이터에만 RandomCrop·RandomHorizontalFlip 증강을 적용해 과적합을 줄인다.
    입력: pin_memory - True면 batch를 고정 메모리에 올려 CPU→GPU 복사가 빨라짐 (CUDA에서만 의미 있음)
    반환: (train_loader, test_loader, test_dataset)
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),       # 4픽셀 패딩 후 무작위 32x32 잘라내기 (위치 변화)
        transforms.RandomHorizontalFlip(p=0.5),           # 절반 확률로 좌우 반전
        transforms.ToTensor(),                            # [H,W,C] 0~255 → [C,H,W] 0~1
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    # 테스트는 증강 없이 정규화만 (평가 조건을 고정)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)

    return train_loader, test_loader, test_dataset


def synchronize(device):
    """GPU에 쌓인 작업이 끝날 때까지 기다린다. 시간 측정 전후에 호출해야 GPU 시간이 빠지지 않는다."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def warm_up(model, device, repeat_count=3):
    """
    GPU warm-up (노트북 J절 11): 첫 CUDA 호출에 섞이는 Context 생성·cuDNN 초기화·메모리 할당 시간을
    학습 시간 측정에서 빼기 위해, 학습 전에 같은 모양의 입력으로 순전파만 몇 번 미리 실행한다.
    """
    warmup_input = torch.zeros(64, 3, 32, 32, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(repeat_count):
            model(warmup_input)
    synchronize(device)


def train_model(model, train_loader, test_loader, device, epochs):
    """
    CIFAR-10을 epochs만큼 학습·평가하며 epoch별 시간과 전체 학습 시간을 출력한다 (노트북 J절 TODO).

    반환: (history 딕셔너리, optimizer) — optimizer는 체크포인트 저장에 필요
    """
    criterion = nn.CrossEntropyLoss()
    # AdamW: Adam에 weight decay(가중치 감쇠)를 분리해 적용한 변형, 과적합 완화에 유리
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}

    synchronize(device)
    training_start_time = time.perf_counter()

    for epoch in range(epochs):
        synchronize(device)
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        synchronize(device)
        epoch_time = time.perf_counter() - epoch_start_time
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    synchronize(device)
    total_training_time = time.perf_counter() - training_start_time
    print(f"Total {device.type.upper()} training time: {total_training_time:.2f}s")

    return history, optimizer


def save_checkpoint(model, optimizer, history, epochs):
    """
    학습 결과를 .pth로 저장한다 (노트북 J절 14).

    가중치(state_dict)만 저장하지 않고 optimizer 상태·정확도·클래스 이름·정규화 값을 함께 담아
    나중에 이어서 학습하거나 추론할 때 필요한 정보를 한 파일에서 꺼낼 수 있게 한다.
    """
    save_data = {
        "model_name": "CIFAR10CNN",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_accuracy": history["test_accuracy"][-1],
        "epochs": epochs,
        "class_names": CIFAR10_CLASSES,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, MODEL_PATH)

    file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"모델 저장: {MODEL_PATH} ({file_size_mb:.1f} MB)")


def load_checkpoint(device):
    """
    저장한 .pth를 새 모델에 불러온다 (노트북 J절 15). 저장이 제대로 됐는지 확인하는 단계.

    반환: 평가 모드로 전환된 모델
    """
    loaded_model = CIFAR10CNN()
    # map_location="cpu": 저장 장치와 달라도 우선 CPU로 읽고, 그 뒤 원하는 장치로 옮김
    # weights_only=True: 가중치·숫자·문자열만 허용해 pickle로 임의 코드가 실행되는 위험을 막음
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.to(device)
    loaded_model.eval()

    print("모델 이름:", checkpoint["model_name"])
    print("Epoch 체크포인트:", checkpoint["epochs"])
    print(f"마지막 테스트 정확도: {checkpoint['test_accuracy'] * 100:.2f}%")

    return loaded_model


def plot_predictions(model, test_dataset, device, start_index=40, image_count=40):
    """
    테스트 이미지 image_count장을 예측해 4x10 격자로 그린다 (노트북 J절 16). 틀린 예측은 제목을 빨간색으로.

    정규화된 Tensor를 그대로 그리면 색이 깨지므로 (x * std + mean)으로 되돌린 뒤 [H,W,C]로 바꿔 그린다.
    """
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    plt.figure(figsize=(15, 10))
    model.eval()
    correct_count = 0

    with torch.no_grad():
        for grid_index in range(image_count):
            image, true_label = test_dataset[start_index + grid_index]
            input_batch = image.unsqueeze(0).to(device)  # [3,32,32] → [1,3,32,32]

            logits = model(input_batch)
            probabilities = torch.softmax(logits, dim=1)
            predicted_label = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_label].item()

            is_correct = predicted_label == true_label
            correct_count += int(is_correct)

            image_rgb = (image * std + mean).permute(1, 2, 0).clamp(0, 1)

            plt.subplot(4, 10, grid_index + 1)
            plt.imshow(image_rgb)
            plt.title(
                f"True: {CIFAR10_CLASSES[true_label]}\n"
                f"Pred: {CIFAR10_CLASSES[predicted_label]}\n"
                f"Conf: {confidence * 100:.1f}%",
                color="black" if is_correct else "red",
            )
            plt.axis("off")

    plt.suptitle(f"CIFAR-10 predictions ({correct_count}/{image_count} correct)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRECTORY / "cifar10_predictions.png")
    print(f"예측 결과 40장 중 정답 {correct_count}장 → {OUTPUT_DIRECTORY / 'cifar10_predictions.png'}")


def resolve_device(device_name):
    """요청한 장치를 확인해 torch.device로 돌려준다. 쓸 수 없으면 노트북과 같이 RuntimeError."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS 장치를 사용할 수 없습니다.")
    return torch.device(device_name)


def parse_arguments():
    """명령행 옵션: 장치 이름, epoch 수."""
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN GPU 학습 및 모델 저장")
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default="cuda",
        help="학습 장치 (기본 cuda = Jetson, mps/cpu = 맥 코드 확인용)",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"학습 epoch 수 (기본 {EPOCHS})")
    return parser.parse_args()


def main():
    """난수 고정 → 데이터 로드 → 모델 생성 → warm-up → GPU 학습 → .pth 저장 → 다시 로드 → 예측 40장 시각화."""
    arguments = parse_arguments()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = resolve_device(arguments.device)
    print("Device:", device)

    train_loader, test_loader, test_dataset = load_cifar10_loaders(pin_memory=(device.type == "cuda"))

    model = CIFAR10CNN().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Total parameters: {parameter_count:,}")

    warm_up(model, device)
    history, optimizer = train_model(model, train_loader, test_loader, device, arguments.epochs)

    save_checkpoint(model, optimizer, history, arguments.epochs)
    loaded_model = load_checkpoint(device)

    plot_predictions(loaded_model, test_dataset, device)
    plt.show()


if __name__ == "__main__":
    main()
