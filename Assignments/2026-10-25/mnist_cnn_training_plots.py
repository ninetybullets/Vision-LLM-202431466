"""
2026-10-25 과제 (1/6): MNIST CNN을 CPU로 3 epoch 학습한 뒤
Epoch별 Loss/Accuracy 그래프와 학습된 모델의 층별 Feature Map(viridis)을 시각화한다.

실행: 아무 위치에서 `python mnist_cnn_training_plots.py`
데이터: 저장소 루트의 src/datasets/MNIST (없으면 자동 다운로드)
출력: matplotlib 창 + output/ 폴더에 PNG 저장
"""

import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 이 파일(Assignments/2026-10-25/)에서 두 단계 위가 저장소 루트 → 그 아래 src/ 를 참조
SOURCE_DIRECTORY = Path(__file__).resolve().parents[2] / "src"
DATA_ROOT = SOURCE_DIRECTORY / "datasets"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"

# 수업 노트북(03_DL-and-GPU.ipynb, F절)과 동일한 학습 설정
SEED = 42              # 재현성을 위한 난수 고정값
BATCH_SIZE = 64        # 학습 batch 크기
TEST_BATCH_SIZE = 1000 # 평가 batch 크기 (역전파가 없어 크게 잡아도 됨)
EPOCHS = 3             # 전체 학습 데이터를 반복해서 보는 횟수
LEARNING_RATE = 0.001  # Adam 기본 학습률


class SimpleCNN(nn.Module):
    """
    28x28 흑백 손글씨를 10개 숫자로 분류하는 CNN (수업 노트북과 동일 구조).

    Conv(1→8) → ReLU → MaxPool → Conv(8→16) → ReLU → MaxPool → FC(784→64) → FC(64→10)
    return_features=True이면 중간 층 출력(Feature Map)도 함께 돌려준다.
    """

    def __init__(self):
        super().__init__()
        # padding=1이면 3x3 커널을 지나도 크기가 28x28로 유지됨
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # 2x2 최대값 풀링 → 크기 절반 (28→14→7)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 마지막 Feature Map 16채널 x 7 x 7 을 한 줄로 펼쳐 입력
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x, return_features=False):
        conv1_output = self.conv1(x)
        relu1_output = F.relu(conv1_output)
        pool1_output = self.pool(relu1_output)
        conv2_output = self.conv2(pool1_output)
        relu2_output = F.relu(conv2_output)
        pool2_output = self.pool(relu2_output)
        flattened = torch.flatten(pool2_output, start_dim=1)
        fc1_output = F.relu(self.fc1(flattened))
        logits = self.fc2(fc1_output)

        if return_features:
            features = {
                "conv1": conv1_output,
                "relu1": relu1_output,
                "pool1": pool1_output,
                "conv2": conv2_output,
                "relu2": relu2_output,
                "pool2": pool2_output,
            }
            return logits, features

        return logits


def load_mnist_loaders():
    """
    MNIST 학습/테스트 데이터셋을 읽어 DataLoader 두 개를 만든다.

    반환:
        train_loader: 매 epoch 순서를 섞는 학습용 loader
        test_loader: 순서 고정 평가용 loader
        test_dataset: Feature Map 시각화용 이미지 한 장을 꺼내기 위해 함께 반환
    """
    # ToTensor: 0~255 uint8 이미지를 0.0~1.0 float Tensor [채널, 높이, 너비]로 변환
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader, test_dataset


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """
    학습 데이터를 한 바퀴 돌며 파라미터를 갱신한다 (수업 노트북과 동일).

    반환: (평균 loss, 정확도)
    """
    model.train()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 이전 batch의 gradient가 더해지지 않도록 초기화
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()   # 역전파로 gradient 계산
        optimizer.step()  # gradient 방향으로 파라미터 수정

        batch_size = images.size(0)
        # loss.item()은 batch 평균이므로 batch 크기를 곱해 합으로 누적
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct_count += (predictions == labels).sum().item()
        sample_count += batch_size

    return total_loss / sample_count, correct_count / sample_count


def evaluate(model, data_loader, criterion, device):
    """
    테스트 데이터로 loss와 정확도만 계산한다 (파라미터 수정 없음).

    반환: (평균 loss, 정확도)
    """
    model.eval()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # gradient 계산을 끄면 메모리·시간 절약
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            correct_count += (predictions == labels).sum().item()
            sample_count += batch_size

    return total_loss / sample_count, correct_count / sample_count


def train_model(model, train_loader, test_loader, device):
    """
    EPOCHS만큼 학습·평가를 반복하고 epoch별 기록을 모은다.

    반환: history 딕셔너리 (train_loss, train_accuracy, test_loss, test_accuracy 리스트)
    """
    criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류 손실 (logit을 직접 받음)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        epoch_time = time.perf_counter() - epoch_start_time
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.perf_counter() - training_start_time
    print(f"Total CPU training time: {total_training_time:.2f}s")

    return history


def plot_loss_history(history):
    """Epoch별 학습/테스트 Loss를 한 그래프에 그린다 (노트북 TODO 243)."""
    # x축은 0이 아니라 1번째 epoch부터 시작
    epoch_axis = range(1, EPOCHS + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_axis, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epoch_axis, history["test_loss"], marker="o", label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss")
    plt.grid()
    plt.legend()
    plt.savefig(OUTPUT_DIRECTORY / "cnn_loss.png")


def plot_accuracy_history(history):
    """Epoch별 학습/테스트 Accuracy를 백분율로 그린다 (노트북 TODO 245)."""
    epoch_axis = range(1, EPOCHS + 1)

    plt.figure(figsize=(8, 5))
    # 0~1 비율을 리스트째로 100배 하려면 NumPy 배열로 바꿔야 함
    plt.plot(epoch_axis, np.array(history["train_accuracy"]) * 100, marker="o", label="Train accuracy")
    plt.plot(epoch_axis, np.array(history["test_accuracy"]) * 100, marker="o", label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CNN Accuracy")
    plt.grid()
    plt.legend()
    plt.savefig(OUTPUT_DIRECTORY / "cnn_accuracy.png")


def show_feature_maps(feature_tensor, title, max_maps=8, cmap="gray"):
    """
    [Batch, Channel, Height, Width] Tensor의 첫 번째 이미지 채널들을 격자로 그린다 (수업 노트북 함수).

    입력:
        feature_tensor: 모델이 돌려준 중간 층 출력
        title: 그림 제목
        max_maps: 보여줄 최대 채널 수
        cmap: matplotlib 컬러맵 이름
    """
    if feature_tensor.ndim != 4:
        raise ValueError("Feature Tensor는 [Batch, Channel, Height, Width] 형태여야 합니다.")

    feature_maps = feature_tensor[0].detach().cpu()

    channel_count = min(feature_maps.shape[0], max_maps)
    column_count = 4
    row_count = int(np.ceil(channel_count / column_count))

    plt.figure(figsize=(12, 3 * row_count))

    for channel_index in range(channel_count):
        plt.subplot(row_count, column_count, channel_index + 1)
        plt.imshow(feature_maps[channel_index], cmap=cmap)
        plt.title(f"Channel {channel_index}")
        plt.axis("off")
    plt.suptitle(title)


def visualize_trained_feature_maps(model, test_dataset, device):
    """
    학습된 모델에 테스트 이미지 한 장을 넣어 층별 Feature Map을 viridis로 시각화한다 (노트북 TODO 248).

    1) Conv1 8개, 2) Conv2 16개, 3) 6개 층 x 대표 채널 4개 비교 그림을 만든다.
    """
    # 이미지 선택: 테스트셋 첫 번째 이미지. 모델 입력은 batch 차원이 필요하므로 [1,28,28] → [1,1,28,28]
    image, true_label = test_dataset[0]
    input_batch = image.unsqueeze(0).to(device)

    # 역전파(파라미터 수정) 없이 순전파만 수행
    model.eval()
    with torch.no_grad():
        _, features = model(input_batch, return_features=True)

    show_feature_maps(features["conv1"], title="Trained Conv1 Feature Maps", max_maps=8, cmap="viridis")
    plt.savefig(OUTPUT_DIRECTORY / "feature_maps_conv1.png")

    show_feature_maps(features["conv2"], title="Trained Conv2 Feature Maps", max_maps=16, cmap="viridis")
    plt.savefig(OUTPUT_DIRECTORY / "feature_maps_conv2.png")

    # 6개 층별 대표 Feature Map 4개씩 비교: 행 = 층, 열 = 앞쪽 채널 4개
    layer_names = ["conv1", "relu1", "pool1", "conv2", "relu2", "pool2"]
    representative_count = 4

    plt.figure(figsize=(10, 3 * len(layer_names)))

    for layer_index, layer_name in enumerate(layer_names):
        feature_maps = features[layer_name][0].detach().cpu()

        for channel_index in range(representative_count):
            subplot_index = layer_index * representative_count + channel_index + 1
            plt.subplot(len(layer_names), representative_count, subplot_index)
            plt.imshow(feature_maps[channel_index], cmap="viridis")
            plt.title(f"{layer_name} / Channel {channel_index}")
            plt.axis("off")

    plt.suptitle(f"Layer-wise Feature Maps (label: {true_label})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRECTORY / "feature_maps_by_layer.png")


def main():
    """난수 고정 → 데이터 로드 → CPU 학습 → Loss/Accuracy 그래프 → Feature Map 시각화."""
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_loader, test_loader, test_dataset = load_mnist_loaders()

    # 수업 F절은 CPU 학습 (GPU 비교는 H절 범위)
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    print("Device:", device)

    history = train_model(model, train_loader, test_loader, device)

    plot_loss_history(history)
    plot_accuracy_history(history)
    visualize_trained_feature_maps(model, test_dataset, device)

    plt.show()


if __name__ == "__main__":
    main()
