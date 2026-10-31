"""
2026-10-25 과제 (6/6): Bounding Box의 IoU 계산과 NMS(비최대 억제)를 직접 구현한다
(노트북 04_DL-Object-Detection.ipynb B절·C절 TODO).

1) 사과 이미지 3장에서 빨간/초록 사과의 Bounding Box를 구하고 두 상자의 IoU를 비교한다.
2) 배경 이미지 위에 난수 Bounding Box 20개(confidence 포함)를 만들어
   Confidence Thresholding → NMS로 정리하고, OpenCV 내장 cv2.dnn.NMSBoxes() 결과와 비교한다.

실행: 아무 위치에서 `python iou_nms.py` (CPU만 사용, 어디서나 실행 가능)
입력: 저장소 루트 src/images/two_apples1~3.png.
      NMS 배경은 scikit-image의 hubble_deep_field 이미지, 없으면 같은 크기의 검은 배경을 쓴다.
출력: output/iou_apples.png, output/nms_result.png (+ matplotlib 창)
"""

import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 수업 노트북은 scikit-image의 허블 우주 사진을 배경으로 쓴다. 맥 환경엔 없을 수 있어 선택 의존성으로 둔다.
try:
    import skimage.data
except ImportError:
    skimage = None

SOURCE_DIRECTORY = Path(__file__).resolve().parents[2] / "src"
APPLE_IMAGE_PATHS = [SOURCE_DIRECTORY / "images" / f"two_apples{index}.png" for index in (1, 2, 3)]
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"

# 노트북 C절과 동일한 NMS 설정
RANDOM_BOX_COUNT = 20        # 난수로 만들 Bounding Box 개수
CONFIDENCE_THRESHOLD = 0.5   # 이보다 낮은 confidence의 상자는 NMS 전에 제거
IOU_THRESHOLD = 0.2          # 선택된 상자와 IoU가 이 값 이상이면 같은 객체로 보고 제거
RANDOM_SEED = 42             # 난수 상자가 실행마다 같게 나오도록 고정 (노트북은 매번 달라짐)
FALLBACK_BACKGROUND_SHAPE = (872, 1000, 3)  # hubble_deep_field 이미지와 같은 크기 (H, W, C)


# ---------------------------------------------------------------- B절: IoU

def convert_bounding_box_to_corners(bounding_box):
    """
    (x, y, w, h) 형식 Bounding Box를 (x1, y1, x2, y2) 모서리 좌표로 바꾼다 (노트북 TODO `convert_bbox_coord`).

    cv2.boundingRect()는 (x, y, w, h)를 돌려주고, IoU 계산·cv2.rectangle()은 모서리 좌표가 편하다.
    """
    x, y, width, height = bounding_box
    return (x, y, x + width, y + height)


def calculate_iou(bounding_box_a, bounding_box_b):
    """
    두 Bounding Box의 IoU(교집합 넓이 / 합집합 넓이)를 계산한다 (노트북 TODO `calculate_iou`).

    입력: 두 상자 모두 (x1, y1, x2, y2) 모서리 좌표
    반환: 0.0(겹치지 않음) ~ 1.0(완전히 같음)
    """
    xa_1, ya_1, xa_2, ya_2 = bounding_box_a
    xb_1, yb_1, xb_2, yb_2 = bounding_box_b

    # 교집합 상자: 왼쪽 위는 두 상자 중 더 큰 값, 오른쪽 아래는 더 작은 값
    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    # 겹치지 않으면 x2 - x1이 음수가 되므로 0으로 막는다
    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = (xa_2 - xa_1) * (ya_2 - ya_1)
    area_b = (xb_2 - xb_1) * (yb_2 - yb_1)
    # 합집합 = A + B - 교집합 (교집합을 두 번 세지 않도록)
    union_area = area_a + area_b - intersection_area

    # 두 상자가 모두 넓이 0이면 0으로 나누게 되므로 IoU 0 처리
    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def draw_bounding_box(image, top_left, bottom_right, color, thickness=3, text=""):
    """이미지 위에 사각형과 라벨 글씨를 그린다 (수업 노트북 draw_bbox 함수). image를 직접 수정한다."""
    _, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    # 글씨는 상자 위쪽 가장자리에 살짝 겹치게
    cv2.putText(image, text, (top_left[0], top_left[1] - text_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def get_apple_contours(image_bgr):
    """
    HSV 색상 범위로 빨간 사과·초록 사과 영역을 분리해 각각 가장 큰 contour를 돌려준다 (수업 노트북 get_contour 함수).

    빨강은 Hue 원형(0~180)에서 양 끝(0~10, 170~180)에 걸쳐 있어 마스크 두 개를 합친다.
    반환: (red_contour, green_contour) — 못 찾으면 None
    """
    red_lower1, red_upper1 = np.array([0, 100, 50]), np.array([10, 255, 255])
    red_lower2, red_upper2 = np.array([170, 100, 50]), np.array([180, 255, 255])
    green_lower, green_upper = np.array([25, 10, 30]), np.array([100, 255, 255])

    # 블러로 잡음을 줄인 뒤 색상 기준으로 분리
    image_blurred = cv2.GaussianBlur(image_bgr, (7, 7), 0)
    image_hsv = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2HSV)

    red_mask = cv2.bitwise_or(cv2.inRange(image_hsv, red_lower1, red_upper1), cv2.inRange(image_hsv, red_lower2, red_upper2))
    green_mask = cv2.inRange(image_hsv, green_lower, green_upper)

    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_contour = max(red_contours, key=cv2.contourArea) if red_contours else None
    green_contour = max(green_contours, key=cv2.contourArea) if green_contours else None

    return red_contour, green_contour


def compute_apple_iou(image_path):
    """
    사과 이미지 한 장에서 두 사과의 Bounding Box를 구해 IoU를 계산하고 상자를 그린다.

    반환: (상자가 그려진 RGB 이미지, IoU 값)
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    red_contour, green_contour = get_apple_contours(image_bgr)
    if red_contour is None or green_contour is None:
        raise ValueError(f"사과 contour를 찾지 못했습니다: {image_path}")

    # cv2.boundingRect → (x, y, w, h) → 모서리 좌표로 변환해 IoU 계산
    red_box = convert_bounding_box_to_corners(cv2.boundingRect(red_contour))
    green_box = convert_bounding_box_to_corners(cv2.boundingRect(green_contour))
    iou = calculate_iou(red_box, green_box)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    draw_bounding_box(image_rgb, red_box[:2], red_box[2:], (255, 0, 0), text="Red Apple")
    draw_bounding_box(image_rgb, green_box[:2], green_box[2:], (0, 255, 0), text="Green Apple")

    return image_rgb, iou


def run_iou_demo():
    """사과 이미지 3장의 IoU를 계산해 한 줄로 나란히 그린다 (노트북 B절 비교 셀)."""
    plt.figure(figsize=(16, 6))

    for subplot_index, image_path in enumerate(APPLE_IMAGE_PATHS, start=1):
        image_rgb, iou = compute_apple_iou(image_path)
        print(f"{image_path.name}: IoU = {iou:.4f}")

        plt.subplot(1, 3, subplot_index)
        plt.imshow(image_rgb)
        plt.title(f"IoU: {iou:.4f}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIRECTORY / "iou_apples.png")


# ---------------------------------------------------------------- C절: NMS

def load_background_image():
    """NMS 시연용 배경. scikit-image가 있으면 허블 우주 사진(수업과 동일), 없으면 같은 크기의 검은 이미지."""
    if skimage is not None:
        return skimage.data.hubble_deep_field().copy()

    print("scikit-image가 없어 검은 배경을 사용합니다.")
    return np.zeros(FALLBACK_BACKGROUND_SHAPE, dtype=np.uint8)


def generate_random_boxes(image, box_count, seed, min_width=100, min_height=400, max_width=600, max_height=600):
    """
    이미지 안에 들어가는 난수 Bounding Box와 confidence를 만든다 (수업 노트북 random_bbox·random_conf 함수).

    confidence는 평균 0.6, 표준편차 0.1인 정규분포에서 뽑아 0~1로 잘라낸다.
    반환: (boxes [(x1, y1, x2, y2), ...], scores [float, ...])
    """
    # 전역 random 모듈 대신 전용 생성기를 써서 다른 코드의 난수 사용에 영향을 받지 않게 한다
    generator = random.Random(seed)
    image_height, image_width = image.shape[:2]

    boxes = []
    scores = []
    for _ in range(box_count):
        width = generator.randint(min_width, min(max_width, image_width))
        height = generator.randint(min_height, min(max_height, image_height))
        x1 = generator.randint(0, image_width - width)
        y1 = generator.randint(0, image_height - height)
        boxes.append((x1, y1, x1 + width, y1 + height))
        scores.append(min(1.0, max(0.0, generator.gauss(0.6, 0.1))))

    return boxes, scores


def prune_low_confidence(boxes, scores, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Confidence Thresholding (노트북 TODO `prune_low_conf`): confidence가 임계값보다 낮은 상자를 버린다.

    입력: boxes [(x1, y1, x2, y2), ...], scores [float, ...]
    반환: (남은 boxes, 남은 scores) — 입력과 같은 리스트 형식
    """
    kept_boxes = []
    kept_scores = []
    for box, score in zip(boxes, scores):
        if score >= confidence_threshold:
            kept_boxes.append(box)
            kept_scores.append(score)

    return kept_boxes, kept_scores


def apply_nms(boxes, scores, iou_threshold=IOU_THRESHOLD):
    """
    NMS (노트북 TODO `apply_nms`): confidence가 높은 상자부터 선택하고, 그것과 많이 겹치는 상자를 제거한다.

    1) confidence 내림차순으로 인덱스 정렬 (np.argsort는 오름차순이므로 뒤집음)
    2) 맨 앞(가장 높은) 상자를 keep
    3) 남은 상자들과의 IoU를 계산해 임계값 미만인 것만 후보로 남김
    4) 후보가 없어질 때까지 반복
    입력: boxes [(x1, y1, x2, y2), ...], scores [float, ...]
    반환: (남은 boxes, 남은 scores) — confidence 내림차순
    """
    boxes_array = np.array(boxes)
    scores_array = np.array(scores)

    order = np.argsort(scores_array)[::-1]
    keep_indices = []

    while len(order) > 0:
        current_index = order[0]
        keep_indices.append(current_index)
        remaining = order[1:]

        if len(remaining) == 0:
            break

        current_box = boxes_array[current_index]
        ious = np.array([calculate_iou(current_box, boxes_array[index]) for index in remaining])
        # IoU가 임계값 이상이면 같은 객체의 중복 상자로 보고 버린다
        order = remaining[ious < iou_threshold]

    nms_boxes = [tuple(boxes[index]) for index in keep_indices]
    nms_scores = [scores[index] for index in keep_indices]

    return nms_boxes, nms_scores


def apply_nms_cv2(boxes, scores, confidence_threshold=CONFIDENCE_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    """
    OpenCV 내장 cv2.dnn.NMSBoxes()로 같은 처리를 한다 (수업 노트북 함수). 직접 구현과 결과를 비교하는 기준.

    NMSBoxes는 (x, y, w, h) 형식을 받고, 남길 상자의 인덱스를 돌려준다. confidence thresholding도 함께 처리한다.
    반환: (남은 boxes (x1, y1, x2, y2), 남은 scores)
    """
    if len(boxes) == 0:
        return [], []

    boxes_xywh = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in boxes]

    keep_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=confidence_threshold,
        nms_threshold=iou_threshold,
    )
    # OpenCV 버전에 따라 [[i], [j]] 또는 [i, j] 형태로 나와 1차원으로 통일
    keep_indices = np.array(keep_indices).reshape(-1).tolist()

    return [boxes[index] for index in keep_indices], [scores[index] for index in keep_indices]


def draw_boxes(background, boxes, scores, seed):
    """배경 복사본 위에 상자들을 서로 다른 난수 색으로 그려 돌려준다. 색도 시드를 고정해 재현 가능."""
    generator = random.Random(seed)
    image = background.copy()
    for box, score in zip(boxes, scores):
        color = (generator.randint(0, 255), generator.randint(0, 255), generator.randint(0, 255))
        draw_bounding_box(image, (box[0], box[1]), (box[2], box[3]), color, text=f"{score:.2f}")
    return image


def run_nms_demo():
    """난수 상자 20개 → Confidence Thresholding → NMS → OpenCV NMS와 비교, 세 단계를 나란히 그린다."""
    background = load_background_image()
    boxes, scores = generate_random_boxes(background, RANDOM_BOX_COUNT, RANDOM_SEED)

    pruned_boxes, pruned_scores = prune_low_confidence(boxes, scores)
    nms_boxes, nms_scores = apply_nms(pruned_boxes, pruned_scores)
    cv2_boxes, cv2_scores = apply_nms_cv2(boxes, scores)

    print(f"난수 상자: {len(boxes)}개")
    print(f"Confidence >= {CONFIDENCE_THRESHOLD} : {len(pruned_boxes)}개")
    print(f"직접 구현 NMS (IoU < {IOU_THRESHOLD}) : {len(nms_boxes)}개")
    print(f"cv2.dnn.NMSBoxes : {len(cv2_boxes)}개")
    # 순서는 다를 수 있으므로 집합으로 비교
    is_same_result = set(nms_boxes) == set(tuple(box) for box in cv2_boxes)
    print("직접 구현과 OpenCV 결과 일치:", is_same_result)

    panels = [
        (draw_boxes(background, boxes, scores, RANDOM_SEED), f"Random boxes ({len(boxes)})"),
        (draw_boxes(background, nms_boxes, nms_scores, RANDOM_SEED), f"My NMS ({len(nms_boxes)})"),
        (draw_boxes(background, cv2_boxes, cv2_scores, RANDOM_SEED), f"cv2.dnn.NMSBoxes ({len(cv2_boxes)})"),
    ]

    plt.figure(figsize=(18, 6))
    for subplot_index, (image, title) in enumerate(panels, start=1):
        plt.subplot(1, 3, subplot_index)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIRECTORY / "nms_result.png")


def main():
    """IoU 시연(사과 3장) → NMS 시연(난수 상자) → 결과 창 표시."""
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    print("=== IoU: 사과 이미지 3장 ===")
    run_iou_demo()

    print("\n=== NMS: 난수 Bounding Box ===")
    run_nms_demo()

    plt.show()


if __name__ == "__main__":
    main()
