"""YOLO11n 검출 + ultralytics 내장 추적으로 프레임 한 장에서 사람(추적 id)과 Held Object 후보 물체를 꺼내는 모듈.

출력은 state_machine의 Target State 판정 함수 입력 모양(사람 [(track_id, bounding_box)], 물체 [(class_name, bounding_box)])이다.
확인 실행: .venv/bin/python detector.py ../src/bus.jpg (이미지 한 장만 추적해 출력, 카메라는 열지 않음)
"""
import sys

import cv2
from ultralytics import YOLO

from platform_config import YOLO_MODEL_PATH

# COCO 클래스 이름. 사람만 Target이 된다
PERSON_CLASS_NAME = "person"
# Held Object로 볼 COCO 클래스. 스펙에서 미정, 시연용 예시(프로토타입·스펙 「임시 결정」 이름표와 같음)
# 바꾸면 reporter.py KOREAN_LABELS에도 한국어 이름표를 같이 넣을 것(없으면 영어 낱말이 프롬프트·Piper 음성에 그대로 들어감)
HELD_OBJECT_CLASS_NAMES = ("knife", "bottle", "cell phone", "backpack")  # 임시값, 실측 후 교체


def load_model(model_path=YOLO_MODEL_PATH):
    """검출 모델을 연다. import 때가 아니라 이 함수를 부를 때만 모델을 읽는다.

    입력: model_path — 모델 파일 경로(기본값은 platform_config의 트랙별 경로: Mac .pt, Jetson .engine)
    출력: ultralytics YOLO 모델 객체
    """
    # Jetson 경로: 보드 미검증 — .engine도 같은 YOLO(path)로 열지만, CPU 빌드 torch의 ultralytics로 추론되는지는 미확인
    return YOLO(str(model_path))


def find_class_ids(model, class_names):
    """클래스 이름 목록을 모델의 클래스 번호 목록으로 바꾼다.

    입력: model — YOLO 모델(model.names는 {번호: 이름} dict), class_names — 찾을 이름들
    출력: list[int] — 이름 순서대로의 클래스 번호. 모델에 없는 이름이 있으면 ValueError
    """
    id_by_name = {name: class_id for class_id, name in model.names.items()}
    missing_names = [name for name in class_names if name not in id_by_name]
    if missing_names:
        raise ValueError(f"모델에 없는 클래스 이름: {missing_names}")
    return [id_by_name[name] for name in class_names]


def detect_frame(model, frame):
    """프레임 한 장을 추적해 사람과 Held Object 후보 물체를 나눠 돌려준다.

    입력: model — load_model()의 모델, frame — BGR 이미지(numpy 배열, cv2 프레임 그대로)
    출력: (persons, objects)
        persons — [(track_id, (x1, y1, x2, y2)), ...] 추적 id가 붙은 사람
        objects — [(class_name, (x1, y1, x2, y2)), ...] HELD_OBJECT_CLASS_NAMES에 든 물체
        박스는 정수 픽셀 좌표(ultralytics xyxy를 int로 자름)
    """
    wanted_class_ids = find_class_ids(model, (PERSON_CLASS_NAME,) + HELD_OBJECT_CLASS_NAMES)
    # persist=True: 다음 호출에도 추적기 상태를 이어 써서 같은 사람에게 같은 id를 붙인다. 신뢰도·입력 크기는 기본값
    result = model.track(frame, persist=True, classes=wanted_class_ids, verbose=False)[0]
    boxes = result.boxes

    # 추적기가 이번 프레임에 확정한 트랙이 하나도 없으면 ultralytics는 id 없이(boxes.id is None) 검출만 돌려준다.
    # id가 없는 사람은 상태기계가 Target으로 이어 붙일 수 없으므로 이 프레임에서는 사람을 내보내지 않는다
    has_track_ids = boxes.id is not None
    track_ids = boxes.id.int().tolist() if has_track_ids else [None] * len(boxes)

    persons = []
    objects = []
    for track_id, class_id, coordinates in zip(track_ids, boxes.cls.int().tolist(), boxes.xyxy.tolist()):
        bounding_box = tuple(int(value) for value in coordinates)
        class_name = model.names[class_id]
        if class_name == PERSON_CLASS_NAME:
            if track_id is not None:
                persons.append((track_id, bounding_box))
        else:
            objects.append((class_name, bounding_box))
    return persons, objects


def main():
    """이미지 경로 하나를 받아 한 장 추적 결과(사람·물체)를 출력한다."""
    if len(sys.argv) != 2:
        print("사용법: .venv/bin/python detector.py <이미지 경로>")
        sys.exit(1)
    image_path = sys.argv[1]
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"이미지를 읽지 못했습니다: {image_path}")
        sys.exit(1)

    model = load_model()
    persons, objects = detect_frame(model, frame)
    print(f"이미지: {image_path} (높이 {frame.shape[0]}px)")
    print(f"사람 {len(persons)}명:")
    for track_id, bounding_box in persons:
        print(f"  #{track_id} {bounding_box}")
    print(f"물체 {len(objects)}개:")
    for class_name, bounding_box in objects:
        print(f"  {class_name} {bounding_box}")


if __name__ == "__main__":
    main()
