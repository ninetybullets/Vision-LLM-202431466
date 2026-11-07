import cv2
import json

from ultralytics import YOLO
from llama_cpp import Llama


YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
JSON_PATH = "src/json/vision_data.json"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 150

def detections_to_text(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        vision_data = json.load(file)
        
    objects = vision_data["objects"]

    if len(objects) == 0:
        return "현재 탐지된 객체가 없습니다."

    sentences = []

    for index, obj in enumerate(objects, start=1):
        sentence = f"{index}번 객체는 {obj['class']}이며, confidence는 {obj['confidence']:.2f}입니다."

        sentences.append(sentence)

    return "\n".join(sentences)


yolo = YOLO(YOLO_MODEL_PATH)
llm = Llama(model_path=GEMMA_MODEL_PATH, n_gpu_layers=-1, n_ctx=CONTEXT_WINDOW, n_batch=32, n_ubatch=32, verbose=False)
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()


print("q : 종료")
print("l : 현재 YOLO 탐지 결과를 Gemma에게 전달")


while True:
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF

    if not ret:
        break
    if key == ord("q"):
        break


    # 1. 현재 프레임 YOLO 탐지
    results = yolo.predict(
        source=frame,
        conf=0.25,
        iou=0.5,
        verbose=False,
    )

    result = results[0]


    # 2. YOLO 결과 화면 출력
    output_frame = result.plot()
    cv2.imshow("YOLO + Gemma", output_frame)


    # TODO: 3. YOLO 탐지 결과를 Dictionary 형태로 변환
    objects = []

    for box in result.boxes:
        class_id = int(box.cls[0].items())
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
        objects.append(...)

    vision_dict = ...

    
    if key == ord("l"):
        # TODO: 4. Dictionary → JSON 파일 저장
        with ...:
            json.dump(...)
            
        # TODO: 5. JSON 파일 → Natural Language 변환
        vision_text = ...

        # TODO: 6. 탐지 결과를 Gemma에 Context로 전달
        response = ...


cap.release()
cv2.destroyAllWindows()