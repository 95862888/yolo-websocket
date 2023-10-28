from ultralytics import YOLO
import base64
import asyncio
import json
from websockets.sync.client import connect
import cv2


host = 'yolov8-backend-1'


def parse_boxes_to_json(boxes):
    # Create a dictionary to store the attributes
    box_data = {
        "conf": boxes.conf.tolist(),
        "xyxy": boxes.xyxy.tolist(),
        "cls": boxes.cls.tolist(),
        "xywh": boxes.xywh.tolist(),
        "xyxyn": boxes.xyxyn.tolist(),
        "xywhn": boxes.xywhn.tolist()
    }

    if boxes.data is not None:
        box_data["data"] = boxes.data.tolist()

    json_data = json.dumps(box_data)

    return json_data


if __name__ == '__main__':
    model = YOLO('weights.pt')
    model.cfg = 'config.yaml'

    preds = model.predict(source=0, stream=True)

    with connect(f"ws://{host}:8765/") as websocket:
        for pred in preds:
            # boxes = pred.boxes
            # json_boxes = parse_boxes_to_json(boxes)
            # websocket.send(json_boxes)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            result, encoded_img = cv2.imencode('.jpg', pred.orig_img, encode_param)

            websocket.send(base64.b64encode(encoded_img))
