from ultralytics import YOLO
import base64
import asyncio
import json
from websockets.sync.client import connect
import cv2
import math
import numpy as np
from PIL import Image

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
    model = YOLO('best.pt')
    # model.cfg = 'config.yaml'

    # preds = model.predict(source=0, stream=True)

    results = model.predict('/home/kirill/PycharmProjects/prepare_dataset/dataset/video2/frames_rgb/', stream=True)

    for r in results:
        img_array = r.plot()  # plot a BGR numpy array of predictions
        img = Image.fromarray(img_array[..., ::-1])  # RGB PIL image
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow('res', cv_img)
        cv2.waitKey(1)
        print(r.probs)

    # classNames = ['wood', 'glass', 'plastic', 'metal']
    #
    # for r in results:
    #     boxes = r.boxes
    #     img = r.orig_img
    #
    #     for box in boxes:
    #         # bounding box
    #         x1, y1, x2, y2 = box.xyxy[0]
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
    #
    #         # put box in cam
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    #
    #         # confidence
    #         confidence = math.ceil((box.conf[0] * 100)) / 100
    #         print("Confidence --->", confidence)
    #
    #         # class name
    #         cls = int(box.cls[0])
    #         print("Class name -->", classNames[cls])
    #
    #         # object details
    #         org = [x1, y1]
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         fontScale = 1
    #         color = (255, 0, 0)
    #         thickness = 2
    #
    #         cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
    #
    #     cv2.imshow('Webcam', img)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    #
    # cv2.destroyAllWindows()

    # for pred in preds:
    #     cv2.imshow('Results', pred.orig_img)
    #     cv2.waitKey(20)



    for r in results:
        with connect(f"ws://0.0.0.0:8765/") as websocket:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            result, encoded_img = cv2.imencode('.jpg', r.orig_img, encode_param)

            websocket.send(base64.b64encode(encoded_img))

        with connect(f"ws://0.0.0.0:8765/") as websocket:
            classes = r.boxes.cls.tolist()

            json_data = {}

            for cls in classes:
                json_data[r.names[cls]] += 1

            json_data = json.dumps(json_data)

            websocket.send(json_data)

    # with connect(f"ws://{host}:8765/") as websocket:
    #     for pred in preds:
    #         # boxes = pred.boxes
    #         # json_boxes = parse_boxes_to_json(boxes)
    #         # websocket.send(json_boxes)
    #
    #         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    #         result, encoded_img = cv2.imencode('.jpg', pred.orig_img, encode_param)
    #
    #         websocket.send(base64.b64encode(encoded_img))

    # with connect(f"ws://{host}:8765/") as websocket:
    #     for pred in preds:
    #         # boxes = pred.boxes
    #         # json_boxes = parse_boxes_to_json(boxes)
    #         # websocket.send(json_boxes)
    #
    #         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    #         result, encoded_img = cv2.imencode('.jpg', pred.orig_img, encode_param)
    #
    #         websocket.send(base64.b64encode(encoded_img))
