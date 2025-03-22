# --- FastAPI Backend (main.py) ---

import cv2
import numpy as np
import tensorflow as tf
import json
import base64
import asyncio
import gc
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from skimage.feature import graycomatrix, graycoprops
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mobilenet_model = tf.keras.applications.MobileNetV2(weights="imagenet")
yolo_model = YOLO("yolov8n.pt")
yolo_model.fuse()

frame_lock = asyncio.Lock()
executor = ThreadPoolExecutor(max_workers=2)


def process_frame(frame: np.ndarray, selected_object: str):
    start_time = time.time()

    frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    results = yolo_model(frame)
    object_segmented = np.zeros_like(frame)
    detections = []
    object_found = False

    for result in results:
        for obj in result.boxes:
            x1, y1, x2, y2 = map(int, obj.xyxy[0])
            yolo_label = yolo_model.names[int(obj.cls[0])]
            conf = float(obj.conf[0])

            if yolo_label == selected_object:
                object_segmented[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                object_found = True

            cropped_obj = frame[y1:y2, x1:x2]
            if cropped_obj.size > 0:
                img = cv2.resize(cropped_obj, (224, 224))
                img = np.expand_dims(img, axis=0)
                img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
                preds = mobilenet_model.predict(img, verbose=0)
                decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]
                _, classification_label, score = decoded_preds[0]
                mobilenet_label = f"{classification_label} ({score:.2f})"
            else:
                mobilenet_label = "Unknown"

            detections.append({
                "label": yolo_label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "classification": mobilenet_label
            })

    if not object_found:
        object_segmented = np.zeros((100, 100, 3), dtype=np.uint8)

    gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    frame_resized = cv2.resize(frame, (640, 480))
    gray_resized = cv2.resize(gray_colored, (640, 480))
    edges_resized = cv2.resize(edges_colored, (640, 480))
    segment_display = cv2.resize(object_segmented, (640, 480))
    top_row = np.hstack((frame_resized, gray_resized))
    bottom_row = np.hstack((edges_resized, segment_display))
    final_display = np.vstack((top_row, bottom_row))

    _, buffer = cv2.imencode('.jpg', final_display, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    img_encoded = base64.b64encode(buffer).decode('utf-8')

    gc.collect()
    elapsed = time.time() - start_time
    print(f"\u23f1\ufe0f Tiempo por frame: {elapsed:.3f}s - FPS: {1 / elapsed:.2f}")

    return img_encoded, detections


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    selected_object = "person"

    while True:
        try:
            message = await websocket.receive()
            if "text" in message and message["text"]:
                try:
                    data_json = json.loads(message["text"])
                    if "selected_object" in data_json:
                        selected_object = data_json["selected_object"]
                        await websocket.send_text(f"selected_object actualizado a {selected_object}")
                        continue
                except json.JSONDecodeError:
                    pass

            text_data = message.get("text")
            binary_data = message.get("bytes")

            if binary_data:
                data = binary_data
            elif text_data:
                try:
                    data_json = json.loads(text_data)
                    if "selected_object" in data_json:
                        selected_object = data_json["selected_object"]
                        await websocket.send_text(f"selected_object actualizado a {selected_object}")
                        continue
                except json.JSONDecodeError:
                    continue
                data = text_data.encode('utf-8')
            else:
                continue

            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            async with frame_lock:
                processed_img, detections = await asyncio.get_event_loop().run_in_executor(
                    executor, process_frame, frame, selected_object
                )

            response = {
                "processed_image": processed_img,
                "detections": detections
            }
            await websocket.send_text(json.dumps(response))

        except Exception as e:
            print("Error en WebSocket:", e)
            break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
