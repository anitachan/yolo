import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from skimage.feature import graycomatrix, graycoprops
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models (do this once at startup)
mobilenet_model = tf.keras.applications.MobileNetV2(weights="imagenet")
yolo_model = YOLO("yolov8s.pt")  

def process_frame(frame: np.ndarray, selected_object: str):
    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Edge detection (Canny)
    edges = cv2.Canny(gray, 50, 150)
    
    # Compute Haralick texture features
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    # Run YOLO detection
    results = yolo_model(frame)
    object_segmented = np.zeros_like(frame)
    detections = []
    object_found = False
    
    for result in results:
        for obj in result.boxes:
            x1, y1, x2, y2 = map(int, obj.xyxy[0])
            yolo_label = yolo_model.names[int(obj.cls[0])]
            conf = float(obj.conf[0])
            
            # Draw bounding box and label on the frame
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{yolo_label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Object segmentation if label matches selected object
            if yolo_label == selected_object:
                object_segmented[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                object_found = True
                
            # MobileNet classification for the detected object
            cropped_obj = frame[y1:y2, x1:x2]
            if cropped_obj.size > 0:
                img = cv2.resize(cropped_obj, (224, 224))
                img = np.expand_dims(img, axis=0)
                img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
                preds = mobilenet_model.predict(img)
                decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]
                _, classification_label, score = decoded_preds[0]
                mobilenet_label = f"{classification_label} ({score:.2f})"
            else:
                mobilenet_label = "Unknown"
            
            # Draw combined label (YOLO | MobileNet) on the frame
            final_label = f"{yolo_label} | {mobilenet_label}"
            cv2.putText(frame, final_label, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Save detection metadata
            detections.append({
                "label": yolo_label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "classification": mobilenet_label
            })
    
    # If no segmentation occurred, use a blank image for segmentation display
    if not object_found:
        object_segmented = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Overlay Haralick texture features on the main frame
    cv2.putText(frame, f"Contrast: {contrast:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Homogeneity: {homogeneity:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Energy: {energy:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Create a grid display combining original, grayscale, edge and segmented images
    gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    frame_resized = cv2.resize(frame, (640, 480))
    gray_resized = cv2.resize(gray_colored, (640, 480))
    edges_resized = cv2.resize(edges_colored, (640, 480))
    segment_display = cv2.resize(object_segmented, (640, 480))
    top_row = np.hstack((frame_resized, gray_resized))
    bottom_row = np.hstack((edges_resized, segment_display))
    final_display = np.vstack((top_row, bottom_row))
    
    # Encode final_display as JPEG
    _, buffer = cv2.imencode('.jpg', final_display)
    img_encoded = base64.b64encode(buffer).decode('utf-8')
    
    return img_encoded, detections

@app.post("/process-frame")
async def process_frame_api(selected_object: str = Form(...), file: UploadFile = File(...)):
    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    processed_img, detections = process_frame(img, selected_object)
    
    return JSONResponse(content={
        "processed_image": processed_img,
        "detections": detections
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
