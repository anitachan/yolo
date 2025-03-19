
#Instalar librerías requeridas para el funcionamiento del código
#pip install opencv-python
#pip install ultralytics opencv-python numpy
#pip install scikit-image

#Código pyhton - modulo 2 visión artificial
#Grupo 1 - Ana Chávez - Adrían Changalombo - Pablo Sosa

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk

# Cargar el modelo YOLOv8 preentrenado
model = YOLO("yolov8n.pt")

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Lista fija de objetos disponibles en el menú desplegable
objetos_fijos = [
    "person", "cell phone", "bicycle", "car", "dog", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "bottle", "cup",
    "spoon", "chair", "door", "mouse", "keyboard", "laptop", "tv", "book"
]

# Crear la interfaz de selección de objetos con Tkinter
root = tk.Tk()
root.title("Seleccionar Objeto")

label = tk.Label(root, text="Elige un objeto para segmentar:")
label.pack()

selected_object = tk.StringVar(root)
dropdown = ttk.Combobox(root, textvariable=selected_object)
dropdown.pack()
dropdown['values'] = objetos_fijos  # Mostrar los nombres amigables en el menú
selected_object.set(objetos_fijos[0])  # Seleccionar el primer objeto por defecto

root.update()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar detección de bordes con Canny
    edges = cv2.Canny(gray, 50, 150)

    # 📌 **Calcular la matriz de coocurrencia de Haralick (GLCM)**
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    # Detectar todos los objetos con YOLOv8
    results = model(frame)

    # Inicializar imagen donde se segmentarán los objetos seleccionados
    object_segmented = np.zeros_like(frame)

    # Obtener el objeto seleccionado en el menú desplegable
    objeto_elegido = selected_object.get()
    object_found = False

    # Dibujar todos los objetos detectados
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja
            conf = float(box.conf[0])  # Confianza de detección
            cls = int(box.cls[0])  # Clase detectada
            label = model.names[cls]  # Nombre del objeto detectado

            # Dibujar caja y etiqueta en la imagen principal (1er recuadro)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Si el objeto detectado coincide con el seleccionado en el dropbox, segmentarlo
            if label == objeto_elegido:
                object_segmented[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                object_found = True

    # Si el objeto seleccionado no se detectó, dejar el recuadro vacío
    if not object_found:
        object_segmented = np.zeros((100, 100, 3), dtype=np.uint8)  # Cuadro vacío pequeño

    # Convertir imágenes a BGR para combinarlas correctamente
    gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 📌 **Redimensionar las imágenes principales**
    frame_resized = cv2.resize(frame, (640, 480))
    gray_resized = cv2.resize(gray_colored, (640, 480))
    edges_resized = cv2.resize(edges_colored, (640, 480))

    # 📌 **Dibujar valores de Haralick en la imagen principal**
    cv2.putText(frame_resized, f"Contrast: {contrast:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame_resized, f"Homogeneity: {homogeneity:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame_resized, f"Energy: {energy:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 📌 **No redimensionar el objeto segmentado, sino colocarlo en un fondo negro**
    segment_display = np.zeros((480, 640, 3), dtype=np.uint8)
    if np.any(object_segmented):  # Si hay algo segmentado, copiarlo al 4to recuadro
        segment_display = object_segmented

    # 📌 **Combinar las imágenes en una sola ventana (matriz 2x2)**
    top_row = np.hstack((frame_resized, gray_resized))
    bottom_row = np.hstack((edges_resized, segment_display))
    final_display = np.vstack((top_row, bottom_row))

    # Mostrar la única ventana con todas las técnicas aplicadas
    cv2.imshow("Detección YOLO + Haralick", final_display)

    # 📌 **Actualizar la ventana de Tkinter para que no se congele**
    root.update()

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()
