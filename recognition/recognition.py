import os
import cv2
import numpy as np
import time
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
from keras_facenet import FaceNet
from ultralytics import YOLO
from utils import (aplicar_resolucion, mejorar_imagen, expandir_caja,
                   dibujar_etiqueta, dibujar_panel_lateral, mostrar_cargando)

DATASET_DIR = "recognition/faces_dataset"
EMBEDDINGS_PATH = "recognition/embeddings.npy"
LABELS_PATH = "recognition/labels.npy"
THRESHOLD = 0.5
HISTORIAL = deque(maxlen=10)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "yolov8n-face-lindevs.pt")

embedder = FaceNet()
detector = YOLO(MODEL_PATH)

def l2_normalize(x):
    return x / np.linalg.norm(x)

def extraer_embedding(rostro_bgr):
    if rostro_bgr is None or rostro_bgr.size == 0:
        raise ValueError("Rostro vacío")
    rostro_rgb = cv2.cvtColor(rostro_bgr, cv2.COLOR_BGR2RGB)
    emb = embedder.embeddings([rostro_rgb])[0]
    return l2_normalize(emb)

def detectar_rostros(frame):
    results = detector(frame, verbose=False)
    rostros = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            rostros.append((x1, y1, x2, y2))
    return rostros

def recolectar_imagenes(nombre):
    cap = cv2.VideoCapture(0)
    # Usar resolución "2" (HD) por defecto
    resolucion_actual = "2"
    aplicar_resolucion(cap, resolucion_actual)
    user_dir = os.path.join(DATASET_DIR, nombre)
    os.makedirs(user_dir, exist_ok=True)
    print(f"[INFO] Recolectando imágenes para '{nombre}'...")

    instrucciones = [
        "Mira al frente",
        "Gira tu rostro a la izquierda",
        "Gira tu rostro a la derecha",
        "Inclina tu rostro hacia arriba",
        "Inclina tu rostro hacia abajo",
        "Perfil izquierdo 90°",
        "Perfil derecho 90°"
    ]

    imagenes_por_instruccion = 10
    count = 0

    for paso, instruccion in enumerate(instrucciones):
        print(f"[PASO {paso + 1}] {instruccion}")
        print("Capturando automáticamente. Mantén la posición...")

        imagenes_capturadas = 0
        last_capture_time = time.time()

        while imagenes_capturadas < imagenes_por_instruccion:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = mejorar_imagen(frame)

            cv2.putText(frame, instruccion, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Imagen {imagenes_capturadas + 1}/{imagenes_por_instruccion}",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

            current_time = time.time()
            if current_time - last_capture_time >= 0.5:
                rostros = detectar_rostros(frame)
                if rostros:
                    x1, y1, x2, y2 = expandir_caja(*rostros[0], frame.shape)
                    rostro = frame[y1:y2, x1:x2].copy()

                    path = os.path.join(user_dir, f"{nombre}_{count}.jpg")
                    cv2.imwrite(path, rostro)
                    count += 1
                    imagenes_capturadas += 1
                    last_capture_time = current_time
                    print(f"[✓] Imagen {imagenes_capturadas}/{imagenes_por_instruccion} guardada: {path}")

            cv2.imshow("Recolección", frame)
            if cv2.waitKey(1) == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()
    print("[✓] Recolección completa.")

def generar_base_de_datos():
    embeddings, labels = [], []
    print("[INFO] Generando base de datos de embeddings...")

    for persona in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, persona)
        for img_file in os.listdir(person_dir):
            path = os.path.join(person_dir, img_file)
            img = cv2.imread(path)
            if img is None:
                continue
            try:
                emb = extraer_embedding(img)
                embeddings.append(emb)
                labels.append(persona)
            except Exception as e:
                print(f"[ERROR] Al procesar {path}: {e}")

    np.save(EMBEDDINGS_PATH, np.array(embeddings))
    np.save(LABELS_PATH, np.array(labels))
    print("[✓] Base de datos generada exitosamente.")

def reconocimiento_tiempo_real():
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(LABELS_PATH):
        print("[X] Primero debes generar la base de datos con la opción 2.")
        return

    print("[INFO] Iniciando reconocimiento en tiempo real...")
    mostrar_cargando("Iniciando reconocimiento...")

    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    cap = cv2.VideoCapture(0)
    resolucion_actual = "2"
    aplicar_resolucion(cap, resolucion_actual)

    modo_resaltado = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = mejorar_imagen(frame)
        rostros = detectar_rostros(frame)

        # Generar imagen de bordes (Canny) con fondo negro
        canny_total = cv2.Canny(frame, 100, 200)
        canny_color = np.zeros_like(frame)
        canny_color[canny_total != 0] = (255, 255, 255)

        if rostros:
            x1, y1, x2, y2 = expandir_caja(*rostros[0], frame.shape)
            rostro = frame[y1:y2, x1:x2].copy()

            try:
                emb = extraer_embedding(rostro)
            except Exception as e:
                print(f"[ERROR] Embedding fallido: {e}")
                continue

            sims = cosine_similarity([emb], embeddings)[0]
            best_idx = np.argmax(sims)
            score = sims[best_idx]
            nombre = labels[best_idx] if score > THRESHOLD else "Desconocido"
            texto = f"{nombre} ({score:.2f})" if nombre != "Desconocido" else nombre

            if nombre != "Desconocido" and texto not in HISTORIAL:
                HISTORIAL.appendleft(texto)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            dibujar_etiqueta(frame, x1, y1, texto)

            if modo_resaltado:
                rostro_area = canny_total[y1:y2, x1:x2]
                canny_color[y1:y2, x1:x2][rostro_area != 0] = (0, 0, 255)
                cv2.rectangle(canny_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                dibujar_etiqueta(canny_color, x1, y1, texto)

        if HISTORIAL:
            dibujar_panel_lateral(frame, HISTORIAL)

        modo_texto = "Modo: Canny" if modo_resaltado else "Modo: Normal"
        color_texto = (0, 255, 0) if not modo_resaltado else (0, 100, 255)

        for vista_texto in [frame, canny_color]:
            overlay = vista_texto.copy()
            cv2.rectangle(overlay, (5, 5), (350, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, vista_texto, 0.6, 0, dst=vista_texto)
            cv2.putText(vista_texto, modo_texto, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2)
            cv2.putText(vista_texto, "Presiona 'b' para cambiar modo", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_texto, 1)
            cv2.putText(vista_texto, "Presiona 'ESC' para salir", (15, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_texto, 1)

        vista = canny_color if modo_resaltado else frame
        cv2.imshow("Reconocimiento Facial", vista)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('b'):
            modo_resaltado = not modo_resaltado
            print(f"[INFO] Modo de bordes cambiado: {'Resaltado' if modo_resaltado else 'Normal'}")
            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
