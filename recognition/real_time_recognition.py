import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras_facenet import FaceNet
from ultralytics import YOLO
from collections import deque
import time

DATASET_DIR = "recognition/faces_dataset"
EMBEDDINGS_PATH = "recognition/embeddings.npy"
LABELS_PATH = "recognition/labels.npy"
THRESHOLD = 0.5
HISTORIAL = deque(maxlen=10)

RESOLUCIONES = {
    "1": (640, 480),      # SD
    "2": (1280, 720),     # HD
    "3": (1920, 1080),    # Full HD
    "4": (2560, 1440),    # QHD
    "5": (3840, 2160)     # 4K (solo si tu cámara lo soporta)
}

RESOLUCION_ACTUAL = "2"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "yolov8n-face-lindevs.pt")

embedder = FaceNet()
detector = YOLO(MODEL_PATH)

def mostrar_cargando(mensaje="Cargando..."):
    pantalla = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(pantalla, mensaje, (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow("Cargando", pantalla)
    cv2.waitKey(1) 
    time.sleep(2)
    cv2.destroyWindow("Cargando")

# === FUNCIONES AUXILIARES ===
def aplicar_resolucion(cap):
    w, h = RESOLUCIONES[RESOLUCION_ACTUAL]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[INFO] Resolución actual de cámara: {int(actual_w)}x{int(actual_h)}")

def detectar_resoluciones_soportadas():
    print("[INFO] Detectando resoluciones soportadas por la cámara...")
    resoluciones_prueba = [
        (640, 480),
        (800, 600),
        (1280, 720),
        (1600, 900),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160)
    ]

    cap = cv2.VideoCapture(0)
    soportadas = []
    
    for w, h in resoluciones_prueba:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (actual_w, actual_h) == (w, h):
            soportadas.append((w, h))
            print(f"[✓] {w}x{h} soportada")
        else:
            print(f"[X] {w}x{h} NO soportada (obtenido: {actual_w}x{actual_h})")

    cap.release()
    return soportadas

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

def mejorar_imagen(frame):
    alpha = 1.3
    beta = 20
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def expandir_caja(x1, y1, x2, y2, frame_shape, margen=0.2):
    h, w = frame_shape[:2]
    dx = int((x2 - x1) * margen)
    dy = int((y2 - y1) * margen)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)
    return x1, y1, x2, y2

def dibujar_etiqueta(frame, x1, y1, texto):
    (text_w, text_h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
    cv2.putText(frame, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def dibujar_panel_lateral(frame):
    panel_width = 250
    x_start = frame.shape[1] - panel_width
    cv2.rectangle(frame, (x_start, 0), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
    cv2.putText(frame, "Reconocidos:", (x_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, nombre in enumerate(HISTORIAL):
        cv2.putText(frame, f"- {nombre}", (x_start + 10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

def aplicar_canny_coloreado(rostro):
    canny = cv2.Canny(rostro, 100, 200)
    coloreado = np.zeros_like(rostro)
    coloreado[:, :, 2] = canny  # canal rojo
    return cv2.addWeighted(rostro, 0.6, coloreado, 0.8, 0)

# === FUNCIONES PRINCIPALES ===
def recolectar_imagenes(nombre):
    cap = cv2.VideoCapture(0)
    aplicar_resolucion(cap)
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
            cv2.putText(frame, f"Imagen {imagenes_capturadas + 1}/{imagenes_por_instruccion}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

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
        print("[X] Primero debes generar la base de datos con la opción 1.")
        return

    print("[INFO] Iniciando reconocimiento en tiempo real...")
    mostrar_cargando("Iniciando reconocimiento...")

    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    cap = cv2.VideoCapture(0)
    aplicar_resolucion(cap)

    modo_resaltado = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = mejorar_imagen(frame)
        rostros = detectar_rostros(frame)

        # Imagen Canny con fondo negro
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
            dibujar_panel_lateral(frame)

        modo_texto = "Modo: Canny" if modo_resaltado else "Modo: Normal"
        color_texto = (0, 255, 0) if not modo_resaltado else (0, 100, 255)  # Verde o rojo-naranja

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

def menu():
    global RESOLUCION_ACTUAL
    while True:
        print("\n=== RECONOCIMIENTO FACIAL ===")
        print("Resolución actual: {}x{}".format(*RESOLUCIONES[RESOLUCION_ACTUAL]))
        print("1. Recolectar imágenes de usuario")
        print("2. Generar base de datos")
        print("3. Reconocimiento en tiempo real")
        print("4. Cambiar resolución")
        print("5. Detectar resoluciones soportadas")
        print("6. Salir")
        opcion = input("Opción: ").strip()

        if opcion == "1":
            nombre = input("Nombre del usuario: ").strip()
            recolectar_imagenes(nombre)
        elif opcion == "2":
            generar_base_de_datos()
        elif opcion == "3":
            reconocimiento_tiempo_real()
        elif opcion == "4":
            print("\nSelecciona resolución:")
            for key, (w, h) in RESOLUCIONES.items():
                print(f"{key}. {w}x{h}")
            nueva = input("Opción: ").strip()
            if nueva in RESOLUCIONES:
                RESOLUCION_ACTUAL = nueva
                print("[✓] Resolución actualizada")
            else:
                print("[X] Opción inválida.")
        elif opcion == "5":
            resoluciones = detectar_resoluciones_soportadas()
            print("[INFO] Resoluciones disponibles:")
            for w, h in resoluciones:
                print(f" - {w}x{h}")
        elif opcion == "6":
            print("Saliendo...")
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    menu()
