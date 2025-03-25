import cv2
import numpy as np
import os
import time

RESOLUCIONES = {
    "1": (640, 480),      # SD
    "2": (1280, 720),     # HD
    "3": (1920, 1080),    # Full HD
    "4": (2560, 1440),    # QHD
    "5": (3840, 2160)     # 4K (solo si tu cámara lo soporta)
}

def mostrar_cargando(mensaje="Cargando..."):
    pantalla = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(pantalla, mensaje, (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow("Cargando", pantalla)
    cv2.waitKey(1)
    time.sleep(2)
    cv2.destroyWindow("Cargando")

def aplicar_resolucion(cap, resolucion_actual):
    w, h = RESOLUCIONES[resolucion_actual]
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

def dibujar_panel_lateral(frame, historial):
    panel_width = 250
    x_start = frame.shape[1] - panel_width
    cv2.rectangle(frame, (x_start, 0), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
    cv2.putText(frame, "Reconocidos:", (x_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, nombre in enumerate(historial):
        cv2.putText(frame, f"- {nombre}", (x_start + 10, 60 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

def aplicar_canny_coloreado(rostro):
    canny = cv2.Canny(rostro, 100, 200)
    coloreado = np.zeros_like(rostro)
    coloreado[:, :, 2] = canny  # canal rojo
    return cv2.addWeighted(rostro, 0.6, coloreado, 0.8, 0)
