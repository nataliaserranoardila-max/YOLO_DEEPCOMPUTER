"""
demo_webcam.py
--------------
Detección en TIEMPO REAL con la webcam. En la ventana se muestra, al mismo tiempo:
  - bounding box de cada figura
  - etiqueta: tipo de figura + color detectado + confianza
  - overlay fijo con el Accuracy de validación (Acc_val, leído de models/metricas.json)
  - FPS

La detección la hace YOLO. OpenCV solo se usa para abrir la cámara, dibujar y,
como paso extra, estimar el color dentro de cada caja detectada.

Uso:
    python demo_webcam.py                 # cámara 0
    python demo_webcam.py --camara 1      # otra cámara
    python demo_webcam.py --fuente foto.jpg   # probar con una imagen
Tecla q o ESC para salir.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Color del recuadro para cada clase (BGR)
COLOR_CLASE = {"circulo": (0, 200, 0), "cuadrado": (255, 120, 0), "triangulo": (0, 0, 230)}


def color_dominante(recorte):
    """Estima el nombre del color de la figura dentro de la caja usando HSV.
    El fondo (hoja blanca) ocupa casi toda la caja, así que no se promedia todo:
    se cuentan los píxeles CON COLOR (saturados) y los píxeles OSCUROS (trazo negro)
    y gana el grupo que tenga más píxeles."""
    if recorte.size == 0:
        return "?"
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    con_color = (s > 80) & (v > 60)   # píxeles con color
    oscuros = (v < 90) & (s <= 80)    # píxeles negros (marcador negro)
    if con_color.mean() < 0.005 and oscuros.mean() < 0.005:
        return "blanco"
    if oscuros.mean() > con_color.mean():
        return "negro"
    tono = np.median(h[con_color])  # en OpenCV el tono (H) va de 0 a 179
    sat = np.median(s[con_color])
    if tono < 8 or tono >= 172:
        return "rojo"
    if tono < 20:
        return "naranja"
    if tono < 38:
        return "amarillo"
    if tono < 80:
        return "verde"
    if tono < 100:
        return "celeste"
    if tono < 130:
        return "azul"
    if tono < 150:
        return "morado"
    return "rosado" if sat < 200 else "fucsia"


def dibujar(frame, res, nombres, acc_val, fps):
    for caja, cls, conf in zip(res.boxes.xyxy.cpu().numpy().astype(int),
                               res.boxes.cls.cpu().numpy().astype(int),
                               res.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = caja
        clase = nombres[cls]
        color = color_dominante(frame[max(0, y1):y2, max(0, x1):x2])
        c = COLOR_CLASE.get(clase, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        # Etiqueta en dos líneas cortas (figura / color + confianza) para que no se monten
        lineas = [clase, f"{color} {conf * 100:.0f}%"]
        fuente, esc, alto = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 20
        ancho = max(cv2.getTextSize(t, fuente, esc, 2)[0][0] for t in lineas) + 8
        y_top = y1 - alto * 2 if y1 - alto * 2 >= 0 else y1  # si no cabe arriba, va adentro
        cv2.rectangle(frame, (x1, y_top), (x1 + ancho, y_top + alto * 2), c, -1)
        for i, t in enumerate(lineas):
            cv2.putText(frame, t, (x1 + 4, y_top + alto * (i + 1) - 5), fuente, esc, (255, 255, 255), 2)

    # Overlay fijo con el accuracy de validación y los FPS
    cv2.rectangle(frame, (8, 8), (260, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"Acc_val = {acc_val * 100:.1f}%", (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default="models/best.pt")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--fuente", default=None, help="ruta de una imagen o video (opcional)")
    ap.add_argument("--conf", type=float, default=0.5, help="confianza mínima para mostrar (súbela, p. ej. 0.7, si salen detecciones falsas)")
    args = ap.parse_args()

    modelo = YOLO(args.pesos)
    metricas = Path("models/metricas.json")
    acc_val = json.loads(metricas.read_text())["accuracy_val"] if metricas.exists() else 0.0
    if not metricas.exists():
        print("Aviso: no existe models/metricas.json; corre primero python validate.py")

    # Modo imagen: útil para probar sin cámara
    if args.fuente and Path(args.fuente).suffix.lower() in (".jpg", ".jpeg", ".png"):
        frame = cv2.imread(args.fuente)
        res = modelo.predict(frame, conf=args.conf, verbose=False)[0]
        salida = dibujar(frame, res, modelo.names, acc_val, 0.0)
        cv2.imwrite("resultado_demo.jpg", salida)
        print("Resultado guardado en resultado_demo.jpg")
        return

    cap = cv2.VideoCapture(args.fuente if args.fuente else args.camara)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara. Prueba con --camara 1")

    previo = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = modelo.predict(frame, conf=args.conf, verbose=False)[0]  # detección YOLO
        ahora = time.time()
        fps = 1.0 / max(ahora - previo, 1e-6)
        previo = ahora
        cv2.imshow("YOLO - Deteccion de formas geometricas", dibujar(frame, res, modelo.names, acc_val, fps))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
