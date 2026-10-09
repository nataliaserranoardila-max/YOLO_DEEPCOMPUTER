"""
validate.py
-----------
Evalúa el modelo entrenado sobre el conjunto de VALIDACIÓN y reporta:
  1. mAP@0.5, precisión y recall (globales y por clase) -> calculados por YOLO
  2. Matriz de confusión por clase                       -> imagen guardada por YOLO
  3. Accuracy_val (definición del proyecto):
       una detección es "correcta" si empareja un Ground Truth con IoU >= 0.5
       y además tiene la clase correcta.
       Accuracy_val = GT detectados correctamente / total de GT en validación

Guarda todo en models/metricas.json; la demo de la webcam lee de ahí el Acc_val.

Uso:
    python validate.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

CLASES = ["circulo", "cuadrado", "triangulo"]


def iou(a, b):
    """IoU entre dos cajas [x1, y1, x2, y2] (lo mismo que vimos en clase)."""
    xA, yA = max(a[0], b[0]), max(a[1], b[1])   # inicios: el mayor
    xB, yB = min(a[2], b[2]), min(a[3], b[3])   # finales: el menor
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def leer_gt(txt, w, h):
    """Lee las etiquetas YOLO (normalizadas) y las pasa a [clase, x1, y1, x2, y2] en píxeles."""
    gts = []
    if txt.exists():
        for linea in txt.read_text().split("\n"):
            if linea.strip():
                c, xc, yc, bw, bh = map(float, linea.split())
                gts.append([int(c), (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
    return gts


def accuracy_val(modelo, carpeta_img, carpeta_lbl, conf=0.25, umbral_iou=0.5):
    total, correctas = 0, 0
    por_clase = {c: [0, 0] for c in CLASES}  # [correctas, total]
    imagenes = sorted(Path(carpeta_img).glob("*.jpg"))
    for res in modelo.predict([str(p) for p in imagenes], conf=conf, verbose=False, stream=True):
        h, w = res.orig_shape
        gts = leer_gt(Path(carpeta_lbl) / (Path(res.path).stem + ".txt"), w, h)
        cajas = res.boxes.xyxy.cpu().numpy()
        clases = res.boxes.cls.cpu().numpy().astype(int)
        usadas = set()  # cada predicción solo puede emparejar un GT
        for g in gts:
            total += 1
            por_clase[CLASES[g[0]]][1] += 1
            # Busca la predicción libre con mayor IoU para este GT
            mejor, idx = 0.0, -1
            for i, caja in enumerate(cajas):
                if i not in usadas:
                    v = iou(g[1:], caja)
                    if v > mejor:
                        mejor, idx = v, i
            if idx >= 0 and mejor >= umbral_iou:
                usadas.add(idx)
                if clases[idx] == g[0]:  # IoU >= 0.5 y clase correcta
                    correctas += 1
                    por_clase[CLASES[g[0]]][0] += 1
    acc = correctas / total if total else 0.0
    return acc, {c: (v[0] / v[1] if v[1] else 0.0) for c, v in por_clase.items()}, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default="models/best.pt")
    ap.add_argument("--data", default="data/formas/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    modelo = YOLO(args.pesos)

    # 1 y 2) Métricas estándar de YOLO + matriz de confusión
    m = modelo.val(data=args.data, split="val", imgsz=args.imgsz, plots=True,
                   name="validacion", exist_ok=True, verbose=False)
    print("\n===== MÉTRICAS EN VALIDACIÓN =====")
    print(f"mAP@0.5      : {m.box.map50:.3f}")
    print(f"mAP@0.5:0.95 : {m.box.map:.3f}")
    print(f"Precisión    : {m.box.mp:.3f}")
    print(f"Recall       : {m.box.mr:.3f}")
    print("\nPor clase:            P      R    mAP@0.5")
    por_clase = {}
    for i, c in enumerate(CLASES):
        p, r, ap50, _ = m.box.class_result(i)
        por_clase[c] = {"precision": round(float(p), 4), "recall": round(float(r), 4), "map50": round(float(ap50), 4)}
        print(f"  {c:<12}      {p:.3f}  {r:.3f}  {ap50:.3f}")
    print(f"\nMatriz de confusión guardada en: {m.save_dir}/confusion_matrix.png")

    # 3) Accuracy de validación (IoU >= 0.5 + clase correcta)
    raiz = Path(args.data).parent
    acc, acc_clase, total = accuracy_val(modelo, raiz / "images" / "val", raiz / "labels" / "val")
    print(f"\nAccuracy_val = {acc * 100:.1f}%   ({total} figuras en validación)")
    for c, v in acc_clase.items():
        print(f"  {c:<12} {v * 100:.1f}%")

    Path("models").mkdir(exist_ok=True)
    resumen = {
        "map50": round(float(m.box.map50), 4), "map50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
        "accuracy_val": round(acc, 4), "accuracy_por_clase": {c: round(v, 4) for c, v in acc_clase.items()},
        "por_clase": por_clase, "figuras_val": total,
    }
    Path("models/metricas.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print("\nResumen guardado en models/metricas.json")


if __name__ == "__main__":
    main()
