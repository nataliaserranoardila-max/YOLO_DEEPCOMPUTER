"""
train.py
--------
Entrena YOLOv8 (versión nano) para detectar círculo, cuadrado y triángulo.

Se parte de los pesos preentrenados en COCO (transfer learning): el backbone ya sabe
extraer bordes y formas, y solo se ajusta a nuestras 3 clases.

Uso:
    python train.py                       # parámetros por defecto
    python train.py --epochs 30 --batch 8 # si tu GPU tiene poca memoria
Al terminar, el mejor modelo queda copiado en models/best.pt
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/formas/data.yaml")
    ap.add_argument("--modelo", default="yolov8n.pt", help="modelo base (>= v8)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="bájalo a 8 si se acaba la memoria")
    ap.add_argument("--patience", type=int, default=15, help="early stopping")
    args = ap.parse_args()

    modelo = YOLO(args.modelo)  # carga YOLOv8n preentrenado
    resultados = modelo.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,  # para si val no mejora en 'patience' épocas
        name="formas",
        exist_ok=True,
        seed=42,
    )

    # YOLO guarda el mejor modelo (mayor mAP en validación) como best.pt
    best = Path(resultados.save_dir) / "weights" / "best.pt"
    Path("models").mkdir(exist_ok=True)
    shutil.copy(best, "models/best.pt")
    print(f"\nMejor modelo copiado a models/best.pt  (original: {best})")
    print("Gráficas de entrenamiento (pérdidas, mAP, matriz de confusión) en:", resultados.save_dir)


if __name__ == "__main__":
    main()
