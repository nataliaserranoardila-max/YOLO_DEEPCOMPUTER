# Detección de formas geométricas con YOLOv8

Proyecto de 2° corte – Electiva IV, Deep Computer Vision (USTA Bucaramanga).

El modelo detecta **círculos, cuadrados y triángulos** en video en vivo desde la webcam. En la ventana muestra al mismo tiempo:
- el bounding box,
- la figura y su color,
- la confianza,
- el **Accuracy de validación** fijo (`Acc_val = xx%`).

## Estructura

```
proyecto_yolo_formas/
├── data/
│   └── generar_dataset.py    # crea el dataset sintético en formato YOLO (70/20/10)
├── models/                   # aquí queda best.pt y metricas.json
├── train.py                  # entrena YOLOv8n
├── validate.py               # mAP@0.5, precisión, recall, matriz de confusión y Accuracy_val
├── demo_webcam.py            # detección en tiempo real con la webcam
├── entrenar_en_colab.ipynb   # mismo flujo, listo para Google Colab (GPU gratis)
├── requirements.txt
└── README.md
```

## 1. Instalación

```bash
pip install -r requirements.txt
```

## 2. Preparación de datos

```bash
# 1) Descargar 128 fotos reales (COCO128) para usar como fondo
#    https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip  -> descomprimir aquí
# 2) Generar el dataset
python data/generar_dataset.py --n 1200 --fondos coco128/images/train2017
```

- Crea `data/formas/` con `images/` y `labels/`, divididos en train (840), val (240) y test (120), más el archivo `data.yaml`.
- Cada imagen tiene un `.txt` con una línea por figura: `clase x_centro y_centro ancho alto`, normalizado entre 0 y 1.
- Las clases son `0 = circulo`, `1 = cuadrado`, `2 = triangulo`.
- Con `--semilla 42`, que es el valor por defecto, siempre se genera el mismo dataset, así que los resultados se pueden reproducir.

## 3. Entrenamiento

```bash
python train.py                  # imgsz=640, epochs=60, batch=16, early stopping (patience=15)
python train.py --batch 8        # si la GPU se queda sin memoria
```

- Parte de `yolov8n.pt`, preentrenado en COCO (transfer learning).
- El mejor modelo se copia a `models/best.pt`.
- Las gráficas de pérdidas y mAP quedan en `runs/detect/formas/`.
- **Sin GPU** es lento. Lo más fácil es entrenar en Colab con `entrenar_en_colab.ipynb` (unos 20–30 min con GPU T4).

## 4. Validación y métricas

```bash
python validate.py
```

- Imprime mAP@0.5, mAP@0.5:0.95, precisión y recall, globales y por clase.
- Guarda la matriz de confusión en `runs/detect/validacion/confusion_matrix.png`.
- Calcula el **Accuracy_val**: una figura del ground truth cuenta como correcta si alguna predicción la empareja con **IoU ≥ 0.5** y además con la **clase correcta**. Accuracy_val = correctas / total de figuras en validación.
- Guarda todo en `models/metricas.json`.

## 5. Demo en tiempo real

```bash
python demo_webcam.py              # webcam 0
python demo_webcam.py --camara 1   # si tienes otra cámara
python demo_webcam.py --fuente data/formas/images/test/img_0545.jpg   # probar con una imagen
```

Para salir, presiona `q` o `ESC`. La detección la hace YOLO; OpenCV solo abre la cámara, dibuja y estima el color dentro de cada caja (con el tono HSV).

**Consejos para la cámara:**
- Funciona con figuras dibujadas con marcador en hoja blanca y con figuras recortadas en cartulina.
- Usa buena luz.
- Si salen detecciones falsas, sube la confianza mínima: `python demo_webcam.py --conf 0.7`.
- Colores que reconoce: rojo, naranja, amarillo, verde, celeste, azul, morado, rosado, fucsia, negro y blanco.

## 6. Reproducir las métricas

```bash
python data/generar_dataset.py --n 1200 --fondos coco128/images/train2017   # mismo dataset (semilla 42)
python validate.py --pesos models/best.pt
```

## Dataset

El dataset es **generado artificialmente por el equipo** con `data/generar_dataset.py`. Las figuras y sus etiquetas las crea el script; lo único que viene de un repositorio abierto son 128 fotos reales que se usan **solo como fondo** (set público **COCO128** de Ultralytics, tomado de COCO).

- **Tamaño:** 1200 imágenes de 640×640, divididas en 840 train, 240 val y 120 test.
- **Clases:** 3 (`circulo`, `cuadrado`, `triangulo`), con unas 1.100 figuras por clase, equilibradas.
- **Formato:** YOLO (un `.txt` por imagen).

Cada imagen varía en:

| Variación | Para qué sirve |
|---|---|
| Fondo tipo hoja (claro u oscuro, degradado, ruido) o **foto real** (55 %) | Que funcione sobre papel y con la habitación de fondo |
| **Imágenes negativas**: fotos reales sin figuras, `.txt` vacío (~20 %) | Que no confunda caras, manos, techos u objetos con figuras |
| 1 a 6 figuras; **40 % pequeñas** (24–70 px) y el resto de 70–280 px | Detectar figuras lejos de la cámara |
| Cualquier color (incluidos rosado, celeste y negro) | Que no dependa del color |
| Contorno (recto o hecho a mano) o relleno tipo **cartulina recortada con sombra** | Figuras dibujadas o recortadas |
| **Dedos** tapando un borde (20 %) | Figuras sostenidas con la mano |
| Rotación y **perspectiva** (hoja inclinada) | Hoja girada o vista en ángulo |
| **Desenfoque de movimiento** (25 %), desenfoque normal, brillo | Cuando la hoja o la cámara se mueven |
| Distractores sin etiqueta (líneas, garabatos, texto) | Que no todo trazo sea una figura |
