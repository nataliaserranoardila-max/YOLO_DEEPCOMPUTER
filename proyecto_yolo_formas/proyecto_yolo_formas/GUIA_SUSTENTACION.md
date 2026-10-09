# Guía para sustentar el proyecto

Esta guía es para estudiar y para la presentación. No es un entregable.

## Qué hace cada archivo (dilo así)

| Archivo | En una frase |
|---|---|
| `data/generar_dataset.py` | Dibuja figuras (pequeñas y grandes, a mano o tipo cartulina, con dedos y movimiento) sobre fondos de hoja o fotos reales (COCO128), agrega imágenes negativas sin figuras, guarda las etiquetas en formato YOLO y divide en 70/20/10. |
| `train.py` | Carga YOLOv8n preentrenado y lo reentrena con nuestras 3 clases (transfer learning). Guarda `best.pt`. |
| `validate.py` | Mide mAP@0.5, precisión, recall y la matriz de confusión, y calcula a mano el Accuracy_val con IoU ≥ 0.5 y clase correcta. |
| `demo_webcam.py` | Lee la cámara, YOLO detecta, y se dibuja la caja, la figura, el color, la confianza, el Acc_val y los FPS. |

## Flujo del proyecto

```
generar_dataset.py → data/formas (imágenes + .txt) → train.py → models/best.pt
                                                   → validate.py → models/metricas.json
best.pt + metricas.json → demo_webcam.py → video con detecciones y Acc_val
```

## Conceptos que te pueden preguntar

**¿Por qué YOLO y no sliding window?**
YOLO es un detector de **una sola pasada** (*You Only Look Once*): mira la imagen completa una vez y predice todas las cajas al mismo tiempo. Sliding window clasifica miles de ventanas una por una, lo que es lentísimo. Por eso YOLO sirve para tiempo real (FPS altos).

**Arquitectura de YOLOv8, en sus tres partes:**
- **Backbone (CSPDarknet con bloques C2f):** extrae las características: bordes, esquinas, formas.
- **Neck (FPN + PAN):** combina mapas de tres escalas, para detectar figuras grandes y pequeñas.
- **Head:** predice para cada posición la caja y la clase. Es *anchor-free*: no usa cajas predefinidas.
- Al final se aplica **NMS** para dejar una sola caja por figura.
- **YOLOv8n** ("nano") es la versión más pequeña, con unos 3 millones de parámetros. Se eligió por ser rápida en un portátil.

**Transfer learning:** partimos de pesos entrenados en COCO (80 clases). El backbone ya sabe "ver" y solo se ajusta a círculo, cuadrado y triángulo. Por eso aprende rápido con solo 840 imágenes de entrenamiento.

**Formato YOLO:** `clase x_centro y_centro ancho alto`, normalizado entre 0 y 1. Se divide entre el ancho y el alto de la imagen para que no dependa del tamaño.

**Parámetros de entrenamiento:**
- `imgsz=640`: tamaño de entrada.
- `epochs=60`: máximo de pasadas por el dataset.
- `batch=16`: imágenes por paso.
- `patience=15`: early stopping. Si el mAP de validación no mejora en 15 épocas, se detiene y evita el sobreajuste.
- `best.pt` es el modelo de la época con mejor métrica en validación; `last.pt` es el de la última época.

**Métricas:**
- **IoU** = intersección / unión entre la caja predicha y la real.
- **Precisión** = TP / (TP + FP): de lo que detectó, cuánto era correcto.
- **Recall** = TP / (TP + FN): de las figuras que había, cuántas encontró.
- **mAP@0.5** es el promedio, entre las clases, del AP (área bajo la curva precisión-recall), contando como acierto una caja con IoU ≥ 0.5.
- **mAP@0.5:0.95** es igual, pero promediando umbrales de IoU de 0.5 a 0.95. Es más exigente.
- **Matriz de confusión:** filas y columnas por clase; muestra qué clase se confunde con cuál. "Background" son las figuras no detectadas y las detecciones falsas.
- **Accuracy_val (nuestra):** por cada figura real de validación se busca la predicción con mayor IoU. Si IoU ≥ 0.5 y la clase coincide, es correcta. Accuracy = correctas / total. Está en `accuracy_val()` dentro de `validate.py`, y la función `iou()` es la misma de clase: el mayor de los inicios y el menor de los finales.

**¿Para qué las imágenes negativas?**
Son fotos reales **sin figuras**, con su `.txt` vacío. YOLO las usa como ejemplos de "aquí no hay nada", así que aprende a no disparar detecciones falsas sobre caras, manos o muebles.

**¿Por qué la caja es un rectángulo y no sigue el contorno?**
El proyecto pide **bounding box**. Seguir el contorno exacto sería **segmentación de instancias**, que es otra tarea, con modelos como YOLOv8-seg.

**¿Cómo sale el color?**
YOLO solo detecta la forma. Dentro de cada caja se pasa a HSV y se toma el **tono (H)** mediano de los píxeles con color (saturación alta), ignorando el fondo blanco. Según el rango de H se dice rojo, naranja, amarillo, verde, celeste, azul, morado, rosado o fucsia. Si dominan los píxeles oscuros sin color, dice negro (el trazo de marcador); si casi no hay nada, blanco. No se promedia toda la caja porque casi toda es papel blanco.

**¿Usaron solo OpenCV?** No. La detección la hace YOLO. OpenCV solo abre la cámara, dibuja y estima el color, que es un paso adicional.

## Historia del proyecto: problemas encontrados y cómo se resolvieron

Esto es **muy valioso** para la presentación, porque muestra el proceso de ingeniería. Usa tus capturas del "antes" y del "después".

| Versión | Qué pasó en la cámara | Por qué | Solución |
|---|---|---|---|
| v1 | La cara salía como "círculo"; el techo, las esquinas y la mano como "cuadrado" o "triángulo" | El dataset solo tenía fondos lisos sintéticos; el modelo nunca vio una foto real | Fotos reales de fondo + **imágenes negativas** sin figuras |
| v1 | Las figuras lejos o pequeñas no se detectaban | La figura más pequeña del dataset medía ~70 px | 40 % de figuras pequeñas (desde 24 px) |
| v1 | Al mover la hoja se perdía la detección | El movimiento genera desenfoque que nunca se vio al entrenar | Desenfoque de movimiento al 25 % de las imágenes |
| v1 | Los recortes en la mano a veces no se detectaban | Nunca vio dedos tapando la figura ni sombra de cartulina | Dedos simulados y sombra de cartulina |
| v1 | El trazo negro salía como "gris" y el rosado como "azul" o "rojo" | Se promediaba la caja entera (casi todo papel) y no existía el rango rosado | Contar píxeles oscuros contra píxeles de color; agregar rosado, fucsia y celeste |
| v1 | El Acc_val daba 100 % pero la cámara fallaba | La validación era tan fácil (sintética) como el entrenamiento | La validación nueva incluye fotos reales y es más exigente: el Acc_val ahora es más realista |

## Estructura sugerida para la presentación (8–10 min)

1. **Objetivo** (1 min): detectar 3 figuras en tiempo real con YOLOv8.
2. **Dataset** (1.5 min): es sintético y generado por ustedes. Muestra `muestras_dataset.png`, explica las variaciones, el uso de COCO128 como fondo y como negativas, el split 70/20/10 y el formato YOLO.
3. **Arquitectura** (1.5 min): backbone, neck y head, transfer learning, por qué la versión nano.
4. **Entrenamiento** (1 min): parámetros y la gráfica `results.png` (las pérdidas bajan y el mAP sube).
5. **Resultados** (1.5 min): mAP@0.5, precisión, recall, matriz de confusión y Accuracy_val.
6. **Problemas y mejoras** (1 min): la tabla de arriba, con capturas del antes y el después.
7. **Demo en vivo** (2 min): hoja con figuras dibujadas, recortes de cartulina de colores, cerca, lejos, girada y en movimiento.
8. **Límites y trabajo futuro** (1 min).
9. **Conclusiones** (1 min).

## Para tus conclusiones: qué observar (escríbelas tú)

El enunciado dice que las conclusiones hechas con generadores de texto se califican con 1.0, así que aquí no hay conclusiones escritas. Hay preguntas para que las saques de **tus propios resultados**:

1. ¿En qué época se detuvo el entrenamiento por early stopping? ¿Qué muestra eso de la dificultad de la tarea?
2. En la matriz de confusión, ¿qué par de clases se confunde más? ¿Por qué crees que pasa (por ejemplo, un cuadrado rotado frente a un triángulo, o un círculo pequeño)?
3. ¿El Accuracy_val en validación se parece a lo que ves en la cámara? Si en la cámara funciona peor, ¿qué diferencias hay entre las imágenes sintéticas y las reales (luz, sombras, textura del papel)?
4. ¿Qué pasa con figuras muy pequeñas, muy lejos, cortadas por el borde o con poca luz?
5. ¿Cuántos FPS obtuviste en tu portátil? ¿Alcanza para tiempo real?
6. ¿La detección del color falla en algún caso (con luz amarilla, figura negra, sombras)?
7. Comparado con sliding window o con un clasificador simple, ¿qué ventaja tuvo YOLO?

**Ideas de límites y trabajo futuro para comentar:**
- Etiquetar fotos reales de las figuras (por ejemplo con Roboflow) y mezclarlas con las sintéticas.
- Seguimiento entre frames (tracking) para que la caja no parpadee.
- Agregar más figuras (estrella, pentágono).
- Probar YOLOv8s.
- Detectar el color como una clase del modelo.
- Exportar a ONNX para que corra más rápido.
