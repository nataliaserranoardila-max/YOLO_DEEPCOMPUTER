"""
generar_dataset.py
------------------
Genera un dataset SINTÉTICO de figuras geométricas (círculo, cuadrado, triángulo)
directamente en formato YOLO.

Por cada imagen se crea un .txt con una línea por figura:
    clase x_centro y_centro ancho alto     (todo normalizado entre 0 y 1)

Para que el modelo funcione después con la cámara, cada imagen varía:
  - fondo: hoja clara u oscura (con degradado y ruido) o FOTO REAL (COCO128)
  - número de figuras (1 a 6), posición, rotación y TAMAÑO (desde muy pequeñas,
    como una figura vista de lejos, hasta grandes)
  - color (cualquier tono, incluidos rosado y celeste, o negro)
  - estilo: contorno (a mano o recto) o rellena tipo cartulina recortada con sombra
  - dedos que tapan un borde de la figura (como cuando la sostienes con la mano)
  - perspectiva, desenfoque de movimiento, desenfoque normal y cambios de brillo
  - imágenes NEGATIVAS: fotos reales sin figuras, con .txt vacío, para que el modelo
    aprenda que caras, manos o techos NO son figuras (evita falsos positivos)

Uso:
    python data/generar_dataset.py --n 1200 --fondos coco128/images/train2017
"""
import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np

CLASES = ["circulo", "cuadrado", "triangulo"]

# Tonos de piel (BGR) para simular los dedos que sostienen la cartulina
PIEL = [(140, 170, 220), (110, 145, 200), (90, 120, 180), (160, 190, 235), (70, 100, 150)]


def color_aleatorio():
    """15 % negro/gris oscuro (marcador); el resto un tono cualquiera bien saturado."""
    if random.random() < 0.15:
        g = random.randint(10, 60)
        return (g, g, g)
    hsv = np.uint8([[[random.randint(0, 179), random.randint(120, 255), random.randint(120, 255)]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def fondo_aleatorio(tam):
    """Crea un fondo claro u oscuro, con degradado y ruido (parecido a una foto real)."""
    claro = random.random() < 0.75  # la mayoría con fondo claro (hoja de papel)
    base = random.randint(180, 250) if claro else random.randint(20, 80)
    img = np.full((tam, tam, 3), base, dtype=np.float32)

    # Degradado suave en una dirección al azar (simula iluminación desigual)
    g = np.linspace(-25, 25, tam, dtype=np.float32)
    img += g[None, :, None] if random.random() < 0.5 else g[:, None, None]

    # Leve tinte de color y ruido gaussiano
    img += np.array([random.randint(-15, 15) for _ in range(3)], dtype=np.float32)
    img += np.random.normal(0, random.uniform(2, 10), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def fondo_foto(fotos, tam):
    """Toma una foto real al azar, recorta un cuadrado y lo lleva a tam x tam."""
    img = cv2.imread(str(random.choice(fotos)))
    h, w = img.shape[:2]
    lado = random.randint(int(min(h, w) * 0.6), min(h, w))  # recorte aleatorio
    y0, x0 = random.randint(0, h - lado), random.randint(0, w - lado)
    return cv2.resize(img[y0:y0 + lado, x0:x0 + lado], (tam, tam))


def puntos_figura(clase, cx, cy, r, ang):
    """Devuelve los vértices de un cuadrado o triángulo centrado en (cx, cy) y rotado ang grados."""
    lados = 4 if clase == "cuadrado" else 3
    inicio = math.radians(ang) + (math.pi / 4 if lados == 4 else -math.pi / 2)
    pts = []
    for k in range(lados):
        a = inicio + 2 * math.pi * k / lados
        pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
    return np.array(pts, dtype=np.int32)


def contorno_a_mano(pts, temblor):
    """Agrega puntos intermedios con un pequeño temblor, para imitar una figura dibujada a mano."""
    nuevos = []
    for i in range(len(pts)):
        p, q = pts[i], pts[(i + 1) % len(pts)]
        for t in np.linspace(0, 1, 12, endpoint=False):
            punto = p + t * (q - p) + np.random.normal(0, temblor, 2)
            nuevos.append(punto)
    return np.array(nuevos, dtype=np.int32)


def dibujar_figura(img, clase, cx, cy, r, color, relleno):
    """Dibuja la figura. Devuelve su bounding box (x_min, y_min, x_max, y_max) y su contorno."""
    grosor = max(2, r // 15) + random.randint(0, 4)  # el trazo depende del tamaño
    if clase == "circulo":
        # El círculo se dibuja como polígono de muchos lados para poder "temblarlo"
        ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        pts = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)
    else:
        pts = puntos_figura(clase, cx, cy, r, random.uniform(0, 360)).astype(np.float64)
    # 30 % de las figuras sin relleno se ven "hechas a mano" (como con marcador en papel)
    if not relleno and random.random() < 0.3:
        pts = contorno_a_mano(pts, temblor=r * 0.02)
    pts = pts.astype(np.int32)

    if relleno:
        # 50 % parecen cartulina recortada: sombra suave detrás y borde un poco más oscuro
        if random.random() < 0.5:
            sombra = np.zeros(img.shape[:2], np.uint8)
            d = max(2, r // 12)
            cv2.fillPoly(sombra, [pts + [d, d]], 255)
            sombra = cv2.GaussianBlur(sombra, (0, 0), d) / 255.0 * 0.45
            img[:] = (img * (1 - sombra[..., None])).astype(np.uint8)
            cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)
            borde = tuple(int(c * 0.7) for c in color)
            cv2.polylines(img, [pts], True, borde, 1, lineType=cv2.LINE_AA)
        else:
            cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)
        m = 0
    else:
        cv2.polylines(img, [pts], True, color, grosor, lineType=cv2.LINE_AA)
        m = grosor // 2
    x, y, w, h = cv2.boundingRect(pts)
    return (x - m, y - m, x + w + m, y + h + m), pts


def distractores(img):
    """Dibuja líneas, garabatos y texto SIN etiqueta: así el modelo aprende que no todo es una figura."""
    tam = img.shape[0]
    for _ in range(random.randint(0, 3)):
        color = tuple(random.randint(0, 255) for _ in range(3))
        tipo = random.random()
        if tipo < 0.4:  # línea recta
            p1 = (random.randint(0, tam), random.randint(0, tam))
            p2 = (random.randint(0, tam), random.randint(0, tam))
            cv2.line(img, p1, p2, color, random.randint(1, 4), cv2.LINE_AA)
        elif tipo < 0.7:  # garabato
            x, y = random.randint(50, tam - 50), random.randint(50, tam - 50)
            pts = np.cumsum(np.random.randint(-15, 16, (10, 2)), axis=0) + [x, y]
            cv2.polylines(img, [pts.astype(np.int32)], False, color, 2, cv2.LINE_AA)
        else:  # texto
            cv2.putText(img, random.choice(["abc", "123", "USTA", "x=2"]),
                        (random.randint(0, tam - 100), random.randint(30, tam)),
                        cv2.FONT_HERSHEY_SIMPLEX, random.uniform(0.6, 1.4), color, 2)


def dedo(img, contorno, r):
    """Dibuja un 'dedo' (elipse color piel) tapando un borde de la figura.
    La etiqueta NO cambia: la caja sigue siendo la figura completa, como en la vida real."""
    px, py = contorno[random.randrange(len(contorno))]  # un punto del borde
    ancho, largo = max(4, int(r * random.uniform(0.25, 0.4))), max(8, int(r * random.uniform(0.6, 1.0)))
    cv2.ellipse(img, (int(px), int(py)), (ancho, largo), random.uniform(0, 180), 0, 360,
                random.choice(PIEL), -1, lineType=cv2.LINE_AA)


def desenfoque_movimiento(img):
    """Simula que la cámara o la hoja se movió: arrastra la imagen en una dirección."""
    k = random.randint(5, 15)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    return cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))


def perspectiva(img, etiquetas_px, tam):
    """Inclina la imagen como si la hoja estuviera vista en ángulo; transforma también las cajas."""
    d = tam * 0.08
    src = np.float32([[0, 0], [tam, 0], [tam, tam], [0, tam]])
    dst = src + np.random.uniform(-d, d, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    borde = tuple(int(v) for v in img[0, 0])
    img = cv2.warpPerspective(img, M, (tam, tam), borderMode=cv2.BORDER_CONSTANT, borderValue=borde)
    nuevas = []
    for clase_id, pts in etiquetas_px:
        p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
        x1, y1 = np.clip(p.min(axis=0), 0, tam)
        x2, y2 = np.clip(p.max(axis=0), 0, tam)
        nuevas.append((clase_id, x1, y1, x2, y2))
    return img, nuevas


def se_superpone(caja, cajas, margen=8):
    """True si la caja nueva choca con alguna ya dibujada (para que no se tapen)."""
    x1, y1, x2, y2 = caja
    for a1, b1, a2, b2 in cajas:
        if x1 < a2 + margen and x2 > a1 - margen and y1 < b2 + margen and y2 > b1 - margen:
            return True
    return False


def generar_imagen(fotos=None, tam=640):
    # Fondo: 55 % foto real (si hay fotos) y el resto fondo sintético tipo hoja
    usar_foto = bool(fotos) and random.random() < 0.55
    img = fondo_foto(fotos, tam) if usar_foto else fondo_aleatorio(tam)

    # Imagen NEGATIVA: 35 % de las fotos reales van sin ninguna figura y con .txt vacío.
    # Así el modelo aprende que una cara, una mano o un techo NO es un círculo ni un cuadrado.
    if usar_foto and random.random() < 0.35:
        if random.random() < 0.3:
            img = desenfoque_movimiento(img)
        return img, []

    distractores(img)
    cajas, figuras = [], []
    for _ in range(random.randint(1, 6)):
        for _intento in range(30):
            clase_id = random.randrange(len(CLASES))
            # "Radio" en píxeles: 40 % pequeñas (figura lejos) y 60 % medianas o grandes
            r = random.randint(12, 35) if random.random() < 0.4 else random.randint(35, 140)
            cx, cy = random.randint(r + 10, tam - r - 10), random.randint(r + 10, tam - r - 10)
            # Caja aproximada para revisar superposición antes de dibujar
            if se_superpone((cx - r, cy - r, cx + r, cy + r), cajas):
                continue
            color = color_aleatorio()
            relleno = random.random() < 0.6
            caja, contorno = dibujar_figura(img, CLASES[clase_id], cx, cy, r, color, relleno)
            if r > 25 and random.random() < 0.2:  # 20 %: un dedo tapa un borde
                dedo(img, contorno, r)
            cajas.append(caja)
            figuras.append((clase_id, contorno.astype(np.float32)))  # se guarda el contorno real
            break

    # 50 % de las imágenes se ven "en ángulo"; si no, las cajas quedan igual
    if random.random() < 0.5:
        img, finales = perspectiva(img, figuras, tam)
    else:
        finales = [(c, *np.clip(p.min(axis=0), 0, tam), *np.clip(p.max(axis=0), 0, tam)) for c, p in figuras]

    # Conversión a formato YOLO: centro y tamaño normalizados (0 a 1)
    etiquetas = []
    for clase_id, x1, y1, x2, y2 in finales:
        xc, yc = (x1 + x2) / 2 / tam, (y1 + y2) / 2 / tam
        w, h = (x2 - x1) / tam, (y2 - y1) / tam
        etiquetas.append(f"{clase_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # Imperfecciones de cámara: movimiento, desenfoque y brillo
    if random.random() < 0.25:
        img = desenfoque_movimiento(img)
    elif random.random() < 0.3:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.convertScaleAbs(img, alpha=random.uniform(0.85, 1.15), beta=random.randint(-20, 20))
    return img, etiquetas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="número total de imágenes")
    ap.add_argument("--salida", default="data/formas", help="carpeta del dataset")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--fondos", default=None, help="carpeta con fotos reales para usar de fondo (opcional)")
    args = ap.parse_args()

    random.seed(args.semilla)
    np.random.seed(args.semilla)
    fotos = sorted(Path(args.fondos).glob("*.jpg")) if args.fondos else []
    if args.fondos:
        print(f"Usando {len(fotos)} fotos reales como fondo")
    negativas = 0
    raiz = Path(args.salida)

    # División 70 / 20 / 10
    n_train, n_val = int(args.n * 0.7), int(args.n * 0.2)
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (args.n - n_train - n_val)

    conteo = {c: 0 for c in CLASES}
    for i, split in enumerate(splits):
        (raiz / "images" / split).mkdir(parents=True, exist_ok=True)
        (raiz / "labels" / split).mkdir(parents=True, exist_ok=True)
        img, etiquetas = generar_imagen(fotos)
        cv2.imwrite(str(raiz / "images" / split / f"img_{i:04d}.jpg"), img)
        (raiz / "labels" / split / f"img_{i:04d}.txt").write_text("\n".join(etiquetas))
        negativas += not etiquetas
        for e in etiquetas:
            conteo[CLASES[int(e.split()[0])]] += 1

    # Archivo de configuración que YOLO necesita para entrenar
    (raiz / "data.yaml").write_text(
        f"path: {raiz.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASES)}\nnames: {CLASES}\n"
    )
    print(f"Dataset listo en {raiz}: {n_train} train / {n_val} val / {args.n - n_train - n_val} test")
    print("Figuras por clase:", conteo)
    print("Imágenes negativas (sin figuras):", negativas)


if __name__ == "__main__":
    main()
