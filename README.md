# PTZ Camera Tracker

Seguimiento automático de una cámara PTZ con visión por computador: la cámara sigue a la persona que está predicando, en tiempo real y sin operador.

Se usa **todos los domingos** en las transmisiones en vivo de mi iglesia, con una cámara PTZ Avkans controlada por ONVIF.

<!-- GIF PRONTO -->

## El problema

- La iglesia no tenía un operador dedicado a la cámara, así que la producción quedaba con un plano fijo o con alguien moviéndola a mano.
- En la tarima suele haber **varias personas a la vez** (músicos, pastores, quien presenta), y la cámara debe seguir solo a una.
- La **iluminación cambia de forma drástica** durante el servicio (pantalla LED, luces de escenario, escenas oscuras). Eso rompe los trackers que dependen solo de la posición o del color.

## Cómo funciona

```
RTSP de la cámara
      │
      ▼
YOLO11 (solo personas) + BoT-SORT ── IDs por frame
      │
      ▼
Selección del objetivo (clic en la ventana)
      │
      ▼
Mantener la identidad
  1. Mismo ID de BoT-SORT  → se sigue directo
  2. Pérdida corta         → búsqueda local por posición, tamaño e histograma HSV
  3. Pérdida larga         → búsqueda global en todo el cuadro
  Toda reentrada se valida con embeddings ReID (OSNet) contra la memoria de identidad
      │
      ▼
Control PTZ por ONVIF
  Zona de confort en el centro del cuadro; fuera de ella → ContinuousMove
  (con RelativeMove como respaldo si la cámara no lo soporta) y un cooldown entre comandos
```

### Memoria adaptativa de identidad

El punto central del proyecto: **no perder a la persona cuando la luz cambia** y **no saltar a otra persona** cuando se cruzan.

- **Embeddings ReID (OSNet x0.75, torchreid):** describen la apariencia de la persona y son mucho más estables ante cambios de luz que un histograma de color.
- **Dos bancos de embeddings:**
  - *Estable* (hasta 6): solo entran muestras con similitud muy alta (≥ 0.86). Es el ancla de quién es la persona.
  - *Reciente* (hasta 15): entra con similitud ≥ 0.75. Se adapta a cómo se ve la persona ahora (otra luz, otra pose).
- **Filtro de diversidad:** una muestra casi idéntica a otra que ya está guardada (similitud > 0.95) se descarta, para que la memoria cubra apariencias distintas en lugar de repetir la misma.
- **Umbrales distintos según el contexto:** la reentrada local exige 0.72, la global 0.78 y 0.82 se trata como coincidencia fuerte. Volver a encontrarla en otro lado del cuadro exige más evidencia.
- **Histograma HSV** como filtro barato y rápido antes de calcular embeddings.

## Evolución del proyecto

| Versión | Archivo | Enfoque | Qué falló / qué mejoró |
|---|---|---|---|
| 1 | `main.py` | YOLO + histograma HSV + posición | Se perdía con los cambios de luz y saltaba a otras personas con ropa parecida |
| 2 | `media.py` | + ReID (OSNet) con un solo banco de embeddings + control PTZ por ONVIF | Más robusto, pero la memoria se "contaminaba" o se quedaba vieja |
| 3 | `track.py` | Memoria adaptativa (estable + reciente), filtro de diversidad, configuración centralizada | La versión en producción: la más consistente ante luz cambiante y varias personas en tarima |

`test.py` es un banco de pruebas para probar variantes sobre video grabado (`videos/predica.mp4`) sin mover la cámara.

Probé varios modelos de detección; Yolo11n, Yolo10, Yolo26. me quedé con **YOLO11m** a `imgsz=720` porque fue el que mantuvo el seguimiento de forma más consistente en condiciones reales y más ligero para un equipo de hardware medio en una iglesia.

## Requisitos

- Python 3.12 y [uv](https://github.com/astral-sh/uv)
- GPU NVIDIA con CUDA 12.6 (también corre en CPU, pero más lento)
- Una cámara PTZ con ONVIF y salida RTSP

## Instalación y uso

```bash
uv sync
```

Configura la cámara con variables de entorno:

```bash
export ONVIF_HOST=192.168.1.18
export ONVIF_PORT=80
export ONVIF_USER=<usuario>
export ONVIF_PASSWORD=<contraseña>
export RTSP_SOURCE=rtsp://192.168.1.18:554/live/av0
# opcionales
export YOLO_MODEL_PATH=yolo11m.pt
export YOLO_CONF=0.25
export YOLO_IMGSZ=720
```

Ejecuta:

```bash
uv run track.py
```

- **Clic** sobre una persona: la selecciona como objetivo.
- **q**: salir.

Todos los parámetros (umbrales, tamaños de los bancos, zona de confort, velocidades PTZ) están en `config.py`.

## Limitaciones y próximos pasos

- **Sin métricas formales todavía.** La versión final se eligió por consistencia observada en servicios reales. El siguiente paso es anotar `videos/predica.mp4` y medir el porcentaje del tiempo con el objetivo en cuadro y los cambios de identidad, comparando las tres versiones.
- La selección inicial del objetivo es manual (clic).
- El control PTZ es por zona de confort con velocidad fija; un controlador proporcional suavizaría el movimiento.
- No hay zoom automático según la distancia de la persona.

## Stack

YOLO11 (Ultralytics) · BoT-SORT · OSNet / torchreid · PyTorch · OpenCV · ONVIF (onvif-zeep) · uv
