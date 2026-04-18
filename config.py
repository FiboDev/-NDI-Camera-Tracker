"""Configuracion centralizada del Face Tracker.

Este archivo concentra constantes y valores de entorno para evitar "magic
numbers" repartidos por el proyecto.
"""

import os

# =========================
# Tracking / reidentificacion
# =========================
SEARCH_RADIUS = 180
SIZE_TOLERANCE = 0.6
LOCAL_LOST_LIMIT = 20
GLOBAL_LOST_LIMIT = 300

HIST_THRESHOLD_LOCAL = 0.55
HIST_THRESHOLD_GLOBAL = 0.40
HIST_UPDATE_ALPHA = 0.15

# =========================
# PTZ: zona de confort y movimiento
# =========================
L_LIMIT_RATIO = 0.38
R_LIMIT_RATIO = 0.62
T_LIMIT_RATIO = 0.35
B_LIMIT_RATIO = 0.65
MOVE_COOLDOWN = 0.4
PTZ_PAN_SPEED = 0.20
PTZ_TILT_SPEED = 0.12

# =========================
# ONVIF / stream (con override por variables de entorno)
# =========================
ONVIF_HOST = os.getenv("ONVIF_HOST", "192.168.1.18")
ONVIF_PORT = int(os.getenv("ONVIF_PORT", "80"))
ONVIF_USER = os.getenv("ONVIF_USER", "admin")
ONVIF_PASSWORD = os.getenv("ONVIF_PASSWORD", "admin")

RTSP_SOURCE = os.getenv("RTSP_SOURCE", "rtsp://192.168.1.18:554/live/av0")

# =========================
# Modelo detector y tracker
# =========================
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11m.pt")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "720"))
YOLO_CLASSES = [0]
YOLO_TRACKER_CFG = os.getenv("YOLO_TRACKER_CFG", "botsort.yaml")

# =========================
# ReID
# =========================
REID_MODEL_NAME = "osnet_x0_75"
REID_MATCH_THRESHOLD_LOCAL = 0.72
REID_MATCH_THRESHOLD_GLOBAL = 0.78
REID_STRONG_THRESHOLD = 0.82

MIN_REID_BOX_W = 40
MIN_REID_BOX_H = 80
MIN_REID_BOX_AREA = 40 * 80

# Memoria adaptativa
STABLE_BANK_SIZE = 6
RECENT_BANK_SIZE = 15
EMB_UPDATE_INTERVAL_SEC = 1.0
RECENT_ADD_THRESHOLD = 0.75
STABLE_ADD_THRESHOLD = 0.86
DIVERSITY_THRESHOLD = 0.95
