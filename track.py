import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchreid
from PIL import Image
from ultralytics import YOLO

from config import (
    B_LIMIT_RATIO,
    DIVERSITY_THRESHOLD,
    EMB_UPDATE_INTERVAL_SEC,
    GLOBAL_LOST_LIMIT,
    HIST_THRESHOLD_GLOBAL,
    HIST_THRESHOLD_LOCAL,
    HIST_UPDATE_ALPHA,
    LOCAL_LOST_LIMIT,
    L_LIMIT_RATIO,
    MIN_REID_BOX_AREA,
    MIN_REID_BOX_H,
    MIN_REID_BOX_W,
    MOVE_COOLDOWN,
    ONVIF_HOST,
    ONVIF_PASSWORD,
    ONVIF_PORT,
    ONVIF_USER,
    PTZ_PAN_SPEED,
    PTZ_TILT_SPEED,
    R_LIMIT_RATIO,
    RECENT_ADD_THRESHOLD,
    RECENT_BANK_SIZE,
    REID_MATCH_THRESHOLD_GLOBAL,
    REID_MATCH_THRESHOLD_LOCAL,
    REID_MODEL_NAME,
    REID_STRONG_THRESHOLD,
    RTSP_SOURCE,
    SEARCH_RADIUS,
    SIZE_TOLERANCE,
    STABLE_ADD_THRESHOLD,
    STABLE_BANK_SIZE,
    T_LIMIT_RATIO,
    YOLO_CLASSES,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_MODEL_PATH,
    YOLO_TRACKER_CFG,
)

try:
    from onvif import ONVIFCamera as ONVIFDevice
except ImportError:
    from onvif import ONVIFClient as ONVIFDevice
from onvif.exceptions import ONVIFError

# =========================
# Variables globales
# =========================
target_id = -1
target_locked = False

last_known_pos = None
last_box_dim = None
target_hist = None

current_boxes = []
current_frame_img = None

lost_frames = 0
frame_idx = 0

# Memoria de identidad
target_embeddings_stable = []
target_embeddings_recent = []
last_embedding_update_time = 0.0

last_move_time = 0.0
last_move_cmd = "STOP"
ptz_move_mode = "continuous"
ptz_enabled = True
ptz_limits = {
    "x_min": -1.0,
    "x_max": 1.0,
    "y_min": -1.0,
    "y_max": 1.0,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Modelo ReID
# =========================
print(f"Cargando ReID: {REID_MODEL_NAME} en {DEVICE}")
reid_model = torchreid.models.build_model(
    name=REID_MODEL_NAME,
    num_classes=1000,
    pretrained=True,
)
reid_model.eval().to(DEVICE)

transform = torchreid.data.transforms.build_transforms(
    height=256,
    width=128,
    is_train=False,
)[1]


# =========================
# Utilidades HSV
# =========================
def get_crop(image, box):
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def extract_histogram(image, box):
    crop = get_crop(image, box)
    if crop is None:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))

    hist = cv2.calcHist([hsv], [0, 1], mask, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def compare_histograms(hist1, hist2):
    if hist1 is None or hist2 is None:
        return 1.0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def update_target_hist(old_hist, new_hist, alpha=HIST_UPDATE_ALPHA):
    if old_hist is None:
        return new_hist
    if new_hist is None:
        return old_hist

    updated = (1 - alpha) * old_hist + alpha * new_hist
    cv2.normalize(updated, updated)
    return updated


# =========================
# Utilidades ReID
# =========================
def shrink_box(x1, y1, x2, y2, factor=0.08):
    w = x2 - x1
    h = y2 - y1
    dx = int(w * factor)
    dy = int(h * factor)

    nx1 = x1 + dx
    ny1 = y1 + dy
    nx2 = x2 - dx
    ny2 = y2 - dy

    if nx2 <= nx1 or ny2 <= ny1:
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2


def valid_box_for_reid(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    area = w * h
    return w >= MIN_REID_BOX_W and h >= MIN_REID_BOX_H and area >= MIN_REID_BOX_AREA


def extract_embedding(image, box):
    if not valid_box_for_reid(box):
        return None

    x1, y1, x2, y2 = shrink_box(*box)
    crop = get_crop(image, (x1, y1, x2, y2))
    if crop is None:
        return None

    try:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        tensor = transform(crop_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = reid_model(tensor)
            feat = F.normalize(feat, p=2, dim=1)

        return feat.squeeze(0).cpu()
    except Exception as e:
        print(f"Error extrayendo embedding: {e}")
        return None


def cosine_sim(a, b):
    return torch.dot(a, b).item()


def get_all_embeddings():
    return target_embeddings_stable + target_embeddings_recent


def best_similarity_to_memory(emb, bank):
    if emb is None or len(bank) == 0:
        return -1.0, -1.0

    sims = [cosine_sim(emb, ref) for ref in bank]
    sims = sorted(sims, reverse=True)
    topk = sims[: min(5, len(sims))]
    return max(sims), float(np.mean(topk))


def best_similarity_to_all_memories(emb):
    stable_max, stable_avg = best_similarity_to_memory(emb, target_embeddings_stable)
    recent_max, recent_avg = best_similarity_to_memory(emb, target_embeddings_recent)

    best_max = max(stable_max, recent_max)
    avgs = [x for x in [stable_avg, recent_avg] if x >= 0]
    best_avg = np.mean(avgs) if len(avgs) > 0 else -1.0

    return best_max, float(best_avg)


def is_diverse_enough(emb, bank, threshold=DIVERSITY_THRESHOLD):
    if emb is None:
        return False
    if len(bank) == 0:
        return True

    max_sim, _ = best_similarity_to_memory(emb, bank)
    return max_sim < threshold


def add_embedding_stable(emb):
    global target_embeddings_stable

    if emb is None:
        return

    if not is_diverse_enough(emb, target_embeddings_stable):
        return

    target_embeddings_stable.append(emb)
    if len(target_embeddings_stable) > STABLE_BANK_SIZE:
        target_embeddings_stable = target_embeddings_stable[-STABLE_BANK_SIZE:]


def add_embedding_recent(emb):
    global target_embeddings_recent

    if emb is None:
        return

    if not is_diverse_enough(emb, target_embeddings_recent):
        return

    target_embeddings_recent.append(emb)
    if len(target_embeddings_recent) > RECENT_BANK_SIZE:
        target_embeddings_recent = target_embeddings_recent[-RECENT_BANK_SIZE:]


def maybe_update_embedding_memory(image, box):
    global last_embedding_update_time

    now = time.time()
    if now - last_embedding_update_time < EMB_UPDATE_INTERVAL_SEC:
        return None, -1.0

    emb = extract_embedding(image, box)
    if emb is None:
        return None, -1.0

    best_sim, avg_top = best_similarity_to_all_memories(emb)

    if best_sim >= RECENT_ADD_THRESHOLD:
        add_embedding_recent(emb)

    if best_sim >= STABLE_ADD_THRESHOLD:
        add_embedding_stable(emb)

    last_embedding_update_time = now
    return emb, best_sim


def verify_identity(image, box, threshold):
    emb = extract_embedding(image, box)
    if emb is None or len(get_all_embeddings()) == 0:
        return False, -1.0, None

    best_sim, avg_top = best_similarity_to_all_memories(emb)
    accepted = best_sim >= threshold
    return accepted, best_sim, emb


# =========================
# Candidatos
# =========================
def compute_candidate_score(box_data, img, use_position=True):
    global last_known_pos, last_box_dim, target_hist

    x1, y1, x2, y2, tid = box_data
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w, h = x2 - x1, y2 - y1

    cand_hist = extract_histogram(img, (x1, y1, x2, y2))
    hist_dist = compare_histograms(target_hist, cand_hist)

    if last_box_dim is not None:
        w_diff = abs(w - last_box_dim[0]) / max(last_box_dim[0], 1)
        h_diff = abs(h - last_box_dim[1]) / max(last_box_dim[1], 1)
        size_penalty = w_diff + h_diff
    else:
        size_penalty = 0

    if use_position and last_known_pos is not None:
        dist = np.sqrt((cx - last_known_pos[0]) ** 2 + (cy - last_known_pos[1]) ** 2)
    else:
        dist = 0

    return {
        "tid": tid,
        "box": (x1, y1, x2, y2),
        "center": (cx, cy),
        "size": (w, h),
        "hist": cand_hist,
        "hist_dist": hist_dist,
        "dist": dist,
        "size_penalty": size_penalty,
    }


def get_profile_token(profile):
    return getattr(profile, "token", None) or getattr(profile, "_token", None)


def resolve_ptz_limits(ptz, profile):
    limits = {
        "x_min": -1.0,
        "x_max": 1.0,
        "y_min": -1.0,
        "y_max": 1.0,
    }

    ptz_config = getattr(profile, "PTZConfiguration", None)
    config_token = get_profile_token(ptz_config) if ptz_config is not None else None
    if not config_token:
        return limits

    try:
        options = ptz.GetConfigurationOptions({"ConfigurationToken": config_token})
    except ONVIFError as exc:
        print(f"No se pudieron leer limites PTZ, usando defaults: {exc}")
        return limits
    spaces = getattr(options, "Spaces", None)
    cont_spaces = (
        getattr(spaces, "ContinuousPanTiltVelocitySpace", None) if spaces else None
    )

    if not cont_spaces:
        return limits

    pan_tilt_space = cont_spaces[0]
    x_range = getattr(pan_tilt_space, "XRange", None)
    y_range = getattr(pan_tilt_space, "YRange", None)

    if x_range:
        limits["x_min"] = float(getattr(x_range, "Min", limits["x_min"]))
        limits["x_max"] = float(getattr(x_range, "Max", limits["x_max"]))
    if y_range:
        limits["y_min"] = float(getattr(y_range, "Min", limits["y_min"]))
        limits["y_max"] = float(getattr(y_range, "Max", limits["y_max"]))

    return limits


def create_onvif_services(host, port, user, password):
    client = ONVIFDevice(host, port, user, password)

    if hasattr(client, "create_devicemgmt_service"):
        devicemgmt = client.create_devicemgmt_service()
        media = client.create_media_service()
        ptz = client.create_ptz_service()
    else:
        devicemgmt = client.devicemgmt()
        media = client.media()
        ptz = client.ptz()

    profiles = media.GetProfiles()
    if not profiles:
        raise RuntimeError("No se encontraron perfiles ONVIF.")

    selected_profile = profiles[0]
    profile_token = get_profile_token(selected_profile)
    if not profile_token:
        raise RuntimeError("No se pudo obtener ProfileToken ONVIF.")

    limits = resolve_ptz_limits(ptz, selected_profile)
    return devicemgmt, ptz, profile_token, limits


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, float(value)))


def ptz_move(ptz, profile_token, pan, tilt):
    global ptz_move_mode

    pan = clamp(pan, ptz_limits["x_min"], ptz_limits["x_max"])
    tilt = clamp(tilt, ptz_limits["y_min"], ptz_limits["y_max"])

    if ptz_move_mode == "continuous":
        try:
            ptz.ContinuousMove(
                {
                    "ProfileToken": profile_token,
                    "Velocity": {"PanTilt": {"x": pan, "y": tilt}},
                    "Timeout": "PT0.3S",
                }
            )
            return
        except ONVIFError as exc:
            print(
                "ContinuousMove fallo; cambiando a RelativeMove. "
                f"Detalle: {exc}"
            )
            ptz_move_mode = "relative"

    ptz.RelativeMove(
        {
            "ProfileToken": profile_token,
            "Translation": {"PanTilt": {"x": pan * 0.05, "y": tilt * 0.05}},
            "Speed": {"PanTilt": {"x": abs(pan), "y": abs(tilt)}},
        }
    )


def ptz_stop(ptz, profile_token):
    ptz.Stop({"ProfileToken": profile_token, "PanTilt": True, "Zoom": True})


def compute_ptz_command(center, frame_shape):
    cx, cy = center
    frame_h, frame_w = frame_shape[:2]

    left_limit = int(frame_w * L_LIMIT_RATIO)
    right_limit = int(frame_w * R_LIMIT_RATIO)
    top_limit = int(frame_h * T_LIMIT_RATIO)
    bottom_limit = int(frame_h * B_LIMIT_RATIO)

    pan = 0.0
    tilt = 0.0
    labels = []

    if cx < left_limit:
        pan = -PTZ_PAN_SPEED
        labels.append("LEFT")
    elif cx > right_limit:
        pan = PTZ_PAN_SPEED
        labels.append("RIGHT")

    if cy < top_limit:
        tilt = PTZ_TILT_SPEED
        labels.append("UP")
    elif cy > bottom_limit:
        tilt = -PTZ_TILT_SPEED
        labels.append("DOWN")

    cmd = "+".join(labels) if labels else "STOP"
    dead_zone = (left_limit, top_limit, right_limit, bottom_limit)
    return cmd, pan, tilt, dead_zone


# =========================
# Selección manual
# =========================
def select_target(event, x, y, flags, param):
    global target_id, target_locked, current_boxes
    global last_known_pos, last_box_dim, target_hist
    global current_frame_img, lost_frames
    global \
        target_embeddings_stable, \
        target_embeddings_recent, \
        last_embedding_update_time

    if event == cv2.EVENT_LBUTTONDBLCLK:
        found = False

        for box_data in current_boxes:
            x1, y1, x2, y2, tid = box_data
            if x1 <= x <= x2 and y1 <= y <= y2:
                box = (x1, y1, x2, y2)

                target_id = tid
                target_locked = True
                last_known_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
                last_box_dim = (x2 - x1, y2 - y1)
                target_hist = extract_histogram(current_frame_img, box)
                lost_frames = 0

                target_embeddings_stable = []
                target_embeddings_recent = []

                emb = extract_embedding(current_frame_img, box)
                if emb is not None:
                    add_embedding_stable(emb)
                    add_embedding_recent(emb)
                    last_embedding_update_time = time.time()

                print(f"--- Sujeto fijado manualmente: ID {target_id} ---")
                found = True
                break

        if not found:
            target_id = -1
            target_locked = False
            last_known_pos = None
            last_box_dim = None
            target_hist = None
            lost_frames = 0
            target_embeddings_stable = []
            target_embeddings_recent = []
            last_embedding_update_time = 0.0
            print("--- Filtro quitado ---")


# =========================
# Main
# =========================
def main():
    global target_id, target_locked, current_boxes, current_frame_img
    global last_known_pos, last_box_dim, target_hist, lost_frames
    global frame_idx, last_move_time, last_move_cmd, ptz_limits, ptz_enabled

    model = YOLO(YOLO_MODEL_PATH)
    devicemgmt, ptz, profile_token, ptz_limits = create_onvif_services(
        ONVIF_HOST, ONVIF_PORT, ONVIF_USER, ONVIF_PASSWORD
    )
    device_info = devicemgmt.GetDeviceInformation()
    print(f"ONVIF conectado: {device_info.Manufacturer} {device_info.Model}")

    cv2.namedWindow("Pro Face Tracker + Adaptive Identity Memory")
    cv2.setMouseCallback("Pro Face Tracker + Adaptive Identity Memory", select_target)

    results = model.track(
        source=RTSP_SOURCE,
        persist=True,
        stream=True,
        classes=YOLO_CLASSES,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        tracker=YOLO_TRACKER_CFG,
    )

    for frame_result in results:
        frame_idx += 1

        img = frame_result.orig_img.copy()
        current_frame_img = img.copy()
        current_boxes = []

        detections = []

        if frame_result.boxes is not None and len(frame_result.boxes) > 0:
            boxes = frame_result.boxes.xyxy.cpu().numpy()

            if frame_result.boxes.id is not None:
                ids = frame_result.boxes.id.cpu().numpy().astype(int)
            else:
                ids = np.arange(len(boxes))

            for i, tid in enumerate(ids):
                x1, y1, x2, y2 = map(int, boxes[i])
                current_boxes.append((x1, y1, x2, y2, tid))
                detections.append((x1, y1, x2, y2, tid))

        target_found = False
        active_box = None
        active_center = None
        last_sim_text = ""

        # =========================
        # 1) Intentar seguir mismo ID
        # =========================
        if target_locked and target_id != -1:
            for det in detections:
                x1, y1, x2, y2, tid = det
                if tid == target_id:
                    box = (x1, y1, x2, y2)

                    last_known_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
                    last_box_dim = (x2 - x1, y2 - y1)

                    new_hist = extract_histogram(img, box)
                    target_hist = update_target_hist(target_hist, new_hist)

                    target_found = True
                    active_box = box
                    active_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    lost_frames = 0

                    # Actualización adaptativa por tiempo
                    emb, sim = maybe_update_embedding_memory(img, box)
                    if sim >= 0:
                        last_sim_text = f"ReID sim: {sim:.3f}"

                    # Verificación ocasional débil: si da muy mal, desconfiar
                    if len(get_all_embeddings()) > 0 and frame_idx % 12 == 0:
                        accepted, sim_verify, emb_verify = verify_identity(
                            img, box, threshold=0.55
                        )
                        last_sim_text = (
                            f"ReID sim: {sim_verify:.3f}"
                            if sim_verify >= 0
                            else last_sim_text
                        )

                        if not accepted:
                            target_found = False
                            active_box = None
                            active_center = None

                    break

        # =========================
        # 2) Si no está: modo perdido
        # =========================
        if target_locked and not target_found:
            lost_frames += 1

            # --------------------------------
            # 2A) Pérdida corta: búsqueda local
            # --------------------------------
            if lost_frames <= LOCAL_LOST_LIMIT and last_known_pos is not None:
                best = None
                best_score = float("inf")

                for det in detections:
                    candidate = compute_candidate_score(det, img, use_position=True)

                    if candidate["dist"] > SEARCH_RADIUS:
                        continue
                    if candidate["hist_dist"] > HIST_THRESHOLD_LOCAL:
                        continue
                    if candidate["size_penalty"] > SIZE_TOLERANCE * 2:
                        continue

                    score = (
                        candidate["dist"] * 0.5
                        + candidate["hist_dist"] * 120
                        + candidate["size_penalty"] * 40
                    )

                    if score < best_score:
                        best_score = score
                        best = candidate

                if best is not None:
                    accepted, sim, emb = verify_identity(
                        img,
                        best["box"],
                        threshold=REID_MATCH_THRESHOLD_LOCAL,
                    )
                    last_sim_text = (
                        f"ReID sim: {sim:.3f}" if sim >= 0 else "ReID sim: N/A"
                    )

                    if accepted:
                        target_id = best["tid"]
                        last_known_pos = best["center"]
                        last_box_dim = best["size"]
                        target_hist = update_target_hist(target_hist, best["hist"])
                        target_found = True
                        active_box = best["box"]
                        active_center = best["center"]
                        lost_frames = 0

                        if emb is not None:
                            if sim >= RECENT_ADD_THRESHOLD:
                                add_embedding_recent(emb)
                            if sim >= STABLE_ADD_THRESHOLD:
                                add_embedding_stable(emb)

                        print(
                            f"--- Recuperación local confirmada por embedding: ID {target_id} ---"
                        )

            # --------------------------------
            # 2B) Pérdida larga / reentrada: búsqueda global
            # --------------------------------
            elif lost_frames <= GLOBAL_LOST_LIMIT:
                best = None
                best_score = float("inf")

                for det in detections:
                    candidate = compute_candidate_score(det, img, use_position=False)

                    if candidate["hist_dist"] > HIST_THRESHOLD_GLOBAL:
                        continue
                    if candidate["size_penalty"] > 2.5:
                        continue

                    score = (
                        candidate["hist_dist"] * 150 + candidate["size_penalty"] * 20
                    )

                    if score < best_score:
                        best_score = score
                        best = candidate

                if best is not None:
                    accepted, sim, emb = verify_identity(
                        img,
                        best["box"],
                        threshold=REID_MATCH_THRESHOLD_GLOBAL,
                    )
                    last_sim_text = (
                        f"ReID sim: {sim:.3f}" if sim >= 0 else "ReID sim: N/A"
                    )

                    if accepted:
                        target_id = best["tid"]
                        last_known_pos = best["center"]
                        last_box_dim = best["size"]
                        target_hist = update_target_hist(target_hist, best["hist"])
                        target_found = True
                        active_box = best["box"]
                        active_center = best["center"]
                        lost_frames = 0

                        if emb is not None:
                            if sim >= RECENT_ADD_THRESHOLD:
                                add_embedding_recent(emb)
                            if sim >= STABLE_ADD_THRESHOLD:
                                add_embedding_stable(emb)

                        print(
                            f"--- Reidentificado globalmente por embedding: ID {target_id} ---"
                        )

                # Fallback: búsqueda global solo por memoria de identidad
                if not target_found and len(get_all_embeddings()) > 0:
                    best_det = None
                    best_sim = -1.0
                    best_emb = None

                    for det in detections:
                        x1, y1, x2, y2, tid = det
                        emb = extract_embedding(img, (x1, y1, x2, y2))
                        if emb is None:
                            continue

                        sim, avg_top = best_similarity_to_all_memories(emb)
                        if sim > best_sim:
                            best_sim = sim
                            best_det = det
                            best_emb = emb

                    last_sim_text = (
                        f"ReID sim: {best_sim:.3f}"
                        if best_sim >= 0
                        else "ReID sim: N/A"
                    )

                    if best_det is not None and best_sim >= REID_MATCH_THRESHOLD_GLOBAL:
                        x1, y1, x2, y2, tid = best_det
                        target_id = tid
                        target_found = True
                        active_box = (x1, y1, x2, y2)
                        active_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        last_known_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
                        last_box_dim = (x2 - x1, y2 - y1)
                        target_hist = extract_histogram(img, (x1, y1, x2, y2))
                        lost_frames = 0

                        if best_emb is not None:
                            if best_sim >= RECENT_ADD_THRESHOLD:
                                add_embedding_recent(best_emb)
                            if best_sim >= STABLE_ADD_THRESHOLD:
                                add_embedding_stable(best_emb)

                        print(
                            f"--- Reidentificado global por memoria adaptativa: ID {target_id} ---"
                        )

        # =========================
        # Dibujo
        # =========================
        for x1, y1, x2, y2, tid in current_boxes:
            if target_locked and target_found and tid == target_id:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 3)
                cv2.putText(
                    img,
                    "CONFERENCISTA",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )
            elif not target_locked:
                cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)

        if target_locked:
            status = "TRACKING" if target_found else f"LOST ({lost_frames})"
            color = (0, 255, 0) if target_found else (0, 0, 255)

            cv2.putText(
                img,
                f"Estado: {status}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            cv2.putText(
                img,
                f"Target ID: {target_id}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv2.putText(
                img,
                f"Stable emb: {len(target_embeddings_stable)}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv2.putText(
                img,
                f"Recent emb: {len(target_embeddings_recent)}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            if last_sim_text:
                cv2.putText(
                    img,
                    last_sim_text,
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        frame_h, frame_w = img.shape[:2]
        left_limit = int(frame_w * L_LIMIT_RATIO)
        right_limit = int(frame_w * R_LIMIT_RATIO)
        top_limit = int(frame_h * T_LIMIT_RATIO)
        bottom_limit = int(frame_h * B_LIMIT_RATIO)
        cv2.rectangle(
            img,
            (left_limit, top_limit),
            (right_limit, bottom_limit),
            (255, 200, 0),
            2,
        )

        ptz_text = "PTZ: STOP"
        now = time.time()
        if not ptz_enabled:
            ptz_text = "PTZ: DISABLED"

        if ptz_enabled and target_locked and target_found and active_center is not None:
            cv2.circle(img, active_center, 5, (0, 255, 255), -1)
            cmd, pan, tilt, _ = compute_ptz_command(active_center, img.shape)
            ptz_text = f"PTZ: {cmd}"

            if now - last_move_time >= MOVE_COOLDOWN and cmd != last_move_cmd:
                try:
                    if cmd == "STOP":
                        ptz_stop(ptz, profile_token)
                    else:
                        ptz_move(ptz, profile_token, pan, tilt)
                    last_move_time = now
                    last_move_cmd = cmd
                except ONVIFError as exc:
                    ptz_enabled = False
                    ptz_text = f"PTZ ERROR: {exc}"
                    print(f"PTZ deshabilitado por error ONVIF: {exc}")
        elif ptz_enabled and now - last_move_time >= MOVE_COOLDOWN and last_move_cmd != "STOP":
            try:
                ptz_stop(ptz, profile_token)
                last_move_time = now
                last_move_cmd = "STOP"
            except ONVIFError as exc:
                ptz_enabled = False
                ptz_text = f"PTZ ERROR: {exc}"
                print(f"PTZ deshabilitado por error ONVIF: {exc}")

        cv2.putText(
            img,
            ptz_text,
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
        )

        cv2.imshow("Pro Face Tracker + Adaptive Identity Memory", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if ptz_enabled and last_move_cmd != "STOP":
        try:
            ptz_stop(ptz, profile_token)
        except ONVIFError as exc:
            print(f"No se pudo detener PTZ al salir: {exc}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
