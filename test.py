import cv2
import numpy as np
from ultralytics import YOLO

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

# =========================
# Parámetros
# =========================
SEARCH_RADIUS = 180
SIZE_TOLERANCE = 0.6
LOCAL_LOST_LIMIT = 20
GLOBAL_LOST_LIMIT = 300
HIST_THRESHOLD_LOCAL = 0.55
HIST_THRESHOLD_GLOBAL = 0.40
HIST_UPDATE_ALPHA = 0.15


# =========================
# Utilidades HSV / Hist
# =========================
def get_crop(image, box):
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

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

    # máscara para ignorar píxeles poco informativos
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


def draw_histogram_2d(hist, scale=10):
    if hist is None:
        return np.zeros((320, 320, 3), dtype=np.uint8)

    hist_norm = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX)
    hist_norm = hist_norm.astype(np.uint8)

    hist_vis = cv2.resize(
        hist_norm,
        (hist.shape[1] * scale, hist.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    hist_vis = cv2.applyColorMap(hist_vis, cv2.COLORMAP_JET)
    return hist_vis


def resize_keep_height(img, target_h):
    ih, iw = img.shape[:2]
    if ih == 0:
        return img
    new_w = int(iw * (target_h / ih))
    return cv2.resize(img, (new_w, target_h))


def build_hsv_debug_view(image, box):
    crop = get_crop(image, box)
    if crop is None:
        return np.zeros((300, 600, 3), dtype=np.uint8)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    mask = cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    hist = extract_histogram(image, box)
    hist_vis = draw_histogram_2d(hist, scale=10)

    target_h = 220
    crop_r = resize_keep_height(crop, target_h)
    hsv_r = resize_keep_height(hsv_bgr, target_h)
    mask_r = resize_keep_height(mask_bgr, target_h)

    top = np.hstack([crop_r, hsv_r, mask_r])

    label_band = np.zeros((30, top.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        label_band,
        "Original | HSV->BGR | Mascara HSV",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    hist_vis = cv2.resize(hist_vis, (top.shape[1], 220))

    debug_view = np.vstack([label_band, top, hist_vis])
    return debug_view


# =========================
# Scoring de candidatos
# =========================
def compute_candidate_score(box, img, use_position=True):
    global last_known_pos, last_box_dim, target_hist

    x1, y1, x2, y2, tid = box
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


# =========================
# Selección manual
# =========================
def select_target(event, x, y, flags, param):
    global target_id, target_locked, current_boxes
    global last_known_pos, last_box_dim, target_hist
    global current_frame_img, lost_frames

    if event == cv2.EVENT_LBUTTONDBLCLK:
        found = False

        for box_data in current_boxes:
            x1, y1, x2, y2, tid = box_data
            if x1 <= x <= x2 and y1 <= y <= y2:
                target_id = tid
                target_locked = True
                last_known_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
                last_box_dim = (x2 - x1, y2 - y1)
                target_hist = extract_histogram(current_frame_img, (x1, y1, x2, y2))
                lost_frames = 0
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
            print("--- Filtro quitado ---")


# =========================
# Main
# =========================
def main():
    global target_id, target_locked, current_boxes, current_frame_img
    global last_known_pos, last_box_dim, target_hist, lost_frames

    input_path = "videos/predica.mp4"
    model = YOLO("yolo11m.pt")

    cv2.namedWindow("Pro Face Tracker")
    cv2.namedWindow("HSV Debug")
    cv2.setMouseCallback("Pro Face Tracker", select_target)

    results = model.track(
        source=input_path,
        persist=True,
        stream=True,
        classes=[0],
        conf=0.25,
        imgsz=1280,
        tracker="botsort.yaml",
    )

    for frame_result in results:
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

        # 1) Intentar seguir mismo ID
        if target_locked and target_id != -1:
            for det in detections:
                x1, y1, x2, y2, tid = det
                if tid == target_id:
                    target_found = True
                    active_box = (x1, y1, x2, y2)
                    last_known_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
                    last_box_dim = (x2 - x1, y2 - y1)

                    new_hist = extract_histogram(img, (x1, y1, x2, y2))
                    target_hist = update_target_hist(target_hist, new_hist)

                    lost_frames = 0
                    break

        # 2) Si no está: modo perdido
        if target_locked and not target_found:
            lost_frames += 1

            # 2A) Pérdida corta: búsqueda local
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
                    target_id = best["tid"]
                    last_known_pos = best["center"]
                    last_box_dim = best["size"]
                    target_hist = update_target_hist(target_hist, best["hist"])
                    target_found = True
                    active_box = best["box"]
                    lost_frames = 0
                    print(f"--- Recuperación local: nuevo ID {target_id} ---")

            # 2B) Pérdida larga: búsqueda global
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
                    target_id = best["tid"]
                    last_known_pos = best["center"]
                    last_box_dim = best["size"]
                    target_hist = update_target_hist(target_hist, best["hist"])
                    target_found = True
                    active_box = best["box"]
                    lost_frames = 0
                    print(f"--- Reidentificado globalmente: nuevo ID {target_id} ---")

        # Dibujar cajas
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

        # Estado visual
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

        # Debug HSV
        debug_img = np.zeros((300, 700, 3), dtype=np.uint8)

        if target_locked and active_box is not None:
            dbg = build_hsv_debug_view(current_frame_img, active_box)
            if dbg is not None:
                debug_img = dbg
        else:
            cv2.putText(
                debug_img,
                "Haz doble click sobre el conferencista",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Pro Face Tracker", img)
        cv2.imshow("HSV Debug", debug_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
