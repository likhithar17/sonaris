import io
import cv2
import numpy as np
import base64
import hashlib
from PIL import Image
import json
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
app = FastAPI(title="SONARIS API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")

KNOWN_CLASSES = {
    0: "shipwreck",
    1: "aircraft_wreck",
    2: "submerged_structure",
    3: "marine_debris",
}
CONF_THRESHOLD = 0.45

MEMORY_FILE = Path("seafloor_memory.json")

if not MEMORY_FILE.exists():
    MEMORY_FILE.write_text("[]", encoding="utf-8")

@app.get("/")
def read_root():
    return {"status": "online", "system": "SONARIS Anomaly Engine"}
def load_memory():
    return json.loads(
        MEMORY_FILE.read_text(encoding="utf-8")
    )
@app.get("/memory")
def get_memory():
    return load_memory()

def save_memory(memory):
    MEMORY_FILE.write_text(
        json.dumps(memory, indent=2),
        encoding="utf-8"
    )
def adaptive_preprocess(image_np):
    """
    Assess sonar-image quality and apply a suitable preprocessing strategy.

    Prototype quality indicators:
    - Contrast
    - Brightness
    - Estimated noise

    The preprocessing is adaptive:
    - Low contrast -> CLAHE contrast enhancement
    - High noise -> denoising
    - Poor brightness -> intensity normalization
    - Good quality -> minimal processing
    """

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Basic image-quality measurements
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Simple noise estimate using local high-frequency variation
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_estimate = float(np.std(gray.astype(np.float32) - blurred.astype(np.float32)))

    processed = image_np.copy()
    preprocessing_steps = []

    # 1. Low-contrast sonar image
    if contrast < 35:
        lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l_channel = clahe.apply(l_channel)

        processed = cv2.cvtColor(
            cv2.merge((l_channel, a_channel, b_channel)),
            cv2.COLOR_LAB2RGB
        )

        preprocessing_steps.append("contrast_enhancement")

    # 2. Noisy sonar image
    if noise_estimate > 18:
        processed = cv2.GaussianBlur(
            processed,
            (3, 3),
            0
        )

        preprocessing_steps.append("noise_reduction")

    # 3. Poor brightness
    if brightness < 60 or brightness > 200:
        processed = cv2.normalize(
            processed,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        preprocessing_steps.append("intensity_normalization")

    # 4. Good-quality image
    if not preprocessing_steps:
        preprocessing_steps.append("minimal_processing")

    quality = {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "noise_estimate": round(noise_estimate, 2)
    }

    return processed, quality, preprocessing_steps

def run_clean_inference(image_np):
    # conf=0.12 catches the small anomaly without losing sensitivity
    return model.predict(
        source=image_np,
        conf=0.12,
        iou=0.40,
        agnostic_nms=True,
        save=False
    )[0]
def calculate_shadow_score(image_np, bbox):
    """
    Estimate acoustic-shadow evidence using:
    1. Adaptive local darkness
    2. Shadow adjacency to the detected target
    3. Connected dark-region analysis
    4. Shadow size and elongation

    This is a prototype heuristic and requires validation
    against labelled side-scan sonar data.
    """

    x1, y1, x2, y2 = map(int, bbox)

    height, width = image_np.shape[:2]

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Keep bounding box inside image
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return {
            "shadow_score": 0.0,
            "shadow_confirmed": False
        }

    target_width = x2 - x1
    target_height = y2 - y1

    # Region used to estimate local seabed intensity
    margin = max(
        20,
        int(max(target_width, target_height) * 0.75)
    )

    rx1 = max(0, x1 - margin)
    ry1 = max(0, y1 - margin)
    rx2 = min(width, x2 + margin)
    ry2 = min(height, y2 + margin)

    local_region = gray[ry1:ry2, rx1:rx2]

    if local_region.size == 0:
        return {
            "shadow_score": 0.0,
            "shadow_confirmed": False
        }

    local_median = float(np.median(local_region))

    # Adaptive threshold based on local seabed brightness.
    # A region significantly darker than the local seabed
    # becomes a shadow candidate.
    adaptive_threshold = local_median * 0.72

    binary = np.zeros_like(gray, dtype=np.uint8)

    binary[ry1:ry2, rx1:rx2] = (
        local_region < adaptive_threshold
    ).astype(np.uint8) * 255

    # Remove tiny isolated noise.
    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # Join nearby dark pixels belonging to the same region.
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Check candidate regions immediately outside the target.
    search_margin = max(
        15,
        int(max(target_width, target_height) * 0.60)
    )

    search_regions = [
        (
            "right",
            x2,
            max(0, y1),
            min(width, x2 + search_margin),
            min(height, y2)
        ),
        (
            "left",
            max(0, x1 - search_margin),
            max(0, y1),
            x1,
            min(height, y2)
        ),
        (
            "below",
            max(0, x1),
            y2,
            min(width, x2),
            min(height, y2 + search_margin)
        ),
        (
            "above",
            max(0, x1),
            max(0, y1 - search_margin),
            min(width, x2),
            y1
        )
    ]

    candidate_scores = []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    target_area = max(
        1,
        target_width * target_height
    )

    for direction, sx1, sy1, sx2, sy2 in search_regions:

        if sx2 <= sx1 or sy2 <= sy1:
            continue

        best_component = None

        for component_id in range(1, num_labels):

            cx = int(centroids[component_id][0])
            cy = int(centroids[component_id][1])

            area = int(stats[component_id, cv2.CC_STAT_AREA])

            if area < max(20, int(target_area * 0.01)):
                continue

            # Component must lie inside the directional search region.
            if not (
                sx1 <= cx <= sx2
                and sy1 <= cy <= sy2
            ):
                continue

            comp_x = int(stats[component_id, cv2.CC_STAT_LEFT])
            comp_y = int(stats[component_id, cv2.CC_STAT_TOP])
            comp_w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            comp_h = int(stats[component_id, cv2.CC_STAT_HEIGHT])

            comp_x2 = comp_x + comp_w
            comp_y2 = comp_y + comp_h

            # Check whether the dark component is close to the target.
            if direction == "right":
                gap = max(0, comp_x - x2)

            elif direction == "left":
                gap = max(0, x1 - comp_x2)

            elif direction == "below":
                gap = max(0, comp_y - y2)

            else:
                gap = max(0, y1 - comp_y2)

            # Shadow should begin close to the target.
            adjacency_score = max(
                0.0,
                1.0 - (gap / max(search_margin, 1))
            )

            if adjacency_score <= 0:
                continue

            # Shadow should have some meaningful size.
            area_score = min(
                1.0,
                area / max(target_area * 0.15, 1)
            )

            # Measure elongation.
            longest_side = max(comp_w, comp_h)
            shortest_side = max(min(comp_w, comp_h), 1)

            elongation = longest_side / shortest_side

            elongation_score = min(
                1.0,
                max(0.0, (elongation - 1.0) / 4.0)
            )

            # Measure darkness of this connected component.
            component_mask = labels == component_id
            component_pixels = gray[component_mask]

            if component_pixels.size == 0:
                continue

            component_median = float(
                np.median(component_pixels)
            )

            contrast = max(
                0.0,
                (local_median - component_median)
                / max(local_median, 1.0)
            )

            contrast_score = min(
                1.0,
                contrast * 2.0
            )

            # Combined evidence.
            score = (
                0.35 * contrast_score
                + 0.30 * adjacency_score
                + 0.20 * area_score
                + 0.15 * elongation_score
            )

            if best_component is None or score > best_component:
                best_component = score

        if best_component is not None:
            candidate_scores.append(best_component)

    if not candidate_scores:
        return {
            "shadow_score": 0.0,
            "shadow_confirmed": False
        }

    shadow_score = max(candidate_scores)

    # Conservative prototype decision.
    shadow_confirmed = bool(
        shadow_score >= 0.50
    )

    return {
        "shadow_score": round(float(shadow_score), 4),
        "shadow_confirmed": shadow_confirmed
    }

def coordinates_match(value_a, value_b):
    """Compare telemetry coordinates safely across strings and numbers."""
    if value_a is None or value_b is None:
        return value_a == value_b

    try:
        return abs(float(value_a) - float(value_b)) < 1e-9
    except (TypeError, ValueError):
        return str(value_a).strip() == str(value_b).strip()


def calculate_iou(box_a, box_b):
    """Calculate intersection-over-union for two [x1, y1, x2, y2] boxes."""
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compare_with_memory(
    memory,
    pred_name,
    bbox,
    latitude,
    longitude,
    survey_id
):
    """Find the closest matching target from a previous survey."""
    best_match = None

    for record in memory:
        # Never compare a target against another record from the
        # same survey ID.
        if record.get("survey_id") == survey_id:
            continue

        # Ignore legacy records without survey IDs because they cannot
        # be safely associated with a previous survey.
        if not record.get("survey_id"):
            continue

        if record.get("predicted_class") != pred_name:
            continue

        if not coordinates_match(record.get("latitude"), latitude):
            continue

        if not coordinates_match(record.get("longitude"), longitude):
            continue

        old_bbox = record.get("bbox")
        if not old_bbox:
            continue

        iou = calculate_iou(bbox, old_bbox)

        if iou >= 0.50:
            if best_match is None or iou > best_match["iou"]:
                best_match = {
                    "match": True,
                    "previous_target_id": record.get("target_id"),
                    "previous_survey_id": record.get("survey_id"),
                    "iou": float(iou)
                }

    if best_match is not None:
        best_match["iou"] = round(best_match["iou"], 4)
        return best_match

    return {
        "match": False,
        "previous_target_id": None,
        "previous_survey_id": None,
        "iou": 0.0
    }


def is_duplicate_detection(
    memory,
    pred_name,
    bbox,
    latitude,
    longitude,
    survey_id,
    image_hash
):
    """
    Prevent the same target from being stored twice.

    Primary duplicate key:
      same uploaded image + same class + same location + IoU >= 0.80

    Secondary duplicate key:
      same survey ID + same class + same location + IoU >= 0.80

    The image hash is important because survey_id is intentionally unique
    for each analysis request, so relying only on survey_id cannot stop a
    user from uploading the same sonar image again.
    """
    for record in memory:
        if record.get("predicted_class") != pred_name:
            continue

        if not coordinates_match(record.get("latitude"), latitude):
            continue

        if not coordinates_match(record.get("longitude"), longitude):
            continue

        old_bbox = record.get("bbox")
        if not old_bbox:
            continue

        iou = calculate_iou(bbox, old_bbox)
        if iou < 0.80:
            continue

        same_survey = (
            bool(survey_id)
            and record.get("survey_id") == survey_id
        )

        same_image = (
            bool(image_hash)
            and record.get("image_hash") == image_hash
        )

        if same_survey or same_image:
            return True

    return False


def build_detection_result(
    results,
    image_np,
    latitude,
    longitude,
    heading,
    survey_id,
    image_hash,
    source_filename
):
    detections = []
    memory = load_memory()
    memory_changed = False

    for index, box in enumerate(results.boxes):
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        new_bbox = [x1, y1, x2, y2]

        shadow_result = calculate_shadow_score(
            image_np,
            new_bbox
        )

        print("SHADOW DEBUG:", shadow_result)
        print("BBOX:", new_bbox)

        pred_name = model.names.get(
            cls_id,
            KNOWN_CLASSES.get(cls_id, "anomaly")
        )

        confidence_pass = conf >= CONF_THRESHOLD

        if confidence_pass and shadow_result["shadow_confirmed"]:
            label = pred_name
            verification_status = "VERIFIED"
        elif confidence_pass:
            label = pred_name
            verification_status = "NEEDS_REVIEW"
        else:
            label = f"UNKNOWN ({pred_name})"
            verification_status = "UNKNOWN"

        comparison = compare_with_memory(
            memory,
            pred_name,
            new_bbox,
            latitude,
            longitude,
            survey_id
        )

        duplicate = is_duplicate_detection(
            memory,
            pred_name,
            new_bbox,
            latitude,
            longitude,
            survey_id,
            image_hash
        )

        if duplicate or comparison["match"]:
            change_status = "EXISTING"
        else:
            change_status = "NEW"

        target_id = f"T{index + 1:03d}"

        # Store a new memory record only when this target is not a duplicate.
        if not duplicate:
            memory.append({
                "target_id": target_id,
                "survey_id": survey_id,
                "image_hash": image_hash,
                "source_filename": source_filename,
                "comparison": comparison,
                "label": label,
                "predicted_class": pred_name,
                "comparison_status": (
                    "EXISTING"
                    if comparison["match"]
                    else "NEW"
                ),
                "previous_target_id": comparison["previous_target_id"],
                "previous_survey_id": comparison["previous_survey_id"],
                "comparison_iou": comparison["iou"],
                "change_status": change_status,
                "confidence": round(conf, 4),
                "verification_status": verification_status,
                "shadow_score": shadow_result["shadow_score"],
                "shadow_confirmed": shadow_result["shadow_confirmed"],
                "latitude": latitude,
                "longitude": longitude,
                "heading": heading,
                "bbox": new_bbox,
                "timestamp": datetime.now().isoformat()
            })

            memory_changed = True

        detections.append({
            "target_id": target_id,
            "label": label,
            "predicted_class": pred_name,
            "comparison_status": (
                "EXISTING"
                if comparison["match"] or duplicate
                else "NEW"
            ),
            "previous_target_id": comparison["previous_target_id"],
            "previous_survey_id": comparison["previous_survey_id"],
            "comparison_iou": comparison["iou"],
            "duplicate_prevented": duplicate,
            "confidence": round(conf, 4),
            "confidence_rate": round(conf, 4),
            "status": (
                "CONFIRMED"
                if confidence_pass
                else "FLAGGED_FOR_REVIEW"
            ),
            "bbox": new_bbox,
            "multi_step_verification": {
                "gate1_confidence_pass": confidence_pass,
                "gate2_shadow_confirmed": shadow_result["shadow_confirmed"],
                "gate3_status": verification_status
            },
            "seafloor_memory_coords": {
                "lat": latitude,
                "lon": longitude
            },
            "risk_score": 0,
            "risk_rate_percent": 0,
        })

    if memory_changed:
        save_memory(memory)

    return detections

@app.post("/detect")
async def detect_sonar_objects(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    np_img = np.array(image)

    results = run_clean_inference(np_img)

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        pred_name = KNOWN_CLASSES.get(cls_id, "anomaly")

        if conf < CONF_THRESHOLD:
            category = f"UNKNOWN ({pred_name})"
            status = "FLAGGED_FOR_REVIEW"
        else:
            category = pred_name
            status = "CONFIRMED"

        detections.append({
            "label": category,
            "predicted_class": pred_name,
            "confidence": round(conf, 4),
            "status": status,
            "bbox": [x1, y1, x2, y2]
        })

    return {
        "filename": file.filename,
        "total_detections": len(detections),
        "detections": detections
    }

@app.post("/detect/visualize", responses={200: {"content": {"image/jpeg": {}}}})
async def detect_and_render_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    np_img = np.array(image)

    results = run_clean_inference(np_img)
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    # Sort small boxes last so small object badges draw cleanly on top
    sorted_boxes = sorted(results.boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]), reverse=True)

    for box in sorted_boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        pred_name = KNOWN_CLASSES.get(cls_id, "anomaly")

        is_unknown = conf < CONF_THRESHOLD
        name_text = f" UNKNOWN: {pred_name.upper()} " if is_unknown else f" {pred_name.upper()} "
        conf_text = f" {conf:.2f} "

        box_color = (0, 0, 220) if is_unknown else (0, 220, 0)
        cv2.rectangle(bgr_img, (x1, y1), (x2, y2), box_color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1

        (w_name, h_name), base_name = cv2.getTextSize(name_text, font, font_scale, thickness)
        (w_conf, h_conf), base_conf = cv2.getTextSize(conf_text, font, font_scale, thickness)
        h_badge = max(h_name, h_conf) + 8

        badge_y1 = max(0, y1 - h_badge - 4)
        badge_y2 = badge_y1 + h_badge

        # Tag 1: Green Background for Anomaly Name
        name_x1 = max(2, x1)
        name_x2 = name_x1 + w_name
        cv2.rectangle(bgr_img, (name_x1, badge_y1), (name_x2, badge_y2), (0, 160, 0), -1)
        cv2.putText(bgr_img, name_text, (name_x1, badge_y2 - 6), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Tag 2: Deep Red Background for Confidence Rate (snapped right next to name)
        conf_x1 = name_x2
        conf_x2 = conf_x1 + w_conf
        cv2.rectangle(bgr_img, (conf_x1, badge_y1), (conf_x2, badge_y2), (0, 0, 180), -1)
        cv2.putText(bgr_img, conf_text, (conf_x1, badge_y2 - 6), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Subtle dark outer stroke around combined badge
        cv2.rectangle(bgr_img, (name_x1, badge_y1), (conf_x2, badge_y2), (20, 20, 20), 1)

    _, encoded_img = cv2.imencode(".jpg", bgr_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/analyze")
async def analyze_sonar(
    file: UploadFile = File(...),
    latitude: str = Form(""),
    longitude: str = Form(""),
    heading: str = Form("")
):
    survey_id = f"S{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    contents = await file.read()

    # Hash the uploaded sonar file so the backend can recognize the
    # exact same survey image even when a new survey_id is generated.
    image_hash = hashlib.sha256(contents).hexdigest()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    np_img = np.array(image)

# Assess sonar quality and adaptively preprocess the image
    processed_img, sonar_quality, preprocessing_steps = adaptive_preprocess(np_img)
# Run YOLO inference on the preprocessed image
    results = run_clean_inference(processed_img)
    # Build the detection structure expected
    # by the existing SONARIS frontend
    detections = build_detection_result(
        results,
        np_img,
        latitude,
        longitude,
        heading,
        survey_id,
        image_hash,
        file.filename
    )

    # Create the visualized sonar image
    bgr_img = cv2.cvtColor(
        np_img,
        cv2.COLOR_RGB2BGR
    )

    for box in results.boxes:

        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        pred_name = KNOWN_CLASSES.get(
            cls_id,
            "anomaly"
        )

        is_unknown = conf < CONF_THRESHOLD

        box_color = (
            (0, 0, 220)
            if is_unknown
            else (0, 220, 0)
        )

        cv2.rectangle(
            bgr_img,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        label = (
            f"UNKNOWN: {pred_name.upper()} {conf:.2f}"
            if is_unknown
            else f"{pred_name.upper()} {conf:.2f}"
        )

        cv2.putText(
            bgr_img,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
            cv2.LINE_AA
        )

    # Encode visualization as JPEG
    _, encoded_img = cv2.imencode(
        ".jpg",
        bgr_img
    )

    # Convert image to base64 for the frontend
    visual_render_base64 = (
        "data:image/jpeg;base64,"
        + base64.b64encode(
            encoded_img.tobytes()
        ).decode("utf-8")
    )

    return {
        "status": "success",
        "filename": file.filename,
        "survey_id": survey_id,

        "telemetry": {
            "latitude": latitude,
            "longitude": longitude,
            "heading": heading
        },

        "total_targets": len(detections),

        "detections": detections,
        "preprocessing_info": {
    "quality": sonar_quality,
    "steps": preprocessing_steps
},
        "visual_render_base64":
            visual_render_base64
    }