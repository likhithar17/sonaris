import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Response
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

@app.get("/")
def read_root():
    return {"status": "online", "system": "SONARIS Anomaly Engine"}

def run_clean_inference(image_np):
    # conf=0.12 catches the small anomaly without losing sensitivity
    return model.predict(
        source=image_np,
        conf=0.12,
        iou=0.40,
        agnostic_nms=True,
        save=False
    )[0]

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
