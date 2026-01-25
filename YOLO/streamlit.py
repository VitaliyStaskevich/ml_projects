import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Page setup ---
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("📦 Object Detection with YOLOv8")

# --- Load YOLO model ---
@st.cache_resource
def load_yolo_model():
    model = YOLO("best_people.pt") 
    return model

model = load_yolo_model()
DISPLAY_SIZE = 600

# --- Draw bounding boxes ---
def draw_boxes(image: Image.Image, results):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Try loading a font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for r in results:
        boxes = r.boxes.xyxy
        classes = r.boxes.cls
        confs = r.boxes.conf

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)

            class_name = model.names[int(cls)]
            label = f"{class_name} {float(conf):.2f}"

            # --- Compute text size using textbbox() ---
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # --- Thicker bounding box ---
            draw.rectangle([x1, y1, x2, y2], outline="red", width=6)

            # --- Background box for label ---
            draw.rectangle(
                [x1, y1 - text_h - 4, x1 + text_w + 4, y1],
                fill="red"
            )

            # --- Text ---
            draw.text((x1 + 2, y1 - text_h - 2), label, fill="white", font=font)

    return img

# --- File uploader ---
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Running YOLOv8 inference..."):
        results = model.predict(np.array(image))
        result_img = draw_boxes(image, results)

    # Resize for display
    image_display = image.resize((DISPLAY_SIZE, DISPLAY_SIZE))
    boxed_display = result_img.resize((DISPLAY_SIZE, DISPLAY_SIZE))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_display, width=DISPLAY_SIZE)

    with col2:
        st.subheader("Detected Objects")
        st.image(boxed_display, width=DISPLAY_SIZE)
