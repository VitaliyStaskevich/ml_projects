import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Настройки страницы ---
st.set_page_config(page_title="U-Net Image Segmentation", layout="wide")
st.title("Image Segmentation with U-Net")

# --- Загрузка модели ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("unet_pet_final_dualloss.h5", compile=False)
    return model

model = load_model()
IMG_SIZE = 128  # размер, используемый при обучении
DISPLAY_SIZE = 500  # размер для отображения на экране

# --- Функция предобработки ---
def preprocess_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Функция постобработки ---
def predict_mask(model, image):
    pred_mask = model.predict(image)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    return pred_mask.squeeze()

# --- Загрузка изображения ---
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Чтение изображения
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Running inference..."):
        processed = preprocess_image(image)
        mask = predict_mask(model, processed)

        overlay = np.array(image.resize((IMG_SIZE, IMG_SIZE))) * 0.6 + np.stack([mask * 255] * 3, axis=-1) * 0.4
        overlay = overlay.astype(np.uint8)

    # --- Изменяем размер для отображения ---
    image_display = image.resize((DISPLAY_SIZE, DISPLAY_SIZE))
    mask_display = Image.fromarray((mask * 255).astype(np.uint8)).resize((DISPLAY_SIZE, DISPLAY_SIZE))
    overlay_display = Image.fromarray(overlay).resize((DISPLAY_SIZE, DISPLAY_SIZE))

    # --- Отображение изображений в одну строку ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.image(image_display, caption="Original Image", width=DISPLAY_SIZE)

    with col2:
        st.subheader("Predicted Mask")
        st.image(mask_display, caption="Segmentation Mask", width=DISPLAY_SIZE)

    with col3:
        st.subheader("Overlay")
        st.image(overlay_display, caption="Overlayed Result", width=DISPLAY_SIZE)
