import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import io

model = YOLO("tb_yolo_results/exp/weights/best.pt")

st.set_page_config(page_title="TB X-ray Detection", layout="centered")
st.title(" Tuberculosis Detection in Chest X-rays")

uploaded_file = st.file_uploader(" Încarcă o imagine radiografică (JPG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagine încărcată", use_column_width=True)

    with st.spinner(" Se analizează imaginea..."):
        results = model.predict(image, conf=0.25)

        res_plotted = results[0].plot()
        st.image(res_plotted, caption=" Rezultatul modelului", use_column_width=True)

        for box in results[0].boxes:
            st.write(f" Detecție: {model.names[int(box.cls[0])]} - Scor: {float(box.conf[0]):.2f}")
