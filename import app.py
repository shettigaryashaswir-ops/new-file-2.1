import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from skimage.color import rgb2lab
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Ultimate Shade Detector", layout="centered")
st.title("🎨 Ultimate Color Shade Detector")
st.write("Identify named shades using your webcam or an uploaded picture!")

CSV_URL = "https://raw.githubusercontent.com/codebrainz/color-names/master/output/colors.csv"
CSV_FILE = "colors.csv"


def rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    rgb_norm = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=np.float64)
    lab = rgb2lab(rgb_norm)[0, 0]
    return float(lab[0]), float(lab[1]), float(lab[2])


@st.cache_data
def load_dataset() -> pd.DataFrame:
    column_names = ["color_name", "hex", "R", "G", "B"]

    if not os.path.exists(CSV_FILE):
        try:
            response = requests.get(CSV_URL, timeout=10)
            response.raise_for_status()
            tmp_file = f"{CSV_FILE}.tmp"
            with open(tmp_file, "wb") as f:
                f.write(response.content)
            os.replace(tmp_file, CSV_FILE)
        except Exception:
            df = pd.DataFrame(
                [
                    ["Pure Black", "#000000", 0, 0, 0],
                    ["Pure White", "#ffffff", 255, 255, 255],
                ],
                columns=column_names,
            )
        else:
            df = pd.read_csv(CSV_FILE, names=column_names, header=None)
    else:
        df = pd.read_csv(CSV_FILE, names=column_names, header=None)

    df[["R", "G", "B"]] = df[["R", "G", "B"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["R", "G", "B"]).copy()
    df[["R", "G", "B"]] = df[["R", "G", "B"]].astype(int)

    rgb_array = df[["R", "G", "B"]].to_numpy(dtype=np.float64).reshape(-1, 1, 3) / 255.0
    lab_array = rgb2lab(rgb_array).reshape(-1, 3)
    df[["L", "A", "B"]] = lab_array
    return df


color_df = load_dataset()


def identify_shade_lab(target_l: float, target_a: float, target_b: float):
    distances = np.sqrt(
        (color_df["L"] - target_l) ** 2
        + (color_df["A"] - target_a) ** 2
        + (color_df["B"] - target_b) ** 2
    )
    match_idx = distances.idxmin()
    delta_e = float(distances.loc[match_idx])
    row = color_df.loc[match_idx]

    matched_r = int(row["R"])
    matched_g = int(row["G"])
    matched_b = int(row["B"])
    similarity = 100.0 / (1.0 + delta_e / 45.0)

    return (
        row["color_name"],
        row["hex"],
        (matched_r, matched_g, matched_b),
        delta_e,
        round(similarity, 1),
    )


mode = st.sidebar.radio(
    "Choose Input Method:", ("📤 Upload an Image", "📷 Take a Webcam Photo")
)
sample_radius = st.sidebar.slider("Sample patch radius (px)", min_value=0, max_value=5, value=1)

target_image = None
source_id = mode

if mode == "📤 Upload an Image":
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        target_image = Image.open(uploaded_file).convert("RGB")
        source_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
elif mode == "📷 Take a Webcam Photo":
    webcam_file = st.camera_input("Snapshot an object")
    if webcam_file is not None:
        target_image = Image.open(webcam_file).convert("RGB")
        source_id = f"webcam_{webcam_file.size}"

if target_image is not None:
    st.info("👇 Click directly on any object inside the image below to identify its shade!")
    target_image.thumbnail((600, 600))
    img_width, img_height = target_image.size

    value = streamlit_image_coordinates(
        target_image,
        width=img_width,
        key=f"color_picker_{source_id}_{img_width}x{img_height}",
    )

    if value is not None:
        x = int(round(value["x"]))
        y = int(round(value["y"]))

        if 0 <= x < img_width and 0 <= y < img_height:
            img_array = np.array(target_image, dtype=np.float64)
            x0, x1 = max(0, x - sample_radius), min(img_width, x + sample_radius + 1)
            y0, y1 = max(0, y - sample_radius), min(img_height, y + sample_radius + 1)
            patch_mean = img_array[y0:y1, x0:x1].mean(axis=(0, 1))
            r, g, b = [int(round(c)) for c in patch_mean]
            sampled_hex = f"#{r:02X}{g:02X}{b:02X}"

            l_val, a_val, b_val = rgb_to_lab(r, g, b)
            shade_name, matched_hex, matched_rgb, delta_e, similarity = identify_shade_lab(
                l_val, a_val, b_val
            )
            matched_r, matched_g, matched_b = matched_rgb

            st.markdown("---")
            st.subheader("🔍 Analysis Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Closest Shade", shade_name)
                st.caption(f"Matched RGB: `({matched_r}, {matched_g}, {matched_b})`")
                st.caption(f"Matched Hex: `{matched_hex}`")
            with col2:
                st.metric("Sampled RGB", f"({r}, {g}, {b})")
                st.caption(f"Sampled Hex: `{sampled_hex}`")
                st.caption(f"Click at `({x}, {y})`")
            with col3:
                st.metric("Similarity", f"{similarity}%")
                st.caption(f"CIELAB ΔE: `{delta_e:.1f}`")

            st.markdown(
                f"""
                <div style="display:flex; gap:16px; margin-top:12px;">
                    <div style="flex:1; background:rgb({r},{g},{b}); height:68px; border-radius:8px; border:1px solid #ccc; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#fff; text-shadow:0 1px 2px rgba(0,0,0,0.85); font-weight:600;">
                        <span>Sampled Color</span>
                        <span style="font-size:0.85em; font-weight:500;">RGB ({r}, {g}, {b})</span>
                    </div>
                    <div style="flex:1; background:rgb({matched_r},{matched_g},{matched_b}); height:68px; border-radius:8px; border:1px solid #ccc; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#fff; text-shadow:0 1px 2px rgba(0,0,0,0.85); font-weight:600;">
                        <span>Matched: {shade_name}</span>
                        <span style="font-size:0.85em; font-weight:500;">RGB ({matched_r}, {matched_g}, {matched_b})</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )