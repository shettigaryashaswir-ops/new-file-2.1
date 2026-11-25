import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import requests
from streamlit_image_coordinates import streamlit_image_coordinates
from skimage.color import rgb2lab

# ==========================================
# STEP 1: WEB PAGE CONFIGURATION & TITLE
# ==========================================
st.set_page_config(page_title="Ultimate Shade Detector", layout="centered")
st.title("🎨 Ultimate Color Shade Detector")
st.write("Identify over 865+ unique shades using your webcam or an uploaded picture!")

# ==========================================
# STEP 2: HELPER FUNCTIONS & DATASET LOADING
# ==========================================
def rgb_to_lab(r, g, b):
    """Converts 0-255 RGB values to integer CIELAB coordinates."""
    rgb_normalized = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=np.float32)
    lab = rgb2lab(rgb_normalized)[0, 0]
    return int(round(lab[0])), int(round(lab[1])), int(round(lab[2]))

CSV_URL = "https://raw.githubusercontent.com/codebrainz/color-names/master/output/colors.csv"
CSV_FILE = "colors.csv"

@st.cache_data
def load_dataset():
    column_names = ["color_name", "hex", "R", "G", "B"]

    if not os.path.exists(CSV_FILE):
        try:
            response = requests.get(CSV_URL, timeout=10)
            with open(CSV_FILE, "wb") as f:
                f.write(response.content)
        except Exception:
            # Fallback dataset if internet fetch fails
            return pd.DataFrame([
                ["Pure Black", "#000000", 0, 0, 0, 0, 0, 0],
                ["Pure White", "#ffffff", 255, 255, 255, 100, 0, 0]
            ], columns=column_names + ["L", "A", "B"])

    df = pd.read_csv(CSV_FILE, names=column_names, header=None)

    # Pre-calculate CIELAB values for perceptual matching
    rgb_array = df[["R", "G", "B"]].values.reshape(-1, 1, 3) / 255.0
    lab_array = rgb2lab(rgb_array).reshape(-1, 3)

    df["L"] = np.round(lab_array[:, 0]).astype(int)
    df["A"] = np.round(lab_array[:, 1]).astype(int)
    df["B"] = np.round(lab_array[:, 2]).astype(int)

    return df

color_df = load_dataset()

# ==========================================
# STEP 3: PERCEPTUAL MATCHING ENGINE (CIELAB)
# ==========================================
def identify_shade_lab(target_l, target_a, target_b):
    """Calculates nearest color shade using Euclidean distance in CIELAB space."""
    distances = np.sqrt(
        (color_df["L"] - target_l) ** 2 +
        (color_df["A"] - target_a) ** 2 +
        (color_df["B"] - target_b) ** 2
    )

    match_idx = distances.idxmin()
    delta_e = distances[match_idx]

    # Calculate accuracy percentage (100% minus Delta E distance error)
    accuracy = max(0.0, 100.0 - delta_e)

    return color_df.loc[match_idx, "color_name"], round(accuracy, 1)

# ==========================================
# STEP 4: SIDEBAR & USER INPUT INTERFACE
# ==========================================
mode = st.sidebar.radio("Choose Input Method:", ("📤 Upload an Image", "📷 Take a Webcam Photo"))

target_image = None

if mode == "📤 Upload an Image":
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        target_image = Image.open(uploaded_file).convert("RGB")

elif mode == "📷 Take a Webcam Photo":
    webcam_file = st.camera_input("Snapshot an object")
    if webcam_file is not None:
        target_image = Image.open(webcam_file).convert("RGB")

# ==========================================
# STEP 5: INTERACTIVE CLICKING & RESULT DISPLAY
# ==========================================
if target_image is not None:
    st.info("👇 Click directly on any object inside the image below to identify its shade!")

    # Scale image nicely to fit screens
    target_image.thumbnail((600, 600))
    img_width, img_height = target_image.size

    # Render image on webpage and capture mouse click coordinates
    value = streamlit_image_coordinates(target_image, width=img_width, key="color_picker")

    if value is not None:
        x, y = value["x"], value["y"]

        # Verify click coordinates are safely inside image bounds
        if x < img_width and y < img_height:
            img_array = np.array(target_image)
            r, g, b = [int(v) for v in img_array[y, x]]  # Extract red, green, and blue values

            # Convert clicked pixel to CIELAB coordinates
            l_val, a_val, b_val = rgb_to_lab(r, g, b)

            # Find closest shade and compute accuracy
            shade_name, accuracy_percentage = identify_shade_lab(l_val, a_val, b_val)

            # --- DISPLAY RESULT BOX WITH COLOR NAME, LAB, RGB, AND ACCURACY ---
            st.markdown("---")
            st.subheader("🔍 Analysis Results (CIELAB)")

            st.markdown(f"""
            <div style="background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0; box-shadow: 0px 4px 10px rgba(0,0,0,0.05);">
                <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 12px;">
                    <div style="background-color: rgb({r}, {g}, {b}); width: 60px; height: 60px;
                                border-radius: 50%; border: 2px solid #ccc; flex-shrink: 0;"></div>
                    <div>
                        <h2 style="margin: 0; color: #111; font-size: 1.8em;">{shade_name}</h2>
                        <p style="margin: 6px 0 0 0; color: #555; font-size: 1.05em;">
                            <b>CIELAB (L*, a*, b*):</b> <code>({l_val}, {a_val}, {b_val})</code>
                        </p>
                        <p style="margin: 4px 0 0 0; color: #555; font-size: 1.05em;">
                            <b>RGB Value:</b> <code>({r}, {g}, {b})</code>
                        </p>
                    </div>
                </div>
                <hr style="border: 0; border-top: 1px solid #eee; margin: 12px 0;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #444; font-weight: bold; font-size: 1.05em;">Match Accuracy:</span>
                    <span style="background-color: #e6f4ea; color: #137333; font-weight: bold; padding: 4px 12px; border-radius: 16px; font-size: 1.05em;">
                        {accuracy_percentage}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)