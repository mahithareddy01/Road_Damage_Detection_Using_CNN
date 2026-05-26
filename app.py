import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Road Damage Detection System",
    page_icon="🛣️",
    layout="wide"
)
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #334155
    );
    color: white;
}

/* Headers */
h1 {
    text-align: center;
    color: #38bdf8;
    font-weight: 800;
}

h2, h3 {
    color: #f8fafc;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    border: 2px dashed #38bdf8;
    border-radius: 15px;
    padding: 20px;
    background-color: rgba(255,255,255,0.05);
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.08);
    border: 1px solid #38bdf8;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
}

/* Recommendation Box */
.stAlert {
    border-radius: 15px;
}

/* Section Containers */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Prediction Card */
.prediction-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    margin-top: 15px;
}

/* Footer */
.footer {
    text-align:center;
    color:#cbd5e1;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)
# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("road_damage_cnn.keras")

@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
label_encoder = load_encoder()

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div style="
padding:25px;
border-radius:20px;
background:linear-gradient(90deg,#0ea5e9,#2563eb);
text-align:center;
">

<h1 style="color:white;">
🛣️ AI-Based Road Damage Detection System
</h1>

<h4 style="color:white;">
Smart City Infrastructure Monitoring using CNN
</h4>

</div>
""", unsafe_allow_html=True)

st.markdown("""
### Smart City Infrastructure Monitoring using CNN
This system automatically identifies road surface damages such as potholes,
cracks, and manholes using Deep Learning and Computer Vision.
""")

st.divider()

# =====================================================
# ABOUT PROJECT
# =====================================================

st.header("📖 About the Project")

st.write("""
Road damage can cause:

- Traffic accidents
- Vehicle damage
- Increased maintenance costs
- Public safety risks

### Role of CNN

Convolutional Neural Networks automatically learn image features and
identify different categories of road damage.

### Industry Applications

- Smart Cities
- Municipal Corporations
- Highway Monitoring
- Road Safety Systems
- Infrastructure Maintenance
""")

st.divider()

# =====================================================
# IMAGE UPLOAD
# =====================================================

st.header("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# FUNCTIONS
# =====================================================

def preprocess_image(img):

    img = img.resize((224, 224))

    img = np.array(img)

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img


def get_severity(prediction, confidence):

    prediction = prediction.lower()

    if prediction == "pothole":

        if confidence > 80:
            return "High"

        return "Medium"

    elif prediction == "crack":

        if confidence > 80:
            return "Medium"

        return "Low"

    else:
        return "Low"


def get_recommendation(prediction):

    prediction = prediction.lower()

    if prediction == "pothole":
        return """
🚨 Immediate maintenance recommended.

High-risk road condition detected.

Potential danger to vehicles and pedestrians.
"""

    elif prediction == "crack":
        return """
⚠ Schedule repair soon.

Cracks may expand over time and
develop into potholes.
"""

    elif prediction == "manhole":
        return """
✅ Routine inspection recommended.

No immediate road damage detected.
"""

    return "Further inspection recommended."


# =====================================================
# PREDICTION
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    st.header("🖼 Uploaded Image Preview")

    st.image(
        image,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    img = preprocess_image(image)

    prediction = model.predict(img)

    probs = prediction[0]

    pred_idx = np.argmax(probs)

    pred_class = label_encoder.inverse_transform(
        [pred_idx]
    )[0]

    confidence = float(np.max(probs)) * 100

    severity = get_severity(
        pred_class,
        confidence
    )

    # =================================================
    # RESULT
    # =================================================

    st.divider()

    st.header("🎯 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Prediction",
            pred_class.capitalize()
        )

    with col2:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    with col3:
        if severity == "High":
            st.error("🔴 High Severity")

        elif severity == "Medium":
            st.warning("🟠 Medium Severity")

        else:
            st.success("🟢 Low Severity")

    # =================================================
    # CHART
    # =================================================

    st.divider()

    st.header("📊 Class Confidence Graph")

    class_names = label_encoder.classes_

    chart_df = pd.DataFrame({
        "Class": class_names,
        "Confidence (%)": probs * 100
    })

    fig = px.bar(
        chart_df,
        x="Class",
        y="Confidence (%)",
        title="Class Probability Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =================================================
    # RECOMMENDATION
    # =================================================

    st.divider()

    st.header("🛠 Recommendations")

    st.warning(
        get_recommendation(pred_class)
    )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
### CNN Features Used

✅ Convolution Layers

✅ MaxPooling Layers

✅ Dropout Regularization

✅ Data Augmentation

✅ Real-Time Prediction

✅ Smart Maintenance Recommendation

Built using:

- TensorFlow / Keras
- Streamlit
- Plotly
- NumPy
- Pillow
""")