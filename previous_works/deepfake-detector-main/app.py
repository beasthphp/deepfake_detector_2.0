import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
from huggingface_hub import hf_hub_download

# Basic Page Setup
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️‍♂️", layout="centered")

# --- 1. THE SMART MODEL LOADER ---
@st.cache_resource
def load_my_model():
    model_filename = "deepfake_detector_93acc.h5"
    
    # Check if the file is already there locally
    if os.path.exists(model_filename):
        return load_model(model_filename)
    
    # If not, download it from your Space automatically
    try:
        with st.spinner("Downloading model weights..."):
            model_path = hf_hub_download(
                repo_id="harshpsingh4002/deepfake-detector", 
                filename=model_filename,
                repo_type="space"
            )
            return load_model(model_path)
    except Exception as e:
        st.error(f"Critical Error: Could not load model. {e}")
        return None

# Start loading the "brain"
model = load_my_model()

# --- 2. USER INTERFACE ---
st.title("🕵️‍♂️ Deepfake Detector")
st.markdown("---")
st.write("Upload a face image below. Our AI will analyze the pixel patterns to determine if the face is **Real** or **AI-Generated (Fake)**.")

if model is not None:
    # The Upload Button
    uploaded_file = st.file_uploader("Choose a JPG or PNG image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the image
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption='Uploaded Image', use_container_width=True)
            
            with st.status("Analyzing Image...", expanded=True) as status:
                st.write("Preprocessing pixels...")
                # Preprocess: Resize to 256x256 (same as training)
                img_resized = img.resize((256, 256))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0 # Normalization

                st.write("Running Neural Network prediction...")
                # Predict
                prediction = model.predict(img_array)
                score = prediction[0][0]
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # --- 3. THE VERDICT ---
            st.markdown("### **Final Verdict:**")
            
            # Logic: 0 = Fake, 1 = Real
            if score < 0.5:
                confidence = (1 - score) * 100
                st.error(f"🚨 **RESULT: FAKE (AI-GENERATED)**")
                st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
                st.write("The model detected digital artifacts and synthetic patterns.")
            else:
                confidence = score * 100
                st.success(f"✅ **RESULT: REAL PHOTOGRAPH**")
                st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
                st.write("The model believes this is a genuine human photograph.")

        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.info("If you get a '403 Error', try opening the site in a Private/Incognito window.")

else:
    st.warning("Model is currently offline. Please check your Hugging Face Space files.")
