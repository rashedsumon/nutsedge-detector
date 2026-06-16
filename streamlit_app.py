import os
import streamlit as st
from PIL import Image
from data_loader import load_and_prepare_data
from model import predict_image

# 1. Page Configuration
st.set_page_config(
    page_title="Nutsedge Weed Detector AI",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Nutsedge Weed Detector AI")
st.write("Upload a crop or field photo to detect the presence of invasive Nutsedge weeds.")

# 2. Lazy Load Data/Weights on Streamlit Cloud Startup
@st.cache_resource
def setup_application():
    try:
        # Downloads and extracts files into the root directory securely if needed
        load_and_prepare_data()
    except Exception as e:
        st.warning(f"Kaggle download skipped or unavailable in Cloud runtime: {e}")
    
    # Path where custom trained weights will live after training
    custom_model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
    return custom_model_path

model_weights = setup_application()

st.divider()

# 3. User Input Section
st.header("1. User Input")
uploaded_file = st.file_uploader("Choose a field image (JPG/PNG format)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display user input image
    input_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image Input")
        st.image(input_image, use_container_width=True)
        
    with col2:
        st.subheader("Model Output Prediction")
        with st.spinner("Analyzing image for weeds..."):
            # Save uploaded file temporarily to pass to the model
            temp_path = f"temp_{uploaded_file.name}"
            input_image.save(temp_path)
            
            # 4. Model Output Execution
            # Uses custom weights if available, fallback to yolov8n.pt if not trained yet
            output_array = predict_image(temp_path, model_path=model_weights)
            
            # Convert resulting BGR OpenCV array back to RGB for PIL displaying
            output_image = Image.fromarray(output_array[..., ::-1])
            
            # Display output
            st.image(output_image, use_container_width=True)
            
            # Cleanup temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    st.success("Analysis complete! Bounding boxes display detected Nutsedge weeds.")
else:
    st.info("💡 Please upload an image above to trigger the AI prediction model.")