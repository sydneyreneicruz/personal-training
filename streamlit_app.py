import streamlit as st
import gradio as gr
from fastai.vision.all import load_learner
from PIL import Image
import threading

# 1. Page Configuration
st.set_page_config(page_title="Fastai Deployment", layout="wide")
st.title("🚀 Fastai Model Deployment")
st.subheader("Powered by Gradio Components inside Streamlit")

# 2. Load Your Saved Fastai Model
@st.cache_resource
def get_model():
    # Ensure 'export.pkl' is in your working directory
    return load_learner('model.pkl')

try:
    learn = get_model()
except Exception as e:
    st.error(f"Could not load the model file. Ensure 'export.pkl' exists. Error: {e}")
    st.stop()

# 3. Define the Prediction Function
def predict_image(img):
    # Convert incoming PIL Image or numpy array to fastai format
    pred, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(learn.dls.vocab))}

# 4. Initialize Gradio Interface
def run_gradio():
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
        title="Gradio Backend Classifier",
        flagging_mode="never"
    )
    # Launch on a specific local port quietly
    interface.launch(server_port=7860, prevent_thread_lock=True, share=False)

# Start Gradio background server once per session lifecycle
if 'gradio_started' not in st.session_state:
    threading.Thread(target=run_gradio, daemon=True).start()
    st.session_state['gradio_started'] = True

# 5. Build the Streamlit Layout and Embed Gradio
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Application Information")
    st.write("This application wraps a `fastai` image classification model.")
    st.info("The prediction processing is executed via a local Gradio server instance running on the backend.")

with col2:
    st.markdown("### Interactive Model Demo")
    # Embed the running Gradio server using Streamlit's iframe component
    st.components.v1.iframe(src="http://localhost:7860", height=500, scrolling=True)