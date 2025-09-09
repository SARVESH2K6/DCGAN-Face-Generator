import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Face Generator",
    page_icon="🤖",
    layout="wide",
)

# --- Model Loading ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_gan_model():
    """Loads the trained Keras generator model."""
    try:
        model = tf.keras.models.load_model('face_generator_model.h5')
        return model
    except (IOError, FileNotFoundError):
        st.error("Model file not found. Please make sure 'face_generator_model.h5' is available.")
        return None

generator = load_gan_model()
LATENT_DIM = 100 # Must match the training script

# --- UI Elements ---
st.title("🤖 AI-Powered Face Generator")
st.markdown(
    "Welcome! This application uses a Generative Adversarial Network (GAN) to create unique, "
    "artificial human faces. The model was trained from scratch on a dataset of real faces. "
    "Click the button below to generate a new, never-before-seen face!"
)

st.divider()

# --- Image Generation Logic ---
col1, col2 = st.columns([1, 2])

with col1:
    if st.button("✨ Generate New Face", type="primary", use_container_width=True):
        if generator is not None:
            # Show a spinner while the model is working
            with st.spinner('Generating...'):
                # 1. Create random noise
                noise = tf.random.normal([1, LATENT_DIM])
                
                # 2. Generate image from noise
                generated_image = generator(noise, training=False)
                
                # 3. Post-process for display
                img_display = (generated_image[0].numpy() + 1) * 127.5
                img_display = img_display.astype(np.uint8)
                
                # Store the image in session state to persist it
                st.session_state.generated_image = Image.fromarray(img_display)
        else:
            st.warning("Generator model is not loaded. Cannot generate image.")

with col2:
    st.markdown("### Generated Image:")
    
    # Display the image if it exists in the session state
    if 'generated_image' in st.session_state:
        st.image(
            st.session_state.generated_image,
            caption="This person does not exist.",
            width=300 # Display image larger for better viewing
        )
    else:
        st.info("Click the button on the left to generate your first face!")
