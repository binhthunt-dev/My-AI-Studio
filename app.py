import streamlit as st
import os
import io
from PIL import Image
import numpy as np
import cv2
from rembg import remove
from streamlit_drawable_canvas import st_canvas
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image

# --- CONFIGURATION & CONSTANTS ---
st.set_page_config(layout="wide", page_title="Pro AI Interior Studio", page_icon="✨")

# --- AUTHENTICATION ---
USERS = {
    "admin": "pass_admin",
    "friend1": "pass_123"
}

def check_password():
    """Returns `True` if the user had a correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in USERS and st.session_state["password"] == USERS[st.session_state["username"]]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password check failed, show inputs for username + password.
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()

# --- GOOGLE DRIVE HELPER ---
@st.cache_resource
def get_drive_service():
    """Setup Google Drive Service."""
    if "gcp_service_account" in st.secrets:
        # Load from Streamlit Secrets
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=creds)
        return service
    else:
        st.warning("⚠️ Google Drive Secrets not found. Using local mock data.")
        return None

def list_files_in_folder(folder_name):
    """List files in a specific folder by name."""
    service = get_drive_service()
    if not service:
        return [] # Mock return empty or local list
    
    try:
        # Find folder ID
        query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])
        
        if not items:
            st.warning(f"Folder '{folder_name}' not found.")
            return []
        
        folder_id = items[0]['id']
        
        # List files in folder
        query_files = f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false"
        results_files = service.files().list(q=query_files, fields="nextPageToken, files(id, name, mimeType)").execute()
        return results_files.get('files', [])
    except Exception as e:
        st.error(f"Error accessing Drive: {e}")
        return []

def download_file_from_drive(file_id):
    """Download image from Drive to memory."""
    service = get_drive_service()
    if not service:
        return None
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        return fh
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None

# --- AI MODELS ---
@st.cache_resource
def load_models():
    """Load Diffusers models. WARNING: High RAM usage."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=dtype)
        # Load Pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype, safety_checker=None
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            
        # Img2Img Pipeline (reuses components to save RAM if possible, or load separately)
        # For simplicity in Streamlit Cloud (low RAM), we might need to rely on one pipeline or swap.
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=dtype, safety_checker=None
        )
        if device == "cuda":
            img2img_pipe.enable_model_cpu_offload()

        return pipe, img2img_pipe, device
    except Exception as e:
        st.error(f"Failed to load models (likely OOM on Cloud Free Tier): {e}")
        return None, None, "cpu"

# --- MAIN APP UI ---
st.title("✨ Pro AI Interior Studio")
st.markdown("Automated Design & Rendering System")

# Initialize models (Lazy load to avoid startup crash if not needed immediately)
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

# Sidebar
with st.sidebar:
    st.header("🎨 Design Styles")
    input_source = st.radio("Source", ["Google Drive", "Upload Local"])
    
    selected_style_image = None
    if input_source == "Google Drive":
        if st.secrets.get("gcp_service_account"):
            folder_name = "My_AI_Design_Studio" # Hardcoded per requirement logic, or config
            files = list_files_in_folder(folder_name)
            if files:
                style_file = st.selectbox("Select Style Image", files, format_func=lambda x: x['name'])
                if style_file:
                    file_content = download_file_from_drive(style_file['id'])
                    if file_content:
                        selected_style_image = Image.open(file_content)
                        st.image(selected_style_image, caption="Reference Style", use_column_width=True)
            else:
                st.info("No files found or Drive not connected.")
        else:
            st.warning("Configure Secrets for Drive access.")
    else:
        uploaded_style = st.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])
        if uploaded_style:
            selected_style_image = Image.open(uploaded_style)
            st.image(selected_style_image, caption="Uploaded Style", use_column_width=True)

# Tabs
tab_photo, tab_sketch, tab_canvas, tab_adv = st.tabs(["📸 Photo Mode", "✏️ Sketch Mode", "🎨 In-painting", "🚀 Advanced"])

# --- PHOTO MODE (Img2Img) ---
with tab_photo:
    st.header("Photos to AI Render")
    input_img = st.file_uploader("Upload Room Photo", type=["png", "jpg"], key="photo_input")
    prompt = st.text_area("Prompt", "Modern living room, cinematic lighting, 8k uhd, photorealistic")
    strength = st.slider("Transformation Strength", 0.0, 1.0, 0.5)
    
    if st.button("Generate Render", key="btn_photo"):
        if input_img and prompt:
            with st.spinner("Rendering... (This may take time on CPU)"):
                # Load models only when needed
                _, img2img_pipe, device = load_models()
                if img2img_pipe:
                    init_image = Image.open(input_img).convert("RGB").resize((512, 512))
                    result = img2img_pipe(prompt=prompt, image=init_image, strength=strength).images[0]
                    st.image(result, caption="AI Result")
                    
                    # Add to history/canvas
                    buf = io.BytesIO()
                    result.save(buf, format="PNG")
                    st.download_button("Download Result", buf.getvalue(), "render.png", "image/png")

# --- SKETCH MODE (ControlNet) ---
with tab_sketch:
    st.header("Sketch to Realism")
    sketch_input = st.file_uploader("Upload Sketch/Lineart", type=["png", "jpg"], key="sketch_input")
    sketch_prompt = st.text_area("Sketch Prompt", "Luxury bedroom, marble floor, 8k uhd, photorealistic, cinematic lighting")
    
    if st.button("Render Sketch", key="btn_sketch"):
        if sketch_input and sketch_prompt:
            with st.spinner("Processing Sketch..."):
                pipe, _, device = load_models()
                if pipe:
                    sketch_img = Image.open(sketch_input).convert("RGB").resize((512, 512))
                    # Preprocess for Canny if needed, or just Scribble
                    # For simplicity using direct scribble model or Canny if input is cleaner
                    # Here assuming user uploads ready lineart or we process it
                    # result = pipe(sketch_prompt, image=sketch_img).images[0] # Implementation detail depends on model
                    # Using Scribble
                    result = pipe(prompt=sketch_prompt, image=sketch_img).images[0]
                    st.image(result, caption="AI Rendered Sketch")

# --- IN-PAINTING / CANVAS ---
with tab_canvas:
    st.header("Editor & In-painting")
    
    # Initialize background for canvas
    bg_image = None
    if st.session_state.get("last_result"):
        bg_image = st.session_state["last_result"]
    else:
        bg_upload = st.file_uploader("Background for Editing", type=["png", "jpg"], key="canvas_bg")
        if bg_upload:
            bg_image = Image.open(bg_upload).convert("RGB")

    if bg_image:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=10,
            stroke_color="#fff",
            background_image=bg_image,
            update_streamlit=True,
            height=512,
            width=512,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if canvas_result.image_data is not None:
            # Mask generation
            mask = canvas_result.image_data[:, :, 3] # Alpha channel
            if st.button("Inpaint Selection"):
                st.info("Inpainting logic placeholder (Requires Inpaint Pipeline)")
                # To implement: load StableDiffusionInpaintPipeline
                
# --- ADVANCED (Moodboard / Object Remove) ---
with tab_adv:
    st.header("Advanced Tools")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Background Removal")
        obj_img = st.file_uploader("Upload Object", type=["png", "jpg"], key="obj_rembg")
        if obj_img:
            input_obj = Image.open(obj_img)
            if st.button("Remove Background"):
                output_obj = remove(input_obj)
                st.image(output_obj, caption="Transparent Object")
    
    with col2:
        st.subheader("Moodboard Extractor")
        # Logic to extract colors from selected style image
        if selected_style_image:
            st.image(selected_style_image, width=150, caption="Current Style Used")
            if st.button("Analyze Colors"):
                # Simple K-Means or just average color mock
                img_array = np.array(selected_style_image)
                avg_color = img_array.mean(axis=(0,1))
                st.color_picker("Dominant Color", f"#{int(avg_color[0]):02x}{int(avg_color[1]):02x}{int(avg_color[2]):02x}")
        else:
            st.info("Select a style from Sidebar first.")

st.markdown("---")
st.caption("Powered by Streamlit & Diffusers")
