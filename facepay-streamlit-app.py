import sys
import os
import io
# Set environment variables before importing other libraries
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

import streamlit as st

# ------------------------- PAGE SETUP (FIRST) ------------------------- #
st.set_page_config(page_title="FacePay Verification", layout="centered")
st.title("🔐 Face Authentication — FacePay")
st.markdown("Upload your face image below to verify your identity.")

# Show startup status
startup_status = st.empty()
startup_status.info("🚀 Starting FacePay application...")

# ------------------------- BASIC IMPORTS FIRST ------------------------- #
import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient
from tempfile import NamedTemporaryFile
from PIL import Image
import json

startup_status.info("📦 Loading basic libraries...")

# ------------------------- CONFIG ------------------------- #
DEBUG = True
EMBEDDING_MODEL = "Facenet"

# ------------------------- DEBUG LOGGER ------------------------- #
def debug_log(message):
    if DEBUG:
        print(f"[DEBUG] {message}")
        st.text(f"[DEBUG] {message}")

# ------------------------- LAZY IMPORTS ------------------------- #
def load_cv2():
    """Lazy load OpenCV"""
    try:
        import cv2
        debug_log("✅ OpenCV loaded")
        return cv2
    except ImportError as e:
        st.error(f"Failed to load OpenCV: {e}")
        return None

def load_deepface():
    """Lazy load DeepFace"""
    try:
        from deepface import DeepFace
        debug_log("✅ DeepFace loaded")
        return DeepFace
    except ImportError as e:
        st.error(f"Failed to load DeepFace: {e}")
        return None

def load_scipy():
    """Lazy load scipy"""
    try:
        from scipy.spatial.distance import cosine
        debug_log("✅ Scipy loaded")
        return cosine
    except ImportError as e:
        st.error(f"Failed to load Scipy: {e}")
        return None

# ------------------------- AZURE STORAGE CONFIGURATION ------------------------- #
def load_storage_config():
    """Load Azure Storage configuration from config.json"""
    try:
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                config = json.load(f)
            debug_log("✅ Config.json loaded")
            return config
        else:
            debug_log("❌ config.json not found")
            return {}
    except Exception as e:
        debug_log(f"❌ Error reading config.json: {e}")
        return {}

def setup_direct_storage_connection():
    """Setup direct Azure Storage connection"""
    try:
        config = load_storage_config()
        
        # Try to get connection string from config
        connection_string = config.get('azure_storage_connection_string')
        if not connection_string:
            debug_log("❌ azure_storage_connection_string not found in config.json")
            return None, None
            
        container_name = config.get('container_name', 'facepay-data')
        
        # Create blob service client
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service.get_container_client(container_name)
        
        debug_log(f"✅ Connected to Azure Storage container: {container_name}")
        
        return blob_service, container_name
        
    except Exception as e:
        debug_log(f"❌ Storage connection error: {str(e)}")
        return None, None

# Initialize storage connection
startup_status.info("🔗 Connecting to Azure Storage...")
blob_service, container_name = setup_direct_storage_connection()

# Configuration for blob file names
EMBEDDING_STORE_BLOB = "embeddings_store.csv"
METADATA_BLOB = "face_metadata.csv"

# ------------------------- AZURE BLOB STORAGE HELPERS ------------------------- #
@st.cache_data(ttl=300)  # Cache for 5 minutes
def download_blob_to_dataframe(blob_name):
    """Download a CSV blob and return as pandas DataFrame"""
    try:
        if not blob_service or not container_name:
            debug_log("❌ Azure Storage not connected")
            return pd.DataFrame()
            
        blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
        
        debug_log(f"Downloading {blob_name} from Azure Blob Storage...")
        
        # Download blob content
        blob_data = blob_client.download_blob()
        blob_content = blob_data.readall()
        
        # Convert to DataFrame
        df = pd.read_csv(io.StringIO(blob_content.decode('utf-8')))
        debug_log(f"Successfully loaded {len(df)} rows from {blob_name}")
        return df
        
    except Exception as e:
        debug_log(f"Blob download error for {blob_name}: {str(e)}")
        return pd.DataFrame()

# ------------------------- LOAD DATA FROM AZURE ------------------------- #
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_metadata():
    """Load metadata from Azure Blob Storage"""
    return download_blob_to_dataframe(METADATA_BLOB)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_embeddings():
    """Load embeddings from Azure Blob Storage"""
    return download_blob_to_dataframe(EMBEDDING_STORE_BLOB)

# ------------------------- EMBEDDING HANDLERS (LAZY LOADED) ------------------------- #
def compute_embedding(image_path):
    """Compute face embedding - loads DeepFace on first call"""
    DeepFace = load_deepface()
    if not DeepFace:
        raise Exception("DeepFace not available")
    
    return DeepFace.represent(img_path=image_path, model_name=EMBEDDING_MODEL, enforce_detection=False)[0]["embedding"]

def get_closest_match(uploaded_emb, embedding_df, threshold=0.3):
    """Find closest face match - loads scipy on first call"""
    cosine = load_scipy()
    if not cosine:
        raise Exception("Scipy not available")
        
    best_match = None
    lowest_dist = float("inf")
    for _, row in embedding_df.iterrows():
        try:
            stored_emb = np.array([row[f"e_{i}"] for i in range(128)])
            dist = cosine(uploaded_emb, stored_emb)
            if dist < lowest_dist and dist < threshold:
                lowest_dist = dist
                best_match = row["photo"]
        except KeyError as e:
            debug_log(f"Missing embedding column: {e}")
    return best_match

# ------------------------- FACE MATCH WORKFLOW ------------------------- #
def process_verification(image_bytes, df, emb_df):
    # Load heavy libraries only when needed
    cv2 = load_cv2()
    if not cv2:
        st.error("OpenCV not available")
        return
        
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(image_bytes)
        uploaded_path = tmp_file.name
    debug_log(f"Saved uploaded image to {uploaded_path}")

    try:
        st.image(image_bytes, caption="Uploaded Image", width=200)
        with st.spinner("🔍 Analyzing image and comparing with database..."):
            uploaded_emb = compute_embedding(uploaded_path)
            debug_log("Computed uploaded face embedding.")

            if emb_df.empty:
                st.warning("No embeddings found in database. Cannot perform comparison.")
                return

            match_filename = get_closest_match(uploaded_emb, emb_df)

            if match_filename:
                photo_name = os.path.splitext(match_filename)[0]
                matches = df[df["photo_path"].str.contains(photo_name, case=False, na=False)]
                
                if not matches.empty:
                    match_row = matches.iloc[0]
                    display_results(match_row, image_bytes)
                
                else:
                    debug_log(f"No metadata row found for matched filename: {match_filename}")
                    st.warning("✅ Face matched in embeddings, but metadata was not found.")
            else:
                debug_log("No close match found in embeddings.")
                st.info("❌ Sorry, we couldn't verify your identity. Please try again with a clearer, frontal photo or contact support if the issue persists.")    

    except Exception as e:
        st.error(f"⚠️ Error during face verification: {str(e)}")
        debug_log(str(e))

    finally:
        os.remove(uploaded_path)

# ------------------------- DISPLAY RESULTS ------------------------- #
def display_results(row, image_bytes):
    st.success("✅ Face Match Found!")
    st.image(image_bytes, caption="Matched Image", width=200)
    st.markdown(f"""
    - **Name**: {row['name']}
    - **DOB**: {row['dob']}
    - **Sex**: {row['sex']}
    - **Address**: {row['address']}
    - **Account Number**: `{row['account_number']}`
    - **Balance**: `${row['balance']:.2f}`
    """)

# ------------------------- CONNECTION TEST ------------------------- #
def test_storage_connection():
    """Test Azure Storage connection"""
    try:
        if not blob_service or not container_name:
            st.error("❌ Azure Storage connection not established")
            return False
            
        container_client = blob_service.get_container_client(container_name)
        blob_list = list(container_client.list_blobs())
        st.success(f"✅ Connected to Azure Storage. Found {len(blob_list)} blobs in container '{container_name}'.")
        
        # Show available blobs
        if blob_list:
            st.write("**Available files in container:**")
            for blob in blob_list:
                st.write(f"- {blob.name}")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Storage connection test failed: {str(e)}")
        return False

# ------------------------- APP EXECUTION ------------------------- #
# Clear startup status
startup_status.success("✅ FacePay application started successfully!")

# Test connection button
if st.sidebar.button("🔗 Test Storage Connection"):
    test_storage_connection()

# Show configuration status
if blob_service and container_name:
    st.success("✅ Azure Storage connection established")
else:
    st.error("❌ Azure Storage connection failed. Check your config.json file.")

# Load data from Azure
if blob_service and container_name:
    debug_log("Loading data from Azure Storage...")
    with st.spinner("📥 Loading data from Azure Blob Storage..."):
        df = load_metadata()
        embeddings_df = load_embeddings()
else:
    df = pd.DataFrame()
    embeddings_df = pd.DataFrame()

# Show data status
if df.empty:
    st.error("❌ Could not load metadata from Azure Blob Storage")
else:
    st.success(f"✅ Loaded {len(df)} user records from Azure")

if embeddings_df.empty:
    st.error("❌ Could not load embeddings from Azure Blob Storage")
else:
    st.success(f"✅ Loaded {len(embeddings_df)} face embeddings from Azure")

# File upload and processing
uploaded_image = st.file_uploader("📤 Upload a face image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
submit = st.button("🔍 Match Face")

if submit and uploaded_image:
    if df.empty or embeddings_df.empty:
        st.error("Cannot process verification - data not loaded from Azure")
    else:
        process_verification(uploaded_image.read(), df, embeddings_df)