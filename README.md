# 🧠 FacePay - Streamlit-based Facial Authentication App

FacePay is a Streamlit application that performs facial authentication using DeepFace and Azure Blob Storage. Once a face is verified, it fetches personal metadata and simulates user verification for payments in systems like automated checkouts.

## 🚀 Features

- Facial recognition using DeepFace
- Embedding comparison with cosine similarity
- Metadata and embeddings loaded from Azure Blob Storage
- Azure configuration via `config.json`
- Optimized for deployment on Azure App Service

## 🗃 Directory Structure

```
facepay-streamlit-app/
├── facepay_streamlit_app.py    # Main Streamlit application
├── config.json                 # Azure Blob Storage connection details
├── requirements.txt            # Python dependencies
├── embeddings_store.csv        # Precomputed face embeddings
├── face_metadata.csv           # Metadata linked to each face image
└── README.md                   # Project documentation
```

## 🧪 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/facepay-streamlit-app.git
   cd facepay-streamlit-app
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Add your Azure connection info to `config.json`:
   ```json
   {
     "azure_storage_connection_string": "YOUR_CONNECTION_STRING",
     "container_name": "YOUR_CONTAINER_NAME"
   }
   ```

5. Run the app:
   ```bash
   streamlit run facepay_streamlit_app.py
   ```

## 🌐 Deployment

Deploy on Azure App Service by:
- Zipping your directory
- Using Azure CLI:
  ```bash
  az webapp deploy --resource-group <rg-name> --name <app-name> --src-path facepay.zip --type zip
  ```

## 📦 Dependencies

- streamlit
- deepface
- pandas
- Pillow
- azure-storage-blob
- scipy
- numpy
- opencv-python

## 🔐 Disclaimer

This app is intended for demonstration purposes and may not meet production-grade security requirements without further hardening.

---

## 📧 Contact

For queries, reach out at [your_email@example.com]
