import os
import pandas as pd
from deepface import DeepFace
import numpy as np
from tqdm import tqdm

# Configurable paths
metadata_file = "face_metadata.csv"
images_dir = "images"
output_file = "embeddings_store.csv"

def generate_embeddings():
    df = pd.read_csv(metadata_file)
    embeddings = []

    print(f"Processing {len(df)} images for embedding generation...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["photo_path"]
        full_path = os.path.join(images_dir, os.path.basename(img_path))

        try:
            embedding_obj = DeepFace.represent(img_path=full_path, model_name="Facenet", enforce_detection=True)
            embedding = embedding_obj[0]["embedding"]  # 128-d vector
            row_dict = row.to_dict()
            for i in range(128):
                row_dict[f"emb_{i}"] = embedding[i]
            embeddings.append(row_dict)
        except Exception as e:
            print(f"❌ Failed to process {full_path}: {e}")

    result_df = pd.DataFrame(embeddings)
    result_df.to_csv(output_file, index=False)
    print(f"\n✅ Done! Saved embeddings to: {output_file}")

if __name__ == "__main__":
    generate_embeddings()