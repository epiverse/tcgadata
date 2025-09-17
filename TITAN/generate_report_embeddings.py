import pickle
import json
from huggingface_hub import hf_hub_download, login
import numpy as np  # Added to handle NumPy arrays

# Log in using the token
login(token="add tokn here")

# Download the precomputed embeddings
slide_feature_path = hf_hub_download(
    "MahmoodLab/TITAN",
    filename="TCGA_TITAN_features.pkl",
)

# Load the embeddings
with open(slide_feature_path, 'rb') as file:
    data = pickle.load(file)

# Custom function to convert NumPy arrays to lists for JSON serialization
def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Save data as a JSON file
output_file = "tcga_titan_embeddings.json"
with open(output_file, 'w') as json_file:
    json.dump(data, json_file, default=convert_to_json_serializable, indent=4)

# Inspect the data
print(data)