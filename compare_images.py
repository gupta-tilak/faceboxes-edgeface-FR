# compare_images.py

import torch
from torch.nn.functional import cosine_similarity
from embeddings import load_model, extract_embedding

# Load the model
model_name = "edgeface_s_gamma_05"  # or "edgeface_xs_gamma_06"
model = load_model(model_name)

# Define image paths
image1_path = 'images/RahulAwasthy_2.jpeg'
image2_path = 'images/Rajath_14.jpg'

# Extract embeddings using the function from the external module
embedding1 = extract_embedding(model, image1_path, face_model='weights/FaceBoxes.pth')
embedding2 = extract_embedding(model, image2_path, face_model='weights/FaceBoxes.pth')

if embedding1 is not None and embedding2 is not None:
    # Compute cosine similarity
    similarity = cosine_similarity(embedding1, embedding2).item()
    print(f"Cosine Similarity: {similarity:.4f}")
else:
    print("Failed to compute embeddings for one or both images.")