import torch
from torchvision import transforms
from backbones import get_model
from detection import detect_faces

# Function to load the model
def load_model(model_name="edgeface_s_gamma_05", checkpoint_dir="checkpoints", device='cpu'):
    model = get_model(model_name)
    checkpoint_path = f'{checkpoint_dir}/{model_name}.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model

# Function to extract embeddings
def extract_embedding(model, image_path, face_model, device='cpu'):
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Detect face and align
    aligned = detect_faces(image_path, trained_model=face_model, cpu=True, 
                           confidence_threshold=0.05, top_k=5000, nms_threshold=0.3,
                           keep_top_k=750, vis_thres=0.5)
    
    if aligned is None:
        print(f"Face alignment failed for image: {image_path}")
        return None
    
    # Preprocess and add batch dimension
    transformed_input = transform(aligned).unsqueeze(0).to(device)
    
    # Extract embedding
    with torch.no_grad():
        embedding = model(transformed_input)
    
    return embedding
