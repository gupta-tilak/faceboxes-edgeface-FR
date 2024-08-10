import torch
from torchvision import transforms
from backbones import get_model
from detection import detect_faces
from PIL import Image

# Load the model
model_name = "edgeface_s_gamma_05"  # or "edgeface_xs_gamma_06"
model = get_model(model_name)
checkpoint_path = f'checkpoints/{model_name}.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

path = '/Users/guptatilak/Documents/C4GT-Face-Recognition/offline-FR/faceboxes-edgeface-FR/images/Aisvarrya_9.jpeg'
# aligned = align.get_aligned_face(path)  # Align face
aligned = detect_faces(path, trained_model='weights/FaceBoxes.pth', cpu=True,
                                      confidence_threshold=0.05, top_k=5000, nms_threshold=0.3,
                                      keep_top_k=750, vis_thres=0.5)

if aligned is None:
    print("Face alignment failed.")
else:
    transformed_input = transform(aligned).unsqueeze(0)  # Preprocess and add batch dimension

    # Extract embedding
    with torch.no_grad():
        embedding = model(transformed_input)

print(embedding)