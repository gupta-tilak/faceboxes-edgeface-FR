import matplotlib.pyplot as plt
import cv2
from detection import detect_faces  # Replace 'main_script' with the actual script name

if __name__ == '__main__':
    image_path = '/Users/guptatilak/Documents/C4GT-Face-Recognition/offline-FR/faceboxes-edgeface-FR/Aisvarrya_9.jpeg'
    highest_score_face = detect_faces(image_path, trained_model='weights/FaceBoxes.pth', cpu=True,
                                      confidence_threshold=0.05, top_k=5000, nms_threshold=0.3,
                                      keep_top_k=750, vis_thres=0.5)
    
    if highest_score_face is not None:
        plt.imshow(cv2.cvtColor(highest_score_face, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        print("No face detected.")
