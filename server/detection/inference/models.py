import os, joblib, torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from config import BASE_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

def load_models():
    detector = YOLO(os.path.join(BASE_DIR, "models", "yolov8n-face.pt"))
    facenet = InceptionResnetV1(pretrained="vggface2").eval()
    facenet = facenet.to(device).half() if device.type == "cuda" else facenet.to(device)
    svm = joblib.load(os.path.join(BASE_DIR, "models", "svm_model.pkl"))
    encoder = joblib.load(os.path.join(BASE_DIR, "models", "in_encoder.pkl"))
    pca_path = os.path.join(BASE_DIR, "models", "pca_model.pkl")
    pca = joblib.load(pca_path) if os.path.exists(pca_path) else None

    # Optional normalizer
    try:
        from cuml.preprocessing import Normalizer
        normalizer = Normalizer(norm="l2")
    except Exception:
        normalizer = None

    return detector, facenet, svm, encoder, pca, normalizer, device