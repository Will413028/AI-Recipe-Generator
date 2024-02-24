import cv2
import numpy as np
import torch
from super_gradients.training import models
from config import CLASS_NAMES, MODEL_CHECKPOINT_PATH

def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.get('yolo_nas_s', num_classes=26, checkpoint_path=MODEL_CHECKPOINT_PATH).to(device)
    return model

def process_image(image_data, model):
    np_image = np.array(image_data)
    frame = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    class_final_names = []

    result = model.predict(frame, conf=0.30)
    if result.prediction is not None:
        for cls in result.prediction.labels:
            class_name = CLASS_NAMES[int(cls)]
            class_final_names.append(class_name)

    ingredients_detected = ','.join(np.unique(class_final_names)) if class_final_names else ''
    return frame, ingredients_detected
