import os
import cv2
import torch
import numpy as np
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet
from matplotlib import pyplot as plt

# ===============================
# PATH CONFIGURATION
# ===============================
# Get the absolute path to the backend/ml directory (parent of 'notebooks')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "raw")

MODEL_PATH = os.path.join(MODELS_DIR, "efficientdet_keyboard.pth")
TEST_IMAGE_PATH = os.path.join(
    DATASET_DIR, "test", "dell5_jpeg.rf.e955243c762bd24a2721306b2408e0d1.jpg"
)

# ===============================
# MODEL LOADING
# ===============================
def load_model(pth_path, num_classes=2, device='cuda'):
    config = get_efficientdet_config('tf_efficientdet_d0')
    config.num_classes = num_classes
    config.image_size = (512, 512)

    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    model = DetBenchPredict(net)

    # ✅ Use weights_only=True for safer loading (torch >= 2.4)
    state_dict = torch.load(pth_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.eval().to(device)
    print(f"✅ Model loaded from {pth_path}")
    return model


# ===============================
# INFERENCE FUNCTION
# ===============================
def infer_and_show(model, image_path, device='cuda', conf_thresh=0.01, save_output=True):
    """Run inference on a single image and display/save the results."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Preprocess: resize and normalize 0–1
    img_resized = cv2.resize(image_rgb, (512, 512)) / 255.0
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    out = outputs[0].detach().cpu().numpy()

    print("\n🔍 Raw model output shape:", outputs[0].shape)
    print("First 5 detections:\n", out[:5])

    if out.shape[0] == 0:
        print("⚠️ No detections returned by model.")
        return

    boxes = out[:, :4]
    scores = out[:, 4]
    labels = out[:, 5].astype(int)

    drawn = 0
    for i, box in enumerate(boxes):
        if scores[i] >= conf_thresh:
            drawn += 1
            x1, y1, x2, y2 = box
            # Scale back to original image size
            x1, x2 = x1 / 512 * w, x2 / 512 * w
            y1, y2 = y1 / 512 * h, y2 / 512 * h
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"Key {labels[i]} ({scores[i]:.3f})",
                (int(x1), max(10, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    print(f"🟩 Boxes drawn: {drawn} (threshold={conf_thresh})")
    if drawn == 0:
        print("⚠️ No boxes passed the threshold. Try lowering conf_thresh to 0.001 for debugging.")

    # Visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Optional save
    if save_output:
        out_path = os.path.splitext(image_path)[0] + "_pred.jpg"
        cv2.imwrite(out_path, image)
        print(f"💾 Saved output to: {out_path}")


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_PATH, num_classes=2, device=device)

    # 💡 Lowered conf_thresh to 0.01 by default to help catch low-confidence boxes
    infer_and_show(model, TEST_IMAGE_PATH, device=device, conf_thresh=0.001)
