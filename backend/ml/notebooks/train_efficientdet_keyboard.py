import os
import cv2
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from tqdm import tqdm

# ===============================
# CONFIGURATION
# ===============================
def find_project_root() -> str:
    """Automatically find the dataset root path"""
    cwd = os.getcwd(); c = [cwd]; cur = cwd
    for _ in range(4):
        cur = os.path.dirname(cur)
        if cur and cur not in c: c.append(cur)
    for base in c:
        if os.path.isdir(os.path.join(base, "backend", "ml", "dataset", "raw")):
            return os.path.join(base, "backend", "ml")
    return cwd

project_root = find_project_root()
dataset_root = os.path.join(project_root, "dataset", "raw")
models_dir = os.path.join(project_root, "models")

# Create models folder if not existing
os.makedirs(models_dir, exist_ok=True)

train_folder = os.path.join(dataset_root, "train")
val_folder = os.path.join(dataset_root, "test")

class_map = {"key": 1}
num_classes = len(class_map) + 1  # +1 for background

# ===============================
# DATASET
# ===============================
class VOCDataset(Dataset):
    def __init__(self, folder, class_map, transforms=None):
        self.folder = folder
        self.images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transforms = transforms
        self.class_map = class_map

    def __len__(self):
        return len(self.images)

    def parse_xml(self, xml_path):
        boxes, labels = [], []
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            label = obj.find("name").text
            label_id = self.class_map.get(label, 0)
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_id)
        return boxes, labels

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.folder, img_name)
        xml_path = os.path.splitext(img_path)[0] + ".xml"

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels = self.parse_xml(xml_path)

        # Skip corrupted images or invalid XMLs
        if image is None or len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]

        if self.transforms:
            try:
                transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)
                image = transformed["image"]
                boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)
            except Exception as e:
                print(f"Skipping {img_name} due to transform error: {e}")
                boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
                labels = torch.tensor([0], dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # ✅ Ensure boxes always have correct shape [N, 4]
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            print(f"Invalid box shape {boxes.shape} in {img_name}, fixing...")
            boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            labels = torch.tensor([0], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return image, target

# ===============================
# AUGMENTATION
# ===============================
def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=15,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT
        ),
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        min_visibility=0.1,
        clip=True
    ))

def get_valid_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        clip=True
    ))

# ===============================
# UTILITIES
# ===============================
def collate_fn(batch):
    return tuple(zip(*batch))

# ===============================
# MODEL SETUP
# ===============================
def create_model(num_classes):
    config = get_efficientdet_config('tf_efficientdet_d0')
    config.num_classes = num_classes
    config.image_size = (512, 512)
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    model = DetBenchTrain(net, config)
    return model

# ===============================
# TRAINING LOOP
# ===============================
def train_model(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in pbar:
            images = torch.stack([img for img in images]).to(device)
            boxes = [t["boxes"].to(device) for t in targets]
            labels = [t["labels"].to(device) for t in targets]

            optimizer.zero_grad()
            loss = model(images, {"bbox": boxes, "cls": labels})
            loss["loss"].backward()
            optimizer.step()

            running_loss += loss["loss"].item()
            pbar.set_postfix(loss=loss["loss"].item())

        print(f"✅ Epoch {epoch+1} Average Loss: {running_loss/len(dataloader):.4f}")

# ===============================
# MAIN EXECUTION
# ===============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = VOCDataset(train_folder, class_map, transforms=get_train_transforms())
    val_dataset = VOCDataset(val_folder, class_map, transforms=get_valid_transforms())

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = create_model(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Train the model
    train_model(model, train_loader, optimizer, device, num_epochs=10)

    # ===============================
    # SAVE MODELS (PyTorch + ONNX)
    # ===============================
    print("Saving trained models...")

    pth_path = os.path.join(models_dir, "efficientdet_keyboard.pth")
    onnx_path = os.path.join(models_dir, "efficientdet_keyboard.onnx")

    # Save PyTorch weights
    torch.save(model.state_dict(), pth_path)
    print(f"Saved PyTorch weights: {pth_path}")

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    torch.onnx.export(
        model.module if hasattr(model, "module") else model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True
    )

    print(f"Exported ONNX model: {onnx_path}")
    print("Training and export complete!")

if __name__ == "__main__":
    main()
