from visionTrans import ViTModel as vit

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.models import ViT_B_16_Weights
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_path = r"C:\Users\sauri\PycharmProjects\PlantDetection\RunScripts\RunScripts\plantvillvalds\PlantVillage\val"
checkpoint_path = "vit_best.pth"

weights = ViT_B_16_Weights.DEFAULT
transform = weights.transforms()

test_data = datasets.ImageFolder(
    root=test_path,
    transform=transform
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
)

num_classes = len(test_data.classes)

print("Number of classes:", num_classes)
print("Number of test images:", len(test_data))

model = vit(num_classes=num_classes).to(device)

state_dict = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

def vit_anom_det(path):
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    predicted_class = test_data.classes[pred_idx]

    return predicted_class, confidence

if __name__ == "__main__":
    y_true = []
    y_pred = []

    with torch.inference_mode():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=test_data.classes,
            digits=4,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(num_classes))
        )
    )