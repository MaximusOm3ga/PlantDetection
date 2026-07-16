import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import datasets
from PIL import Image

from visionTrans import ViTModel

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import ViT_B_16_Weights

data_path = r"C:\Users\snagrath002\Downloads\archive\plantvillage dataset\color"
dataset = datasets.ImageFolder(root=data_path)

model_path = "vit_best.pth"
image_path = r"C:\Users\snagrath002\Downloads\archive\plantvillage dataset\color\Potato___Early_blight\f917c191-cca4-4a78-8238-a40b586f9058___RS_Early.B 7945.JPG"

num_classes = len(dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ViTModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

for p in model.parameters():
    p.requires_grad = True

target_layers = [model.model.encoder.layers[-1].ln_1]

def reshape_transform(tensor):
    tensor = tensor[:, 1:, :]
    h = w = int(tensor.shape[1] ** 0.5)
    tensor = tensor.reshape(tensor.size(0), h, w, tensor.size(2))
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


cam = GradCAMPlusPlus(
    model=model,
    target_layers=target_layers,
    reshape_transform=reshape_transform
)


weights = ViT_B_16_Weights.DEFAULT
transform = weights.transforms()

image = Image.open(image_path).convert("RGB")
image_resized = image.resize((224, 224))

input_tensor = transform(image).unsqueeze(0).to(device)

outputs = model(input_tensor)
probs = F.softmax(outputs, dim=1)[0]

pred_class = torch.argmax(probs).item()
confidence = probs[pred_class].item()

print(f"Prediction: {dataset.classes[pred_class]}")
print(f"Confidence: {confidence:.4f}")

targets = [ClassifierOutputTarget(pred_class)]

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

img_np = np.array(image_resized) / 255.0


visualization = show_cam_on_image(
    img_np,
    grayscale_cam[0],
    use_rgb=True
)

cv2.imshow("GradCAM", visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()