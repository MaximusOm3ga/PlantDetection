import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt

from cnn import CNN

import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

data_path = r"C:\Users\snagrath002\Downloads\archive\plantvillage dataset\color"
image_path = r"C:\Users\snagrath002\Downloads\archive\plantvillage dataset\color\Potato___Early_blight\f917c191-cca4-4a78-8238-a40b586f9058___RS_Early.B 7945.JPG"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = datasets.ImageFolder(root=data_path)
class_names = dataset.classes
num_classes = len(class_names)

model = CNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("cnn_classifier_best.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

image = Image.open(image_path).convert("RGB")
image_resized = image.resize((128,128))
image_np = np.array(image_resized)

def predict_fn(images):
    images = torch.tensor(images).permute(0,3,1,2).float().to(device)
    images = (images/255.0 - 0.5)/0.5
    outputs = model(images)
    probs = F.softmax(outputs, dim=1)
    return probs.detach().cpu().numpy()

input_tensor = transform(image).unsqueeze(0).to(device)
output = model(input_tensor)
pred_class = torch.argmax(output).item()

print("Prediction:", class_names[pred_class])

explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(
    image_np,
    predict_fn,
    top_labels=1,
    num_samples=1000
)

temp, mask = explanation.get_image_and_mask(
    pred_class,
    positive_only=True,
    num_features=5,
    hide_rest=False
)

plt.imshow(mark_boundaries(temp, mask))
plt.title("LIME")
plt.axis("off")

masker = shap.maskers.Image("inpaint_telea", image_np.shape)

explainer_shap = shap.Explainer(predict_fn, masker)

shap_values = explainer_shap(
    np.expand_dims(image_np, 0),
    max_evals=50,
    batch_size=50
)

shap.image_plot(shap_values)