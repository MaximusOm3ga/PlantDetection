import os
import tempfile
import streamlit as st
from PIL import Image
from torchvision import transforms,datasets
import torch
import torch.nn.functional as f
from cnn_model import CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = r"C:\Users\sauri\Coding\NN\proj_code\datasets\plantvillage_DatasetNew\color"
transform=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

test_data = datasets.ImageFolder(root=data_path, transform = transform)

model=CNN(num_classes=len(test_data.classes)).to(device)
model.load_state_dict(torch.load("cnn_classifier_best.pth"))
model.eval()

def cnn_anom_det(path):
    img=Image.open(path).convert("RGB")
    img_tensor=transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs=model(img_tensor)
        probs=f.softmax(outputs,dim=1)[0]
        pred_idx=torch.argmax(probs).item()
        confidence=probs[pred_idx].item()

    predicted_class=test_data.classes[pred_idx]

    return predicted_class, confidence
st.set_page_config(page_title="Plant Detection")
st.title("Plant Disease Classifier")
st.write("Upload a leaf image to get the predicted class.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Running prediction..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            try:
                pred_class, confidence = cnn_anom_det(temp_path)
                st.success(f"Predicted class: {pred_class}")
                st.write(f"Confidence: {confidence:.2%}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
