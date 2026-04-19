import torch
from torchvision import transforms,datasets
from cnn_model import CNN
from PIL import Image
import torch.nn.functional as f
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report,confusion_matrix

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:",device)

test_path = "../datasets/plantvillage_DatasetNew/color"


transform=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


test_data=datasets.ImageFolder(root=test_path,transform=transform)
test_loader=DataLoader(test_data,batch_size=32, shuffle=False)

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


if __name__ =="__main__":
    y_true,y_pred=[],[]

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = f.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())


    print("Classification Report: ")
    print(classification_report(y_true,y_pred,target_names=test_data.classes))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))