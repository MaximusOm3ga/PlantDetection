from visionTrans import ViTModel as vit

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision.models import ViT_B_16_Weights

data_path= r"C:\Users\sauri\PycharmProjects\PlantDetection\RunScripts\RunScripts\plantvillvalds\PlantVillage\train"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

criterion = nn.CrossEntropyLoss()

weights = ViT_B_16_Weights.DEFAULT
transform = weights.transforms()

dataset = datasets.ImageFolder(root=data_path, transform=transform)
labels = [label for _, label in dataset.samples]

train_idx, val_idx = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    stratify=labels
)

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

num_classes = len(dataset.classes)
print("Num classes:", num_classes)

model = vit(num_classes=num_classes).to(device)

for param in model.model.parameters():
    param.requires_grad = False

for param in model.model.heads.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=3e-4)

num_epochs = 20

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        val_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = val_loss / len(loader.dataset)
    acc = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, acc

def train(model, loader, val_loader, optimizer, epochs, device, criterion, checkpoint_path):
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total if total > 0 else 0.0

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved at epoch {epoch+1} with val acc {val_acc:.2f}%")

    print("Training complete. Best Val Acc: {:.2f}%".format(best_val_acc))

train(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    device,
    criterion,
    checkpoint_path="vit_best.pth"
)