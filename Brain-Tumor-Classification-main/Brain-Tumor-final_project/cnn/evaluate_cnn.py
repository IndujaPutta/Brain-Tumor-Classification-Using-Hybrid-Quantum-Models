import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(__file__)
device = torch.device("cpu")

tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = ImageFolder(os.path.join(BASE_DIR, "dataset"), transform=tf)
class_names = dataset.classes

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_ds = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
model.fc = nn.Linear(512, len(class_names))

ckpt = torch.load(os.path.join(BASE_DIR, "cnn_model.pth"), map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in val_loader:
        out = model(x)
        preds = out.argmax(1)
        y_true.extend(y.numpy())
        y_pred.extend(preds.numpy())

print("\nCLASSIFICATION REPORT (CNN ONLY)\n")
print(classification_report(y_true, y_pred, target_names=class_names))
print("CONFUSION MATRIX\n")
print(confusion_matrix(y_true, y_pred))
