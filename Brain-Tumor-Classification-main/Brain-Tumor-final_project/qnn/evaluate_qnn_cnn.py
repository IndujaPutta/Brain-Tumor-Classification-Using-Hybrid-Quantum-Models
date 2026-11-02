import os
import torch

BASE_DIR = os.path.dirname(__file__)
device = torch.device("cpu")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from qnn_model import QNNLayer


device = torch.device("cpu")

# ---------------- LOAD DATA ----------------
tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = ImageFolder(os.path.join(BASE_DIR, "dataset"), transform=tf)

class_names = dataset.classes

# SAME split logic as training
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_ds = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# ---------------- MODEL DEFINITION ----------------
class CNN_QNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
        self.cnn.fc = nn.Identity()

        self.reduce = nn.Linear(512, 4)
        self.qnn = QNNLayer()
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.reduce(x)
        x = torch.tanh(x) * 3.1415
        x = self.qnn(x)
        return self.classifier(x)

# ---------------- LOAD TRAINED MODEL ----------------
ckpt = torch.load(
    os.path.join(BASE_DIR, "cnn_qnn.pth"),
    map_location=device
)



model = CNN_QNN(len(class_names))
model.load_state_dict(ckpt["model_state"])
model.eval()

# ---------------- EVALUATION ----------------
y_true = []
y_pred = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        outputs = model(x)
        preds = outputs.argmax(dim=1)

        y_true.extend(y.numpy())
        y_pred.extend(preds.numpy())

# ---------------- RESULTS ----------------
print("\n CLASSIFICATION REPORT (CNN + QNN)\n")
print(classification_report(y_true, y_pred, target_names=class_names))

print(" CONFUSION MATRIX\n")
print(confusion_matrix(y_true, y_pred))
