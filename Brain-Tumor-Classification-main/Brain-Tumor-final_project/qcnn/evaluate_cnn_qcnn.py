import os
BASE_DIR = os.path.dirname(__file__)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from qcnn_model import QCNN


device = torch.device("cpu")

# ---------------- DATA ----------------
tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = ImageFolder(os.path.join(BASE_DIR, "dataset"), transform=tf)

class_names = dataset.classes

# SAME SPLIT AS TRAINING (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_ds = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# ---------------- MODEL ----------------
class HybridCNN_QCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
        self.cnn.fc = nn.Identity()   # 512 features

        self.qcnn = QCNN()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.qcnn(x)
        return self.classifier(x)

# ---------------- LOAD TRAINED MODEL ----------------
ckpt = torch.load(
    os.path.join(BASE_DIR, "cnn_qcnn.pth"),
    map_location=device
)


model = HybridCNN_QCNN(num_classes=len(class_names))
model.load_state_dict(ckpt["model"])
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
print("\nCLASSIFICATION REPORT (CNN + QCNN)\n")
print(classification_report(y_true, y_pred, target_names=class_names))

print("CONFUSION MATRIX\n")
print(confusion_matrix(y_true, y_pred))
