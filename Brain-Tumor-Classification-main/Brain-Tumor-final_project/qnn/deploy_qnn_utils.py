import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from qnn_model import QNNLayer

# ---------------- LOAD CHECKPOINT ----------------
ckpt = torch.load("qnn/cnn_qnn.pth", map_location="cpu")
class_names = ckpt["class_names"]

# ---------------- MODEL ----------------
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

model = CNN_QNN(len(class_names))
model.load_state_dict(ckpt["model_state"])
model.eval()

# ---------------- TRANSFORMS ----------------
tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------- PREDICT ----------------
def predict_qnn(image: Image.Image):
    img = tf(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]

    idx = probs.argmax().item()
    return class_names[idx], float(probs[idx])
