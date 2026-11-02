import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import gradio as gr
from PIL import Image

# ---------------- LOAD CHECKPOINT ----------------
ckpt = torch.load("cnn_model.pth", map_location="cpu")
class_names = ckpt["classes"]

# ---------------- MODEL ----------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
model.fc = nn.Linear(512, len(class_names))
model.load_state_dict(ckpt["model"])
model.eval()

# ---------------- TRANSFORMS ----------------
tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------- PREDICT ----------------
def predict(image):
    img = tf(image).unsqueeze(0)
    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)[0]

    idx = prob.argmax().item()
    return f"{class_names[idx]} (Confidence: {prob[idx]:.4f})"

# ---------------- UI ----------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Brain Tumor Classification (CNN Only)"
)

demo.launch()
