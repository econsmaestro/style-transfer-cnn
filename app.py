import os
import uuid
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from flask import Flask, request, render_template, send_from_directory
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
OUTPUT_FOLDER = os.path.join("static", "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval().to(device)
for param in vgg.parameters():
    param.requires_grad_(False)


def load_image(path, size=400):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(image).unsqueeze(0).to(device)


def save_image(tensor, path):
    out = tensor.clone().detach().cpu()
    out = out * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = out + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    out = out.clamp(0, 1).squeeze(0)
    transforms.ToPILImage()(out).save(path)


def gram_matrix(x):
    _, c, h, w = x.shape
    x = x.view(c, h * w)
    return torch.mm(x, x.t()) / (c * h * w)


def extract_features(image):
    layers = {"0": "conv1_1", "5": "conv2_1", "10": "conv3_1", "19": "conv4_1", "28": "conv5_1"}
    features = {}
    x = image
    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def run_style_transfer(content_path, style_path, steps=500):
    content_img = load_image(content_path)
    style_img = load_image(style_path)
    target = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([target], lr=0.003)

    content_features = extract_features(content_img)
    style_features = extract_features(style_img)

    for _ in range(steps):
        target_features = extract_features(target)
        c_loss = torch.mean((target_features["conv4_1"] - content_features["conv4_1"]) ** 2)
        s_loss = sum(
            torch.mean((gram_matrix(target_features[l]) - gram_matrix(style_features[l])) ** 2)
            for l in target_features
        )
        total_loss = c_loss + 1e6 * s_loss
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

    output_filename = f"{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    save_image(target, output_path)
    return output_filename


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/transfer", methods=["POST"])
def transfer():
    content_file = request.files.get("content")
    style_file = request.files.get("style")

    if not content_file or not style_file:
        return render_template("index.html", error="Please upload both images.")

    content_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.jpg")
    style_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.jpg")
    content_file.save(content_path)
    style_file.save(style_path)

    output_filename = run_style_transfer(content_path, style_path)
    return render_template("index.html", output=output_filename)


if __name__ == "__main__":
    app.run(debug=True)
