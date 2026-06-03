import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval().to(device)
for param in vgg.parameters():
    param.requires_grad_(False)


def load_image(image, size=400):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(device)


def tensor_to_image(tensor):
    out = tensor.clone().detach().cpu()
    out = out * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = out + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    out = out.clamp(0, 1).squeeze(0)
    return transforms.ToPILImage()(out)


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


def style_transfer(content_image, style_image, steps=300, style_weight=1e6):
    content_img = load_image(content_image)
    style_img = load_image(style_image)
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
        total_loss = c_loss + style_weight * s_loss
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

    return tensor_to_image(target)


demo = gr.Interface(
    fn=style_transfer,
    inputs=[
        gr.Image(type="pil", label="Content Image"),
        gr.Image(type="pil", label="Style Image"),
        gr.Slider(100, 500, value=300, step=50, label="Steps"),
        gr.Slider(1e4, 1e7, value=1e6, label="Style Weight"),
    ],
    outputs=gr.Image(type="pil", label="Output"),
    title="Neural Style Transfer",
    description="Upload a content image and a style image to blend them together using VGG19.",
)

if __name__ == "__main__":
    demo.launch()
