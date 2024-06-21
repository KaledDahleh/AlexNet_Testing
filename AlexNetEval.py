import torch
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.alexnet(pretrained=True)

with torch.inference_mode():
    pass