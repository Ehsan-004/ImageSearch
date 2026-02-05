import torch
import torch.nn.functional as F
import torchvision.transforms as T

def get_model():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval()
    return model

def get_transforms():
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
