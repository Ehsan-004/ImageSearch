import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T

def get_model():
    # استفاده از مدل DINOv2 نسخه Large که تعادل عالی بین دقت و سرعت دارد
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval()
    return model

def get_transforms():
    # استانداردهای DINOv2: ابعاد 224 در 224 با نرمالیزاسیون ImageNet
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
