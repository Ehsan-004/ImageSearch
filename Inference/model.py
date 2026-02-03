import torch
import torchvision
import torch.nn.functional as F


def get_model():
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT, progress=True)
    model.fc = torch.nn.Identity()
    return model


def get_transforms():
    pre_process = torchvision.models.ResNet152_Weights.DEFAULT.transforms()
    return pre_process


if __name__ == "__main__":
    model = get_model()
