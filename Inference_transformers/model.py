from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from PIL import Image
import torch

# def get_model():
#     model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
#     return model

# def get_processor():
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
#     return processor


def get_model():
    # model = AutoModel.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("Trendyol/trendyol-dino-v2-ecommerce-256d", trust_remote_code=True)

    return model

def get_processor():
    # processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    processor = AutoImageProcessor.from_pretrained("Trendyol/trendyol-dino-v2-ecommerce-256d", trust_remote_code=True)

    return processor
