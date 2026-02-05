import os
import torch
from torch.utils.data import DataLoader

from Inference.chroma_configs import get_cat_collection, get_client, get_pr_collection
from Inference.dataset import (
    ImageDataset, 
    collate_fn, 
    extract_batch_embeddings_collection
)
# from Inference.model import get_model, get_transforms, Embedder
from Inference.model import get_model, get_transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM = 1024


transforms = get_transforms()
model = get_model().to(DEVICE)


client = get_client()
products_collection = get_pr_collection(client)
categories_collection = get_cat_collection(client)


# preprocess_dataset("data/", save_dir="data/")
dataset = ImageDataset("data/data.parquet", transforms)

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=False,
    num_workers=os.cpu_count() - 2,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn,
    prefetch_factor=2
)


if __name__ == "__main__":
    print(f"Starting embedding on {DEVICE}...")
    extract_batch_embeddings_collection(loader, model, DEVICE, products_collection)
    print("Done!")
