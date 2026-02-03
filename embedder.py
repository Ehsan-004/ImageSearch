import os
import torch
from torch.utils.data import DataLoader

from Inference.chroma_configs import get_cat_collection, get_client, get_pr_collection
from Inference.dataset import (
    ImageDataset, 
    collate_fn, 
    extract_batch_embeddings,
    preprocess_dataset,
    extract_batch_embeddings_collection
)
# from Inference.model import get_model, get_transforms, Embedder
from Inference.model import get_model, get_transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM = 2048


transforms = get_transforms()
model = get_model().to(DEVICE)


client = get_client()
products_collection = get_pr_collection(client)
categories_collection = get_cat_collection(client)


# preprocess_dataset("data/", save_dir="data/")
dataset = ImageDataset("data/data.parquet", transforms)

loader = DataLoader(
    dataset,
    batch_size=256,          # بسته به VRAM
    shuffle=False,
    num_workers=os.cpu_count() - 2,          # CPU cores
    pin_memory=True,        # مهم برای GPU
    persistent_workers=True,
    collate_fn=collate_fn,
    prefetch_factor=2
)


print("starting to embedd images ...")
# all_embeddings, all_metadatas = extract_batch_embeddings(loader, model, DEVICE)
extract_batch_embeddings_collection(loader, model, DEVICE, products_collection)
print("done!")

# ids = [str(i) for i in range(1, all_embeddings.shape[0]+1)]
# ids = [str(uuid.uuid4()) for _ in range(all_embeddings.shape[0])]

# products_collection.add(
#     embeddings=all_embeddings,
#     metadatas=all_metadatas,
#     ids=ids
# )
