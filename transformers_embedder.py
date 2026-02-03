import torch
from torch.utils.data import DataLoader

from Inference_transformers.chroma_config import get_cat_collection, get_client, get_pr_collection
from Inference_transformers.dataset import (
    ImageDataset, 
    collate_fn, 
    extract_batch_embeddings, 
    preprocess_dataset
)
# from Inference.model import get_model, get_transforms, Embedder
from Inference_transformers.model import get_model, get_processor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DB_STORE_PATH = "./database"
EMB_DIM = 768


# model
processor = get_processor()
model = get_model().to(DEVICE)


# chroma DB
client = get_client()
products_collection = get_pr_collection(client)
categories_collection = get_cat_collection(client)


preprocess_dataset("data/", save_dir="data/")
dataset = ImageDataset("data/data.parquet")

loader = DataLoader(
    dataset,
    batch_size=128,          # بسته به VRAM
    shuffle=False,
    num_workers=8,          # CPU cores
    pin_memory=True,        # مهم برای GPU
    persistent_workers=True,
    collate_fn=collate_fn
)

print("starting to embedd images ...")

all_embeddings, all_metadatas = extract_batch_embeddings(loader, model, processor,  DEVICE)

ids = [str(i) for i in range(1, all_embeddings.shape[0]+1)]
# ids = [str(uuid.uuid4()) for _ in range(all_embeddings.shape[0])]

print("adding embeddings to chroma database ...")
products_collection.add(
    embeddings=all_embeddings,
    metadatas=all_metadatas,
    ids=ids
)

print("done!")